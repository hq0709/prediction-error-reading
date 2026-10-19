"""Which NIH ChestX-ray14 image (and patient) each ChestX-Det image is: ChestX-Det renames its NIH images, so they are
matched by content (32 x 32 block means of the 512-canvas image, correlation with every NIH image).

    python match_cxdet_nih.py
Writes results/cxr_cxdet_nih_match.csv (cxdet key, NIH image, patient, correlation, runner-up correlation).
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from perr.cxr import NIH, letterbox, load

ROOT = "/raid/home/CAMCA/hj880/perr"


def sig_arr(a):
    x = a.astype(np.float32).reshape(32, 16, 32, 16).mean((1, 3)).ravel()
    x -= x.mean()
    return x / (np.linalg.norm(x) + 1e-6)


def nih_sig(name):
    p = f"{ROOT}/../perr/cache/cxr/nih/{name[:-4]}.npy"
    try:
        a = np.load(p) if os.path.exists(p) else letterbox(Image.open(f"{NIH}/png/images/{name}"))[0]
        return sig_arr(a)
    except Exception:
        return np.zeros(1024, np.float32)


def main():
    d = pd.read_csv(f"{NIH}/Data_Entry_2017_v2020.csv")
    names = d["Image Index"].tolist()
    cx = pd.read_csv(f"{ROOT}/results/cxr_cxdet.csv")
    with Pool(48) as p:
        A = np.stack(p.map(nih_sig, names, chunksize=256))
        B = np.stack([sig_arr(load(k)) for k in cx.key])
    S = B @ A.T
    top2 = np.argsort(-S, 1)[:, :2]
    c1 = S[np.arange(len(B)), top2[:, 0]]
    c2 = S[np.arange(len(B)), top2[:, 1]]
    out = pd.DataFrame({"key": cx.key, "split": cx.split, "nih_image": [names[i] for i in top2[:, 0]],
                        "patient": [names[i][:8] for i in top2[:, 0]], "corr": c1, "runner_up": c2,
                        "nih_labels": [d["Finding Labels"].iloc[i] for i in top2[:, 0]]})
    out.to_csv(f"{ROOT}/results/cxr_cxdet_nih_match.csv", index=False)
    print("best correlation quantiles 1/10/50%:", np.quantile(c1, [0.01, 0.1, 0.5]).round(4),
          "runner-up median:", np.median(c2).round(4), "matched (> 0.99):", int((c1 > 0.99).sum()), "of", len(c1))
    print(out[out["corr"] > 0.99].nih_labels.value_counts().head(8))


if __name__ == "__main__":
    main()
