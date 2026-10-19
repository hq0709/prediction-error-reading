"""NIH ChestX-ray14 radiographs whose labels say "No Finding", the training set of a radiograph world model for the
NIH site (train_cxr_wm.py --table): no image of a patient with an NIH box (the NIH pointing test) or with a ChestX-Det
image (ChestX-Det renames its NIH images; match_cxdet_nih.py finds each one's NIH image and patient by content), and,
as a check, no image whose 32 x 32 block means correlate above 0.98 with a ChestX-Det image. 2% of patients are held
out for validation.

    python build_nih_normals.py
Writes cache/cxr/nih/<image>.npy and results/cxr_nih_normals.csv (key, path, patient, split, geometry).
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from build_cxr import _build
from perr.cxr import NIH, load

ROOT = "/raid/home/CAMCA/hj880/perr"


def signature(key):
    x = load(key).astype(np.float32).reshape(32, 16, 32, 16).mean((1, 3)).ravel()
    x -= x.mean()
    return x / (np.linalg.norm(x) + 1e-6)


def main():
    d = pd.read_csv(f"{NIH}/Data_Entry_2017_v2020.csv")
    boxed = {i[:8] for i in pd.read_csv(f"{NIH}/BBox_List_2017.csv")["Image Index"]}
    n = d[d["Finding Labels"] == "No Finding"].copy()
    n["patient"] = n["Image Index"].str[:8]
    print(f"{len(n)} 'No Finding' images; {n.patient.isin(boxed).sum()} of patients with an NIH box left out", flush=True)
    cxp = set(pd.read_csv(f"{ROOT}/results/cxr_cxdet_nih_match.csv").patient.astype(str).str.zfill(8))
    print(f"{(~n.patient.isin(boxed) & n.patient.isin(cxp)).sum()} more of patients with a ChestX-Det image left out", flush=True)
    n = n[~n.patient.isin(boxed) & ~n.patient.isin(cxp)]
    n["key"] = "nih/" + n["Image Index"].str[:-4]
    n["path"] = f"{NIH}/png/images/" + n["Image Index"]
    with Pool(32) as p:
        geo = p.map(_build, list(zip(n.key, n.path)), chunksize=64)
    g = pd.DataFrame(geo)
    if "error" in g:
        print("unreadable:", g.error.notna().sum(), flush=True)
        g = g[g.error.isna()].drop(columns="error")
    n = n.merge(g, on="key")
    cx = pd.read_csv(f"{ROOT}/results/cxr_cxdet.csv")
    with Pool(32) as p:
        A = np.stack(p.map(signature, n.key, chunksize=256))
        B = np.stack(p.map(signature, cx.key, chunksize=64))
    best = np.concatenate([(A[i:i + 4096] @ B.T).max(1) for i in range(0, len(A), 4096)])
    print("nearest ChestX-Det correlation, quantiles 50/99/99.9%:", np.quantile(best, [0.5, 0.99, 0.999]).round(4),
          "; above 0.99:", int((best > 0.99).sum()), "above 0.98:", int((best > 0.98).sum()), flush=True)
    n["cxdet_corr"] = best
    n = n[n.cxdet_corr <= 0.98]
    pts = sorted(n.patient.unique())
    rng = np.random.default_rng(0)
    val = set(rng.choice(pts, size=len(pts) // 50, replace=False))
    n["split"] = ["validate" if p in val else "train" for p in n.patient]
    cols = ["key", "Image Index", "path", "patient", "split", "w", "h", "scale", "pad_x", "pad_y", "cxdet_corr"]
    n[cols].to_csv(f"{ROOT}/results/cxr_nih_normals.csv", index=False)
    print(f"{(n.split == 'train').sum()} training and {(n.split == 'validate').sum()} validation images, "
          f"{len(pts)} patients", flush=True)


if __name__ == "__main__":
    main()
