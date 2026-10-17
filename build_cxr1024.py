"""The canonical grid at 1024 (perr/cxr.py letterbox with size 1024) for the high-resolution radiograph world model:
cache/cxr1024/<key>.npy for 40,000 "No Finding" MIMIC training images (world-model training), MIMIC validation and test
images, the NIH box images and ChestX-Det. Point (r, c) on this grid is (r / 2, c / 2) on the 512 grid (within 1 px).

    python build_cxr1024.py
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from perr.cxr import letterbox, mimic_frontal

OUT = "/raid/home/CAMCA/hj880/perr/cache/cxr1024"


def work(item):
    key, path = item
    out = f"{OUT}/{key}.npy"
    if os.path.exists(out):
        return 1
    try:
        arr, _, _ = letterbox(Image.open(path), 1024)
    except (FileNotFoundError, OSError):
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, arr)
    return 1


def main():
    m = mimic_frontal()
    m["key"] = "mimic/" + m.dicom_id
    d = pd.read_csv("results/cxr_mimic.csv")
    nf = d[(d.split == "train") & (d["No Finding"] == 1)].key.sample(n=40000, random_state=0)
    keep = set(nf) | set(d[d.split.isin(["validate", "test"])].key)
    items = list(zip(m[m.key.isin(keep)].key, m[m.key.isin(keep)].path))
    nih = pd.read_csv("results/cxr_nih.csv")
    items += list(zip(nih.key, nih.path))
    cx = pd.read_csv("results/cxr_cxdet.csv")
    items += [(k, f"/raid/home/CAMCA/hj880/data_ext/chestx_det/images/{s}/{i}") for k, s, i in zip(cx.key, cx.split, cx.image)]
    open("runs_wm/cxr1024_train_keys.txt", "w").write("\n".join(nf))
    with Pool(64) as p:
        ok = p.map(work, items, chunksize=16)
    print(f"{sum(ok)} of {len(items)} images cached at 1024", flush=True)


if __name__ == "__main__":
    main()
