"""Precompute the radiograph world model's precision-weighted error map for the DenseNet reader's "wm" channel:
Z = smoothed (observed - expected) / smoothed predicted s.d. (sigma 4 px), signed, clipped to +-4 and stored as uint8
((Z + 4) / 8 * 255) in cache/cxr_z/<key>.npy, for the 60k training images and every validation and test image.

    python make_cxr_zmaps.py
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from perr.cxr_wm import error_map

WM = "/raid/home/CAMCA/hj880/perr/cache/cxr_wm"
OUT = "/raid/home/CAMCA/hj880/perr/cache/cxr_z"


def work(key):
    out = f"{OUT}/{key}.npy"
    if os.path.exists(out):
        return 1
    try:
        R = np.load(f"{WM}/{key}.cxrR.npy").astype(np.float32)
        S = np.load(f"{WM}/{key}.cxrS.npy").astype(np.float32)
    except FileNotFoundError:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    z = np.clip(error_map(R, S, 4.0, sign=1), -4, 4)
    np.save(out, np.round((z + 4) / 8 * 255).astype(np.uint8))
    return 1


if __name__ == "__main__":
    keys = open("results/cxr_train60k_keys.txt").read().split()
    d = pd.read_csv("results/cxr_mimic.csv")
    keys += d[d.split.isin(["validate", "test"])].key.tolist()
    with Pool(48) as p:
        ok = p.map(work, keys, chunksize=32)
    print(f"{sum(ok)} of {len(keys)} error maps written or present")
