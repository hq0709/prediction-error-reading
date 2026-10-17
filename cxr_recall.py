"""Chest radiographs, NIH ChestX-ray14 boxes (984 boxes, 880 images, 8 findings): how often the world model's top-K
addresses point inside a ground-truth box (pointing game), per finding.

World models compared: the learned radiograph model (precision-weighted error, both signs or one), a mechanical
"left-right symmetry" model (each pixel expected to equal its mirror image across the radiograph's midline), the
brightest points, and the chance level of a uniformly random point inside the radiograph.

    python cxr_recall.py <tag>
"""
import json
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from perr.cxr import CACHE
from perr.cxr_wm import error_map, peaks, valid_region

WM = "/raid/home/CAMCA/hj880/perr/cache/cxr_wm"
TAG = "cxr"
KS = (1, 3, 5, 10, 20)


def boxes(which="nih"):
    """Boxes of a set (nih: ChestX-ray14's 984; cxdet: ChestX-Det, build_cxdet.py) on the 512 canvas."""
    b = pd.read_csv(f"results/cxr_{which}_boxes.csv")
    g = pd.read_csv(f"results/cxr_{which}.csv").set_index("image")
    b["key"] = [g.loc[i, "key"] for i in b.image]
    for c in ("x", "y", "w", "h"):
        b[c] = b[c] * [g.loc[i, "scale"] for i in b.image]
    b["x"] += [g.loc[i, "pad_x"] for i in b.image]
    b["y"] += [g.loc[i, "pad_y"] for i in b.image]
    return b


def first_hit(pk, box):
    x, y, w, h = box
    return next((k + 1 for k, (r, c, _) in enumerate(pk) if x <= c <= x + w and y <= r <= y + h), None)


def work(key):
    img = np.load(f"{CACHE}/{key}.npy")
    V = valid_region(img)
    X = img.astype(np.float32) / 255
    out = {"key": key}
    R = np.load(f"{WM}/{key}.{TAG}R.npy").astype(np.float32)
    S = np.load(f"{WM}/{key}.{TAG}S.npy").astype(np.float32)
    for sg in (2.0, 4.0, 8.0):
        for sign, nm in ((0, "abs"), (1, "pos"), (-1, "neg")):
            out[f"learned{sg:g}_{nm}"] = peaks(error_map(R, S, sg, sign), V)
        out[f"raw{sg:g}_abs"] = peaks(np.abs(ndi.gaussian_filter(R, sg)), V)
    rows, cols = np.nonzero(img.any(1))[0], np.nonzero(img.any(0))[0]
    mid = (cols.min() + cols.max()) / 2
    mirror = np.zeros_like(X)
    for c in range(X.shape[1]):
        m = int(round(2 * mid - c))
        if 0 <= m < X.shape[1]:
            mirror[:, c] = X[:, m]
    out["mirror_abs"] = peaks(np.abs(ndi.gaussian_filter(X - mirror, 4.0)), V)
    out["bright"] = peaks(ndi.gaussian_filter(X, 4.0), V)
    out["valid_area"] = int(V.sum())
    return out


if __name__ == "__main__":
    TAG = sys.argv[1] if len(sys.argv) > 1 else TAG
    b = boxes()
    with Pool(32) as p:
        res = {r["key"]: r for r in p.map(work, sorted(b.key.unique()), chunksize=4)}
    names = [k for k in next(iter(res.values())) if k not in ("key", "valid_area")]
    rows = []
    for name in names:
        f = [first_hit(res[r.key][name], (r.x, r.y, r.w, r.h)) for r in b.itertuples()]
        rows.append([name] + [np.mean([x is not None and x <= K for x in f]) for K in KS])
    chance = np.mean([(r.w * r.h) / res[r.key]["valid_area"] for r in b.itertuples()])
    print(f"{len(b)} boxes on {b.key.nunique()} images; chance of one random point hitting: {chance:.3f}")
    for r in sorted(rows, key=lambda r: -r[3]):
        print(f"{r[0]:18s}", " ".join(f"K={K}:{v:.2f}" for K, v in zip(KS, r[1:])))
    best = max(rows, key=lambda r: r[3])[0]
    print(f"per finding ({best}):")
    for fnd, g in b.groupby("finding"):
        f = [first_hit(res[r.key][best], (r.x, r.y, r.w, r.h)) for r in g.itertuples()]
        print(f"  {fnd:14s} n={len(g):3d}", " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in f]):.2f}" for K in KS))
    json.dump({k: {n: v for n, v in r.items() if n != "key"} for k, r in res.items()}, open(f"results/cxr_recall_{TAG}.json", "w"))
