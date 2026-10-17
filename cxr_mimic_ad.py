"""MIMIC-CXR, unsupervised abnormality detection with the radiograph world model (trained on "No Finding" images only).

Each frontal image gets a score from its prediction-error map; normal = "No Finding" positive, abnormal = any other
finding positive (CheXpert-labeler labels, official split). Configurations (smoothing scale, sign, statistic, precision
weighting) are compared on the validation split; the chosen one is reported on the test split, overall and per
finding (that finding positive vs "No Finding").

    python cxr_mimic_ad.py <tag>
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from perr.cxr import CACHE, FINDINGS
from perr.cxr_wm import error_map, valid_region

WM = "/raid/home/CAMCA/hj880/perr/cache/cxr_wm"
TAG = "cxr"


def stats(W, V):
    v = W[V]
    q = np.sort(v)[::-1]
    n = max(1, len(q) // 100)
    return {"max": float(q[0]), "top1pct": float(q[:n].mean()), "top10pct": float(q[:10 * n].mean())}


def work(key):
    img = np.load(f"{CACHE}/{key}.npy")
    V = valid_region(img, margin=16)
    R = np.load(f"{WM}/{key}.{TAG}R.npy").astype(np.float32)
    S = np.load(f"{WM}/{key}.{TAG}S.npy").astype(np.float32)
    out = {"key": key}
    for sg in (2.0, 4.0, 8.0):
        for sign, nm in ((0, "abs"), (1, "pos"), (-1, "neg")):
            for st, v in stats(error_map(R, S, sg, sign), V).items():
                out[f"pw{sg:g}_{nm}_{st}"] = v
        for st, v in stats(np.abs(ndi.gaussian_filter(R, sg)), V).items():
            out[f"raw{sg:g}_abs_{st}"] = v
    X = img.astype(np.float32) / 255
    cols = np.nonzero(img.any(0))[0]
    mid = (cols.min() + cols.max()) / 2
    idx = np.clip(np.round(2 * mid - np.arange(X.shape[1])).astype(int), 0, X.shape[1] - 1)
    for st, v in stats(np.abs(ndi.gaussian_filter(X - X[:, idx], 4.0)), V).items():
        out[f"mirror_{st}"] = v
    return out


def auroc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


if __name__ == "__main__":
    TAG = sys.argv[1] if len(sys.argv) > 1 else TAG
    d = pd.read_csv("results/cxr_mimic.csv")
    d = d[d.split.isin(["validate", "test"])].copy()
    others = [f for f in FINDINGS if f != "No Finding"]
    d["abnormal"] = (d[others] == 1).any(axis=1).astype(int)
    d = d[(d["No Finding"] == 1) | (d.abnormal == 1)]
    with Pool(32) as p:
        sc = pd.DataFrame(p.map(work, d.key.tolist(), chunksize=8))
    d = d.merge(sc, on="key")
    d.to_csv(f"results/cxr_mimic_ad_{TAG}.csv", index=False)
    cfgs = [c for c in sc.columns if c != "key"]
    val, test = d[d.split == "validate"], d[d.split == "test"]
    res = sorted(((auroc(val[c], val.abnormal), c) for c in cfgs), reverse=True)
    print(f"validation: {len(val)} images ({val.abnormal.sum()} abnormal); test: {len(test)} ({test.abnormal.sum()} abnormal)")
    print("best on validation:", " | ".join(f"{c} {a:.3f}" for a, c in res[:6]))
    for fam in ("pw4_abs_top1pct", "raw4_abs_top1pct", "mirror_top1pct"):
        print(f"  validation {fam}: {auroc(val[fam], val.abnormal):.3f}")
    best = res[0][1]
    print(f"TEST, configuration chosen on validation ({best}): normal vs abnormal AUROC {auroc(test[best], test.abnormal):.3f}")
    for c in ("raw" + best[2:] if best.startswith("pw") else best, "mirror_top1pct"):
        if c in test:
            print(f"  test {c}: {auroc(test[c], test.abnormal):.3f}")
    nf = test[test["No Finding"] == 1]
    for f in others:
        pos = test[test[f] == 1]
        if len(pos) >= 20:
            sub = pd.concat([pos.assign(y=1), nf.assign(y=0)])
            print(f"  {f:28s} n={len(pos):4d}  AUROC vs No Finding {auroc(sub[best], sub.y):.3f}")
