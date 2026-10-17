"""MIMIC-CXR disease course, world models alone: does a pair's prediction-error map tell pairs with a new finding from
pairs without one? Scores per pair (top 1% of the smoothed error inside the radiograph):
  course   the prior-conditioned world model's precision-weighted error (sign chosen on validation);
  anatomy  the single-image world model's precision-weighted error on the current image;
  nochange |current - registered prior|, smoothed.
AUROC for "any new finding" and per finding (new vs not new), on the test pairs; configuration chosen on validation.

    python cxr_course_eval.py <course tag> <anatomy tag>
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from perr.cxr import CACHE
from perr.cxr_course import LABELS, REG
from perr.cxr_wm import error_map, valid_region

WM = "/raid/home/CAMCA/hj880/perr/cache/cxr_wm"
CTAG, ATAG = "cxrc", "cxr"


def top1(W, V):
    v = np.sort(W[V])[::-1]
    return float(v[:max(1, len(v) // 100)].mean())


def work(key):
    img = np.load(f"{CACHE}/{key}.npy")
    V = valid_region(img, margin=16)
    out = {"key": key}
    R = np.load(f"{WM}/{key}.{CTAG}R.npy").astype(np.float32)
    S = np.load(f"{WM}/{key}.{CTAG}S.npy").astype(np.float32)
    for sg in (4.0, 8.0):
        for sign, nm in ((0, "abs"), (1, "pos"), (-1, "neg")):
            out[f"course{sg:g}_{nm}"] = top1(error_map(R, S, sg, sign), V)
    R = np.load(f"{WM}/{key}.{ATAG}R.npy").astype(np.float32)
    S = np.load(f"{WM}/{key}.{ATAG}S.npy").astype(np.float32)
    out["anatomy4_pos"] = top1(error_map(R, S, 4.0, 1), V)
    pri = np.load(f"{REG}/{key}.npy").astype(np.float32)
    out["nochange4"] = top1(np.abs(ndi.gaussian_filter(img.astype(np.float32) - pri, 4.0)), V & (pri > 0))
    return out


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


if __name__ == "__main__":
    CTAG, ATAG = sys.argv[1], sys.argv[2]
    d = pd.read_csv("results/cxr_pairs.csv")
    d = d[d.split.isin(["validate", "test"])].copy()
    d["any_new"] = (d[[f"new_{f}" for f in LABELS]] == 1).any(axis=1).astype(int)
    with Pool(32) as p:
        sc = pd.DataFrame(p.map(work, d.key.tolist(), chunksize=8))
    d = d.merge(sc, on="key")
    d.to_csv(f"results/cxr_course_eval_{CTAG}.csv", index=False)
    cfg = [c for c in sc.columns if c.startswith("course")]
    val, test = d[d.split == "validate"], d[d.split == "test"]
    best = max(cfg, key=lambda c: auroc(val[c], val.any_new))
    print(f"validation pairs {len(val)} ({val.any_new.sum()} with a new finding); test {len(test)} ({test.any_new.sum()})")
    print("validation:", " | ".join(f"{c} {auroc(val[c], val.any_new):.3f}" for c in cfg + ["anatomy4_pos", "nochange4"]))
    print(f"TEST any new finding: course ({best}) {auroc(test[best], test.any_new):.3f} | anatomy {auroc(test.anatomy4_pos, test.any_new):.3f}"
          f" | no change {auroc(test.nochange4, test.any_new):.3f}")
    for f in LABELS:
        t = test[test[f"new_{f}"].notna()]
        if (t[f"new_{f}"] == 1).sum() >= 20:
            y = t[f"new_{f}"]
            print(f"  {f:28s} n_new={int(y.sum()):4d}  course {auroc(t[best], y):.3f}  anatomy {auroc(t.anatomy4_pos, y):.3f}  no change {auroc(t.nochange4, y):.3f}")
