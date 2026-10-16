"""How often the top-K addresses of a proposal contain a nodule >= 6 mm: shape world model vs densest points.

    python recall_curve.py 60 260
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from perr.score import score_peaks
from perr.study import load_luna, pilot_cases
from perr.worldmodels import IntensityPeaks, ShapePrior, lung_domain

KMAX = 50


def nms(W, spacing, k, offset=(0, 0, 0), nms_mm=10.0):
    r = [max(1, int(np.ceil(nms_mm / s))) for s in spacing]
    out = []
    for _ in range(k):
        i = int(np.argmax(W))
        z, y, x = np.unravel_index(i, W.shape)
        v = float(W[z, y, x])
        if v <= 0:
            break
        out.append((z + offset[0], y + offset[1], x + offset[2], v))
        W[max(0, z - r[0]):z + r[0] + 1, max(0, y - r[1]):y + r[1] + 1, max(0, x - r[2]):x + r[2] + 1] = 0
    return out


def work(uid):
    s = load_luna(uid)
    shape = nms(np.asarray(ShapePrior().error_volume(s)).astype(np.float32), s.spacing, KMAX)
    ip = IntensityPeaks()
    dom = np.asarray(lung_domain(s)) > 0
    V = ndimage.gaussian_filter(np.clip(np.asarray(s.vol).astype(np.float32), -1000, ip.cap), ip.sigma / np.array(s.spacing))
    dense = nms(np.where(dom, V + 1001.0, 0.0).astype(np.float32), s.spacing, KMAX)
    first = lambda pk: next((k + 1 for k in range(len(pk)) if score_peaks(s, pk[k:k + 1], 1)), None)
    return {"uid": uid, "shape_first": first(shape), "dense_first": first(dense)}


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    with Pool(24) as p:
        res = p.map(work, pilot_cases(hi)[lo:hi], chunksize=1)
    json.dump(res, open(f"results/recall_curve_{lo}_{hi}.json", "w"))
    for name in ("shape", "dense"):
        f = [r[f"{name}_first"] for r in res]
        print(name, " ".join(f"K={k}:{np.mean([x is not None and x <= k for x in f]):.2f}" for k in (1, 5, 10, 20, 30, 50)))
