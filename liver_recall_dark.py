"""Liver ablation without a world model: the darkest points of the (eroded) liver, smoothed at 1.5 mm, same NMS."""
import json
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

from perr.liver import cases, has_target, label_at, load_liver


def darkest(s, k=30, nms_mm=15.0):
    M = np.asarray(s.lung) > 0
    sp = np.array(s.spacing, float)
    r = [max(1, int(round(3 / x))) for x in sp]
    core = ndi.binary_erosion(M, structure=np.ones([2 * q + 1 for q in r]))
    Vs = ndi.gaussian_filter(np.clip(np.asarray(s.vol).astype(np.float32), -200, 300), 1.5 / sp)
    W = np.where(core, 301.0 - Vs, 0).astype(np.float32)
    rr = [max(1, int(np.ceil(nms_mm / x))) for x in sp]
    out = []
    for _ in range(k):
        i = int(np.argmax(W)); z, y, x = np.unravel_index(i, W.shape); v = float(W[z, y, x])
        if v <= 0:
            break
        out.append((z, y, x, v))
        W[max(0, z - rr[0]):z + rr[0] + 1, max(0, y - rr[1]):y + rr[1] + 1, max(0, x - rr[2]):x + rr[2] + 1] = 0
    return out


def work(case):
    s = load_liver(case)
    if not has_target(s):
        return None
    pk = darkest(s)
    return next((k + 1 for k, p in enumerate(pk) if label_at(case, p[:3]) == "big"), None)


if __name__ == "__main__":
    with Pool(24) as p:
        res = [r for r in p.map(work, cases(), chunksize=1)]
    t = [r for r, c in zip(res, cases()) if load_liver(c) and has_target(load_liver(c))]
    print("darkest points:", " ".join(f"K={k}:{np.mean([x is not None and x <= k for x in t]):.2f}" for k in (1, 5, 10, 20, 30)), f"(n={len(t)})")
