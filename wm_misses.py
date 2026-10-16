"""Where the shape world model misses, measured on the training pool only (no evaluation scan is read).

For every nodule >= 6 mm in a pool scan: is its centre inside the lung domain, how deep (mm from the nearest
non-lung voxel), the largest shape error within its radius, and the rank of the first shape peak (top 50) that
hits it. A nodule is juxtapleural when its centre lies within (radius + 2 mm) of non-lung tissue.

    python wm_misses.py
"""
import json
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from perr.score import MIN_DIAM, TOL_MM, _mm
from perr.study import load_luna
from perr.worldmodels import ShapePrior, lung_domain


def work(uid):
    s = load_luna(uid)
    big = [n for n in s.nodules if n.diameter_mm >= MIN_DIAM]
    if not big:
        return []
    sp = np.array(s.spacing, float)
    dom = np.asarray(lung_domain(s)) > 0
    depth = ndimage.distance_transform_edt(dom, sampling=sp)
    R = np.asarray(ShapePrior().error_volume(s)).astype(np.float32)
    peaks = ShapePrior().top(s, 50)
    out = []
    for n in big:
        c = np.array(n.zyx, float)
        ci = tuple(int(round(v)) for v in c)
        r = n.diameter_mm / 2
        lo = [max(0, int(np.floor(c[i] - r / sp[i]))) for i in range(3)]
        hi = [min(s.shape[i], int(np.ceil(c[i] + r / sp[i])) + 1) for i in range(3)]
        g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing="ij"), -1)
        ball = np.linalg.norm((g - c) * sp, axis=-1) <= r
        sub = tuple(slice(lo[i], hi[i]) for i in range(3))
        rank = next((k + 1 for k, (z, y, x, _) in enumerate(peaks)
                     if np.linalg.norm(_mm((z, y, x), sp) - _mm(c, sp)) <= r + TOL_MM), None)
        out.append({"uid": uid, "diam": n.diameter_mm, "in_lung": bool(np.asarray(s.lung)[ci]), "in_domain": bool(dom[ci]),
                    "depth_mm": float(depth[ci]), "juxta": bool(depth[ci] <= r + 2.0),
                    "frac_in_domain": float(dom[sub][ball].mean()), "maxR": float(R[sub][ball].max()),
                    "peak10": float(peaks[min(9, len(peaks) - 1)][3]) if peaks else 0.0, "rank": rank})
    return out


if __name__ == "__main__":
    uids = open("results/train_uids.txt").read().split()
    with Pool(48) as p:
        rows = [r for rs in p.map(work, uids, chunksize=1) for r in rs]
    json.dump(rows, open("results/wm_misses_pool.json", "w"))
    by = {}
    for r in rows:
        by.setdefault(r["uid"], []).append(r)
    first = [min((r["rank"] for r in rs if r["rank"]), default=None) for rs in by.values()]
    print(f"pool scans with a nodule >= 6 mm: {len(by)}; nodules: {len(rows)}")
    print("scan recall@K", " ".join(f"{k}:{np.mean([f is not None and f <= k for f in first]):.2f}" for k in (1, 5, 10, 20, 30, 50)))
    for name, sel in (("found (rank <= 10)", lambda r: r["rank"] and r["rank"] <= 10),
                      ("missed (rank > 10)", lambda r: not (r["rank"] and r["rank"] <= 10))):
        g = [r for r in rows if sel(r)]
        print(f"{name}: n={len(g)} diam median {np.median([r['diam'] for r in g]):.1f}  juxta {np.mean([r['juxta'] for r in g]):.2f}  "
              f"centre in lung {np.mean([r['in_lung'] for r in g]):.2f}  in domain {np.mean([r['in_domain'] for r in g]):.2f}  "
              f"frac in domain {np.mean([r['frac_in_domain'] for r in g]):.2f}  maxR/peak10 median "
              f"{np.median([r['maxR'] / max(r['peak10'], 1e-6) for r in g]):.2f}  rank<=50 {np.mean([bool(r['rank']) for r in g]):.2f}")
    m = [r for r in rows if not (r["rank"] and r["rank"] <= 10)]
    j = [r for r in m if r["juxta"]]
    print(f"missed juxtapleural: {len(j)}; of these centre outside domain {np.mean([not r['in_domain'] for r in j]):.2f}, "
          f"maxR == 0 {np.mean([r['maxR'] == 0 for r in j]):.2f}")
    print(f"missed small (< 8 mm): {sum(r['diam'] < 8 for r in m)}; missed non-juxta >= 8 mm: {sum(r['diam'] >= 8 and not r['juxta'] for r in m)}")
