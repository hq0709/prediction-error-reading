"""Where to enlarge: share of radiologist boxes whose centre lies inside one of K evidence views (squares of 0.2 of the
radiograph's longer side, the two-stream reader's crops), for view placements from the pixel world model (thorax
domain, smoothing sigma, sign, suppression radius), PatchCore, random thorax points and the six fixed lung places.

    python cxr_coverage.py nih
    python cxr_coverage.py cxdet train
"""
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from cxr_recall import boxes
from perr.cxr import CACHE
from perr.cxr_anat import search_domain
from perr.cxr_reader import grid_points
from perr.cxr_wm import error_map, peaks, valid_region

ROOT = "/raid/home/CAMCA/hj880/perr"
KS = (2, 4, 6, 8)


def placements(key):
    img = np.load(f"{CACHE}/{key}.npy")
    V = valid_region(img)
    D = search_domain(key, V)
    R = np.load(f"{ROOT}/cache/cxr_wm/{key}.cxrR.npy").astype(np.float32)
    S = np.load(f"{ROOT}/cache/cxr_wm/{key}.cxrS.npy").astype(np.float32)
    out = {"grid6": grid_points(img)}
    for sg in (8, 16):
        for sign, nm in ((1, "pos"), (0, "abs")):
            E = error_map(R, S, sg, sign)
            for nms in (40, 64):
                out[f"wm_s{sg}_{nm}_nms{nms}"] = peaks(E, D, k=8, nms=nms)
            out[f"wm_s{sg}_{nm}_radiograph"] = peaks(E, V, k=8, nms=40)
    pc = ndi.zoom(np.load(f"{ROOT}/cache/cxr_maps/patchcore/{key}.npy").astype(np.float32), 512 / 37, order=1)
    out["patchcore_nms64"] = peaks(pc, D, k=8, nms=64)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    out["random_nms64"] = peaks(rng.random((512, 512)).astype(np.float32), D, k=8, nms=64)
    rows, cols = np.nonzero(img.any(1))[0], np.nonzero(img.any(0))[0]
    side = 0.2 * max(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    return key, out, side


def covered(pts, side, box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    return [any(abs(c - cx) <= side / 2 and abs(r - cy) <= side / 2 for r, c, _ in pts[:k]) for k in KS]


def main():
    which = sys.argv[1]
    b = boxes(which)
    if len(sys.argv) > 2:
        b = b[b.split == sys.argv[2]].reset_index(drop=True)
    with Pool(48) as p:
        res = {k: (o, s) for k, o, s in p.map(placements, sorted(b.key.unique()), chunksize=4)}
    names = list(next(iter(res.values()))[0])
    rows = []
    for n in names:
        c = np.array([covered(res[r.key][0][n], res[r.key][1], (r.x, r.y, r.w, r.h)) for r in b.itertuples()])
        rows.append({"placement": n, **{f"K={k}": c[:, i].mean() for i, k in enumerate(KS)}})
    print(f"{len(b)} boxes on {b.key.nunique()} images")
    print(pd.DataFrame(rows).sort_values("K=6", ascending=False).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
