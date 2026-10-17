"""Pointing at radiologist boxes with every address proposer, inside a chosen search domain.

Proposers: the pixel world model (precision-weighted error, smoothing sigma, sign), the multi-scale latent world model
(per scale and summed; raw and precision-weighted), PatchCore, the brightest points, and uniformly random points.
Domains: the radiograph (letterbox padding removed, the old protocol) or the thorax (perr.cxr_anat.domain).
A box is hit at K if one of the top-K addresses (square suppression) lies inside it.

    python cxr_point.py nih
    python cxr_point.py cxdet test        # ChestX-Det test split
    python cxr_point.py nih --scales cxr,cxr2,cxr4     # multi-scale world model (normalisers from cxr_ad.py)
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from cxr_recall import KS, boxes, first_hit
from perr.cxr import CACHE
from perr.cxr_anat import search_domain
from perr.cxr_wm import error_map, peaks, valid_region

ROOT = "/raid/home/CAMCA/hj880/perr"
MAPS = f"{ROOT}/cache/cxr_maps"
WM = f"{ROOT}/cache/cxr_wm"
DOMAINS = {"radiograph": None, "hull": ("hull", 0), "hull+8": ("hull", 8), "organs+8": ("organs", 8),
           "organs+16": ("organs", 16)}
NMS = 24
NORMS = {}
EXTRA = ()


def up(a):
    a = np.asarray(a, np.float32)
    return ndi.zoom(a, 512 / a.shape[-1], order=1)


SCALES = ("cxr",)                     # pixel world models read (tags of infer_cxr_wm.py); cxr2 / cxr4: 2x / 4x coarser


def maps(key):
    """name -> 512 x 512 address map."""
    out = {}
    for sg in (4, 8, 16, 24):
        for sign, nm in ((0, "abs"), (1, "pos")):
            parts = []
            for t in SCALES:
                R = np.load(f"{WM}/{key}.{t}R.npy").astype(np.float32)
                S = np.load(f"{WM}/{key}.{t}S.npy").astype(np.float32)
                E = error_map(R, S, sg, sign)
                out[f"pixel_{t}_s{sg}_{nm}" if len(SCALES) > 1 else f"pixel_s{sg}_{nm}"] = E
                if len(SCALES) > 1 and f"{t}_s{sg}_{nm}|thorax" in NORMS:
                    parts.append(E / NORMS[f"{t}_s{sg}_{nm}|thorax"])
            if len(parts) == len(SCALES) > 1:
                out[f"pixel_max_s{sg}_{nm}"] = np.max(parts, 0)
    lat = np.load(f"{MAPS}/latent_ms/{key}.npy").astype(np.float32)               # (3, 2, G, G)
    for j, nm in enumerate(("raw", "pw")):
        for si, sc in enumerate((3, 6, 12)):
            out[f"latent_{nm}{sc}"] = up(lat[si, j])
        out[f"latent_{nm}_sum"] = up(sum(lat[si, j] / np.median(lat[si, j]) for si in range(3)))
    out["patchcore"] = up(np.load(f"{MAPS}/patchcore/{key}.npy"))
    for name in EXTRA:                                  # further 37 x 37 maps in cache/cxr_maps/<name>
        out[name] = up(np.load(f"{MAPS}/{name}/{key}.npy"))
    img = np.load(f"{CACHE}/{key}.npy").astype(np.float32) / 255
    out["bright"] = ndi.gaussian_filter(img, 4.0)
    return out


def work(key):
    img = np.load(f"{CACHE}/{key}.npy")
    V = valid_region(img)
    doms = {d: V if spec is None else search_domain(key, V, *spec) for d, spec in DOMAINS.items()}
    res = {"key": key, "area": {d: int(m.sum()) for d, m in doms.items()}}
    rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
    for name, W in maps(key).items():
        for d, D in doms.items():
            res[(name, d)] = peaks(W, D, k=20, nms=NMS)
    for d, D in doms.items():                                                   # random points, 10 draws
        res[("random", d)] = [peaks(rng.random((512, 512)).astype(np.float32), D, k=20, nms=NMS) for _ in range(10)]
    return res


def hit_rates(b, res, name, d):
    if name == "random":
        out = []
        for j in range(10):
            f = [first_hit(res[r.key][(name, d)][j], (r.x, r.y, r.w, r.h)) for r in b.itertuples()]
            out.append([np.mean([x is not None and x <= K for x in f]) for K in KS])
        return list(np.mean(out, 0))
    f = [first_hit(res[r.key][(name, d)], (r.x, r.y, r.w, r.h)) for r in b.itertuples()]
    return [np.mean([x is not None and x <= K for x in f]) for K in KS]


def main():
    global SCALES, NORMS, EXTRA
    if "--extra" in sys.argv:
        i = sys.argv.index("--extra")
        EXTRA = tuple(sys.argv[i + 1].split(","))
        del sys.argv[i:i + 2]
    if "--scales" in sys.argv:
        i = sys.argv.index("--scales")
        SCALES = tuple(sys.argv[i + 1].split(","))
        del sys.argv[i:i + 2]
        NORMS = json.load(open("results/cxr_scale_norms.json"))
    which = sys.argv[1] if len(sys.argv) > 1 else "nih"
    b = boxes(which)
    if len(sys.argv) > 2:
        b = b[b.split == sys.argv[2]].reset_index(drop=True)
        which = f"{which}_{sys.argv[2]}"
    with Pool(48) as p:
        res = {r["key"]: r for r in p.map(work, sorted(b.key.unique()), chunksize=2)}
    names = [k[0] for k in next(iter(res.values())) if isinstance(k, tuple) and k[1] == "radiograph"]
    rows = []
    for d in DOMAINS:
        for name in names:
            rows.append({"domain": d, "proposer": name, **{f"K={K}": v for K, v in zip(KS, hit_rates(b, res, name, d))}})
    df = pd.DataFrame(rows)
    df.to_csv(f"results/cxr_point_{which}.csv", index=False)
    pd.set_option("display.width", 200)
    for d in DOMAINS:
        print(f"--- domain {d}")
        print(df[df.domain == d].sort_values("K=10", ascending=False).head(12).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
