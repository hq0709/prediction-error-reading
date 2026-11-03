"""Focal-box recall@K of an anomaly map's candidates on the verifier's held-out tenth of ChestX-Det train (302 images,
165 Nodule / Mass / Calcification / Fracture boxes), for choosing a baseline's free inference settings.

Candidates are exactly those of `cxr_focal_data.py cxdet --proposer map:<name> --k 45`: the map (float16 on disk, as
cache/cxr_maps stores it) resized to 512 with ndi.zoom(order=1), top-45 peaks with 16 px square suppression inside the
thorax search mask (perr.cxr_anat.search_domain of perr.cxr_wm.valid_region). A box is recalled when one of the top-K
candidates lies inside it (the label rule of cxr_focal_data.py). Boxes: cxr_recall.boxes("cxdet").

Each --maps entry is DIR[:POST,POST,...]; POST is applied to the loaded map before peak finding:
  id          the map as stored
  sS          scipy gaussian_filter(sigma=S px on the 512 canvas)
  gG_sS       clip(map, 0, G) / G, then gaussian_filter(sigma=S)    (MAD-AD's evaluate(): G = 0.25, S = 5 at 256 px)

    python -I -B sota/madad/dev_recall.py --keys sota/madad/keys_dev.txt --maps cache/uad_dev/madad/s10_t10:g0.25_s10,s10 \
        --out sota/madad/dev_recall.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(0, ROOT)
from cxr_recall import boxes                         # noqa: E402  (project code)
from perr.cxr_anat import search_domain              # noqa: E402
from perr.cxr_wm import peaks, valid_region          # noqa: E402

TYPES = ("Nodule", "Mass", "Calcification", "Fracture")
K, NMS = 45, 16
KS = (10, 30, 45)


def post(M, spec):
    if spec == "id":
        return M
    if spec.startswith("g"):
        g, s = spec[1:].split("_s")
        g, s = float(g), float(s)
        M = np.clip(M, 0.0, g) / g
        return ndi.gaussian_filter(M, s) if s > 0 else M
    if spec.startswith("s"):
        return ndi.gaussian_filter(M, float(spec[1:]))
    raise ValueError(spec)


def work(args):
    key, entries = args
    img = np.load(f"{ROOT}/cache/cxr/{key}.npy")
    D = search_domain(key, valid_region(img))
    out = {}
    for d, specs in entries:
        M0 = np.load(f"{d}/{key}.npy").astype(np.float32)
        for spec in specs:
            M = post(M0, spec).astype(np.float16).astype(np.float32)        # stored as float16 in cache/cxr_maps
            W = ndi.zoom(M, (512 / M.shape[0], 512 / M.shape[1]), order=1)
            out[f"{os.path.basename(d.rstrip('/'))}:{spec}"] = peaks(W, D, k=K, nms=NMS)
    return key, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    keys = open(a.keys).read().split()
    entries = []
    for m in a.maps:
        d, _, specs = m.partition(":")
        entries.append((d if d.startswith("/") else f"{ROOT}/{d}", (specs or "id").split(",")))
    b = boxes("cxdet")
    b = b[b.key.isin(set(keys)) & b.finding.isin(TYPES)].reset_index(drop=True)
    with Pool(a.workers) as p:
        res = dict(p.map(work, [(k, entries) for k in keys], chunksize=2))
    names = list(next(iter(res.values())))
    table = {}
    for n in names:
        hits = {k: [any(r.x <= c <= r.x + r.w and r.y <= rr <= r.y + r.h for rr, c, _ in res[r.key][n][:k])
                    for r in b.itertuples()] for k in KS}
        row = {f"recall@{k}": float(np.mean(hits[k])) for k in KS}
        for t in TYPES:
            sel = (b.finding == t).to_numpy()
            row[f"{t}@45"] = float(np.mean(np.array(hits[45])[sel])) if sel.any() else float("nan")
        row["n_cand"] = float(np.mean([len(res[k][n]) for k in keys]))
        table[n] = row
    print(f"{len(b)} focal boxes on {b.key.nunique()} of {len(keys)} images", flush=True)
    for n, r in sorted(table.items(), key=lambda kv: -kv[1]["recall@45"]):
        print(f"{n:40s} " + " ".join(f"{c} {v:.3f}" for c, v in r.items()), flush=True)
    if a.out:
        json.dump({"boxes": int(len(b)), "images": len(keys), "table": table}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
