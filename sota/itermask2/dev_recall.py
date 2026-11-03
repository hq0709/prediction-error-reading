"""Focal-box recall of a map proposer on the radiograph verifier's held-out tenth of ChestX-Det train (302 images,
sota/itermask2/dev_keys.txt): the share of Nodule / Mass / Calcification / Fracture boxes that contain one of the top-K
addresses. The addresses are cxr_focal_data.addresses() itself (thorax search domain, 16 px suppression, K per map),
so they are the ones `cxr_focal_data.py cxdet --proposer map:<name> --k 45` writes, without rendering views.

    python sota/itermask2/dev_recall.py map:itermask2_s0 map:itermask2_s2 patchcore random
    python sota/itermask2/dev_recall.py --jsonl data/cxr_focal/itermask2_dev/cxdet_train.jsonl   # from written rows
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cxr_focal_data as cfd  # noqa: E402
from cxr_recall import boxes  # noqa: E402

TYPES = ("Nodule", "Mass", "Calcification", "Fracture")
KS = (10, 30, 45)


def recall(addr, b):
    """addr: {key: [(r, c, v), ...] ranked}; b: focal boxes. Box recall and image recall at each K."""
    out = {}
    for k in KS:
        hit = np.array([any(r.x <= c <= r.x + r.w and r.y <= rr <= r.y + r.h for rr, c, _ in addr.get(r.key, [])[:k])
                        for r in b.itertuples()])
        img = b.assign(h=hit).groupby("key").h.any()
        out[k] = {"box": float(hit.mean()), "image": float(img.mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proposers", nargs="*")
    ap.add_argument("--keys", default=f"{ROOT}/sota/itermask2/dev_keys.txt")
    ap.add_argument("--jsonl", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="append the results as JSON lines to this file")
    a = ap.parse_args()
    keys = open(a.keys).read().split()
    b = boxes("cxdet")
    b = b[b.key.isin(set(keys)) & b.finding.isin(TYPES)].reset_index(drop=True)
    print(f"{len(b)} focal boxes on {b.key.nunique()} of {len(keys)} images", flush=True)
    res = {}
    for prop in a.proposers:
        cfd.PROPOSER, cfd.K, cfd.NMS, cfd.SIGMA, cfd.SIGN = prop, max(KS), 16, 2.0, 1
        with Pool(48) as p:
            addr = dict(zip(keys, p.map(cfd.addresses, keys, chunksize=2)))
        res[prop] = recall(addr, b)
    for path in a.jsonl:
        addr = {}
        for r in map(json.loads, open(path)):
            if r["src"] == "wm":
                addr.setdefault(r["key"], []).append((r["rank"], r["r"], r["c"]))
        addr = {k: [(r, c, 0.0) for _, r, c in sorted(v)] for k, v in addr.items()}
        res[path] = recall(addr, b)
    for name, r in res.items():
        print(f"{name:48s} " + "  ".join(f"K={k}: box {v['box']:.3f} image {v['image']:.3f}" for k, v in r.items()),
              flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for name, r in res.items():
                f.write(json.dumps({"proposer": name, "n_boxes": len(b), "recall": r}) + "\n")


if __name__ == "__main__":
    main()
