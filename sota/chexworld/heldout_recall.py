"""Focal-box recall on the verifier's held-out tenth of the ChestX-Det training images (302 images; never test): the share
of positive (image, finding) pairs -- an image with a box of Nodule, Mass, Calcification or Fracture -- whose box
contains one of the proposer's top-K addresses. Addresses come from cxr_focal_data.py itself: its address function
(thorax search mask, peak finder, 16 px suppression) imported and run with the CLI's settings, or its CLI output files.

    python sota/chexworld/heldout_recall.py map:wmx_search/cxw_R224_b6x6_loss patchcore wmu --k 45
    python sota/chexworld/heldout_recall.py --glob 'wmx_search/cxw_*' --k 45      # every search map of a model
    python sota/chexworld/heldout_recall.py --jsonl data/cxr_focal/chexworld/cxdet_train.jsonl   # CLI candidates
Run with the project environment (venvs/perr) from the project root; CPU only.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cxr_focal_data as cfd  # noqa: E402
from cxr_recall import boxes  # noqa: E402

HELD = f"{ROOT}/sota/chexworld/heldout_keys.txt"
KS = (10, 20, 30, 45, 60)


def heldout_pairs(keys):
    """{(key, finding): [boxes]} for the held-out images only (boxes on the 512 canvas)."""
    b = boxes("cxdet")
    b = b[b.key.isin(set(keys)) & b.finding.isin(cfd.TYPES)]
    assert set(b.split) <= {"train"}
    return {kf: g[["x", "y", "w", "h"]].to_numpy() for kf, g in b.groupby(["key", "finding"])}


def addresses(proposer, keys, k, nms=16, sigma=2.0, sign=1):
    cfd.PROPOSER, cfd.NAME, cfd.K, cfd.NMS, cfd.SIGMA, cfd.SIGN = proposer, "unused", k, nms, sigma, sign
    with get_context("fork").Pool(32) as p:
        return dict(zip(keys, p.map(cfd.addresses, keys, chunksize=2)))


def first_hit(pts, bx):
    return next((i + 1 for i, (r, c) in enumerate(pts)
                 if any(x <= c <= x + w and y <= r <= y + h for x, y, w, h in bx)), None)


def summary(name, addr, pairs):
    hits = {kf: first_hit([(p[0], p[1]) for p in addr.get(kf[0], [])], bx) for kf, bx in pairs.items()}
    out = {"name": name, "pairs": len(hits), "cands": float(np.mean([len(v) for v in addr.values()])),
           "recall": float(np.mean([h is not None for h in hits.values()]))}
    out.update({f"@{K}": float(np.mean([h is not None and h <= K for h in hits.values()])) for K in KS})
    for t in cfd.TYPES:
        v = [h is not None for (kk, f), h in hits.items() if f == t]
        out[f"{t[:4]} (n={len(v)})"] = float(np.mean(v)) if v else float("nan")
    return out


def from_jsonl(path):
    addr = {}
    for line in open(path):
        r = json.loads(line)
        if r["src"] == "wm":
            addr.setdefault(r["key"], []).append((r["rank"], r["r"], r["c"]))
    return {k: [(r, c) for _, r, c in sorted(v)] for k, v in addr.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proposers", nargs="*", help="cxr_focal_data.py proposers, e.g. map:<name>, patchcore, wmu, random")
    ap.add_argument("--glob", default=None, help="every map directory cache/cxr_maps/<glob> (as map:<name>)")
    ap.add_argument("--jsonl", nargs="*", default=[], help="candidate files written by cxr_focal_data.py")
    ap.add_argument("--k", type=int, default=45)
    ap.add_argument("--out", default=None, help="also write the table as csv")
    a = ap.parse_args()
    keys = open(HELD).read().split()
    pairs = heldout_pairs(keys)
    props = list(a.proposers)
    if a.glob:
        props += ["map:" + os.path.relpath(p, f"{ROOT}/cache/cxr_maps")
                  for p in sorted(glob.glob(f"{ROOT}/cache/cxr_maps/{a.glob}")) if os.path.isdir(p)]
    rows = [summary(p, addresses(p, keys, 30 if p.startswith("wm") else a.k), pairs) for p in props]   # wm*: CLI default
    rows += [summary(j, from_jsonl(j), pairs) for j in a.jsonl]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"held-out tenth: {len(keys)} images, {len(pairs)} positive (image, finding) pairs; K = {a.k} for "
          "single-map proposers (wmu: both world models' 30 each, union)")
    print(df.to_string(index=False, float_format="%.3f"))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
