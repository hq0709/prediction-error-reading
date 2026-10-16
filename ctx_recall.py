"""Recall of the learned anatomy world model's addresses vs the shape world model, on the world-model dev split
(60 training-pool scans held out of world-model training) or any uid list.

    python ctx_recall.py ctx runs_wm/ctx/split.json            # dev split
    python ctx_recall.py ctx uids.txt
Prints scan recall@K (a nodule >= 6 mm among the top K) for the shape prior, the learned model at several
smoothing scales (signed or positive-part smoothing), and rank-interleaved combinations; writes results/ctx_recall_<tag>.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from perr.context_wm import ISO, interleave, load_iso, peaks_from_error, to_original
from perr.score import score_peaks
from perr.study import load_luna
from perr.worldmodels import ShapePrior

TAG = "ctx"
SIGMAS = (1.0, 2.0, 3.0)
KS = (1, 5, 10, 20, 30, 50)


def first_hit(s, pk):
    return next((k + 1 for k in range(len(pk)) if score_peaks(s, pk[k:k + 1], 1)), None)


def variants(has_s):
    out = [(sg, signed, "none") for sg in SIGMAS for signed in (True, False)]
    if has_s:
        out += [(sg, True, w) for sg in SIGMAS for w in ("z", "pool")]
    return out


def name_of(sg, signed, w):
    return f"{sg:g}{'s' if signed else 'p'}{'' if w == 'none' else w}"


def work(uid):
    s = load_luna(uid)
    R = np.load(f"{ISO}/{uid}.{TAG}R.npy", mmap_mode="r")
    S = np.load(f"{ISO}/{uid}.{TAG}S.npy", mmap_mode="r") if os.path.exists(f"{ISO}/{uid}.{TAG}S.npy") else None
    _, D, meta = load_iso(uid)
    shape = ShapePrior().top(s, 50)
    res = {"uid": uid, "shape": first_hit(s, shape)}
    for sg, signed, w in variants(S is not None):
        name = name_of(sg, signed, w)
        pk = [(*to_original(p[:3], meta), p[3]) for p in peaks_from_error(R, D, k=50, sigma=sg, signed=signed, S=S, weight=w)]
        res[f"ctx{name}"] = first_hit(s, pk)
        res[f"mix{name}"] = first_hit(s, interleave(shape, pk, s.spacing))
        res[f"peaks{name}"] = [list(map(float, p)) for p in pk]
    return res


if __name__ == "__main__":
    TAG = sys.argv[1]
    src = sys.argv[2]
    uids = json.load(open(src))["dev"] if src.endswith(".json") else open(src).read().split()
    with Pool(30) as p:
        res = p.map(work, uids, chunksize=1)
    json.dump(res, open(f"results/ctx_recall_{TAG}.json", "w"))
    names = [name_of(*v) for v in variants(f"ctx{name_of(1.0, True, 'z')}" in res[0])]
    keys = ["shape"] + [f"ctx{n}" for n in names] + [f"mix{n}" for n in names]
    for k in keys:
        f = [r[k] for r in res]
        print(f"{k:8s}", " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in f]):.2f}" for K in KS))
