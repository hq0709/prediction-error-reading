"""Liver: recall of the learned world model's addresses vs the local-mean world model (LiverPrior) and the darkest
liver points, on the liver world-model dev split (15 pool cases held out of world-model training) or a case list.

    python liver_ctx_recall.py <tag> runs_wm/liver_nll/split.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from perr.context_wm import ORGANS, load_iso, peaks_from_error, to_original
from perr.liver import LiverPrior, label_at, load_liver

TAG = "lnll"
KS = (1, 5, 10, 20, 30)
VARIANTS = [(sg, w) for sg in (1.0, 2.0, 3.0, 4.0) for w in ("none", "z", "pool")]


def first_hit(case, pk):
    return next((k + 1 for k, p in enumerate(pk) if label_at(case, p[:3]) == "big"), None)


def work(case):
    s = load_liver(case)
    iso = ORGANS["liver"]["iso"]
    R = -np.load(f"{iso}/{case}.{TAG}R.npy").astype(np.float32)           # tumours are darker than expected
    S = np.load(f"{iso}/{case}.{TAG}S.npy") if os.path.exists(f"{iso}/{case}.{TAG}S.npy") else None
    _, D, meta = load_iso(case, "liver")
    res = {"case": case, "liverprior": first_hit(case, LiverPrior().top(s, 50))}
    for sg, w in VARIANTS:
        if w != "none" and S is None:
            continue
        pk = [(*to_original(p[:3], meta), p[3]) for p in peaks_from_error(R, D, k=50, sigma=sg, S=S, weight=w, nms_mm=15.0)]
        res[f"ctx{sg:g}{w}"] = first_hit(case, pk)
    return res


if __name__ == "__main__":
    TAG = sys.argv[1]
    src = sys.argv[2]
    cases = json.load(open(src))["dev"] if src.endswith(".json") else open(src).read().split()
    with Pool(15) as p:
        res = p.map(work, cases, chunksize=1)
    json.dump(res, open(f"results/liver_ctx_recall_{TAG}.json", "w"))
    for k in res[0]:
        if k == "case":
            continue
        f = [r[k] for r in res]
        print(f"{k:14s}", " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in f]):.2f}" for K in KS))
