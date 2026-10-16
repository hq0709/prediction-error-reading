"""Ways to combine the shape and learned world models, compared on the world-model dev split.

    python fuse_recall.py results/ctx_recall_ctx10k.json 2s
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

from perr.context_wm import fuse_rrf, interleave
from perr.score import score_peaks
from perr.study import load_luna
from perr.worldmodels import ShapePrior

KS = (1, 5, 10, 20, 30)


def first_hit(s, pk):
    return next((k + 1 for k in range(len(pk)) if score_peaks(s, pk[k:k + 1], 1)), None)


def work(args):
    r, name = args
    s = load_luna(r["uid"])
    shape, ctx = ShapePrior().top(s, 50), [tuple(p) for p in r[f"peaks{name}"]]
    out = {"shape": first_hit(s, shape), "ctx": first_hit(s, ctx), "interleave": first_hit(s, interleave(shape, ctx, s.spacing))}
    for c in (1, 3, 5, 10, 20):
        out[f"rrf{c}"] = first_hit(s, fuse_rrf([shape, ctx], s.spacing, c=c))
    return out


if __name__ == "__main__":
    res = json.load(open(sys.argv[1]))
    with Pool(30) as p:
        out = p.map(work, [(r, sys.argv[2]) for r in res])
    for k in out[0]:
        f = [o[k] for o in out]
        print(f"{k:10s}", " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in f]):.2f}" for K in KS))
