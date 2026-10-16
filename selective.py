"""Selective answering: answer only the scans where the reader is most confident in its chosen candidate.

The chosen candidate is the reader's max-P(yes) among the world model's top K; its P(yes) is the confidence. Prints
accuracy (hit rate) at 50 / 70 / 90 / 100% coverage and the AUROC of the confidence for right vs wrong answers.

    python selective.py results/reader_<name>_eval_60_160.json results/reader_<name>_eval_160_260.json [--k 10]
"""
import json
import sys
from collections import defaultdict

import numpy as np


def chosen(paths, k=10):
    out = []
    for path in paths:
        by = defaultdict(list)
        for r in json.load(open(path)):
            by[r["uid"]].append(r)
        for cs in by.values():
            c = max(sorted(cs, key=lambda r: r["peak"])[:k], key=lambda r: r["p"])
            out.append((c["p"], int(c["label"])))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    k = 10
    if "--k" in args:
        i = args.index("--k")
        k = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    ch = sorted(chosen(args, k), key=lambda t: -t[0])
    y = np.array([t[1] for t in ch])
    p = np.array([t[0] for t in ch])
    acc = {cov: y[:max(1, int(round(cov * len(y))))].mean() for cov in (0.5, 0.7, 0.9, 1.0)}
    pos, neg = p[y == 1], p[y == 0]
    auc = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]) if len(pos) and len(neg) else float("nan")
    print(f"n={len(y)} K={k} " + " ".join(f"acc@{int(c * 100)}% {v:.2f}" for c, v in acc.items()) + f" | confidence AUROC {auc:.3f}")
