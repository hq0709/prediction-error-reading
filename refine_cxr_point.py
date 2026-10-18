"""Where to point once the reader has scored the addresses: the maximum (default) or the reader-weighted centre of the
addresses near it that the reader also calls positive (a mass seen from several addresses on its rim is pointed at its
middle). The rule and its two constants are chosen on the held-out tenth of ChestX-Det train (mean hit over the four
findings), then applied unchanged to the test set.

    python refine_cxr_point.py cxr_focal wmu 60
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

from cxr_recall import boxes
from score_cxr_focal import TYPES, auroc, lroc

R = "results/cxr_focal"


def point(g, t, alpha, rho):
    p = g[f"p_{t}"].to_numpy()
    i = int(p.argmax())
    if alpha >= 1:
        return p[i], g.r.iloc[i], g.c.iloc[i]
    d = np.maximum(np.abs(g.r.to_numpy() - g.r.iloc[i]), np.abs(g.c.to_numpy() - g.c.iloc[i]))
    m = (d <= rho) & (p >= alpha * p[i])
    w = p[m]
    return p[i], float((g.r.to_numpy()[m] * w).sum() / w.sum()), float((g.c.to_numpy()[m] * w).sum() / w.sum())


def evaluate(d, b, alpha, rho):
    out = {}
    keys = sorted(d.key.unique())
    groups = dict(tuple(d.groupby("key")))
    for t in TYPES:
        bt = b[b.finding == t]
        byk = {k: v for k, v in bt.groupby("key")}
        res = [point(groups[k], t, alpha, rho) for k in keys]
        pos = np.array([k in byk for k in keys])
        hit = np.array([k in byk and any(r.x <= c <= r.x + r.w and r.y <= rr <= r.y + r.h for r in byk[k].itertuples())
                        for k, (_, rr, c) in zip(keys, res)])
        s = np.array([x[0] for x in res])
        out[t] = (auroc(s, pos), hit[pos].mean(), lroc(s, pos, hit))
    return out


def load(path, k):
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    return d[(d.src == "wm") & (d["rank"] < k)]


def main():
    run, prop, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    b = boxes("cxdet")
    va, te = load(f"{R}/{run}/{prop}_cxdet_val.jsonl", k), load(f"{R}/{run}/{prop}_cxdet_test.jsonl", k)
    bva, bte = b[b.split == "train"], b[b.split == "test"]
    grid = [(1.0, 0)] + list(itertools.product((0.5, 0.7, 0.9), (16, 32, 48, 64)))
    scores = {c: np.mean([v[1] for v in evaluate(va, bva, *c).values()]) for c in grid}
    best = max(grid, key=lambda c: scores[c])
    print(f"validation mean hit: max-p {scores[(1.0, 0)]:.3f}; best {best} {scores[best]:.3f}")
    for label, c in (("max-p", (1.0, 0)), (f"centre alpha={best[0]} rho={best[1]}", best)):
        r = evaluate(te, bte, *c)
        print(f"test {label}: AUROC {np.mean([v[0] for v in r.values()]):.3f} hit {np.mean([v[1] for v in r.values()]):.3f} "
              f"LROC {np.mean([v[2] for v in r.values()]):.3f} | " + " ".join(f"{t[:4]} {v[1]:.3f}" for t, v in r.items()))


if __name__ == "__main__":
    main()
