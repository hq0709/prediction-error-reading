"""Choose the selection rule from cross-fitted trained-reader scores on the training pool (never the evaluation sets).

Each pool scan's world-model candidates were scored by a reader trained on the other half of the pool, so the scores
behave like the final reader's scores on unseen scans. Rules are compared by top-1 hit rate over pool scans with a
nodule >= 6 mm, at K = 5 and 10.

    python choose_rule_cv.py results/reader_q8_ctx_cvA_heldout.json results/reader_q8_ctx_cvB_heldout.json
"""
import json
import sys
from collections import defaultdict

import numpy as np


def maxp(cs):
    return max(cs, key=lambda c: c["p"])


def tiebreak(cs, digits=2):
    """Highest P(yes) to two decimals; ties go to the world model's higher-ranked address."""
    return max(cs, key=lambda c: (round(c["p"], digits), -c["peak"]))


def fallback(tau, pick=maxp):
    """The reader's pick, unless it rejects every candidate (max P(yes) < tau): then the world model's top address."""
    return lambda cs: pick(cs) if max(c["p"] for c in cs) >= tau else cs[0]


def wm_x_reader(cs):
    return max(cs, key=lambda c: c["wm"] / cs[0]["wm"] * c["p"])


def largest_confirmed(tau, otherwise=wm_x_reader):
    """The task asks for the largest nodule >= 6 mm: among candidates the reader confirms (P(yes) >= tau), the one
    with the largest harness-measured diameter; if it confirms none, the other rule."""
    def pick(cs):
        ok = [c for c in cs if c["p"] >= tau]
        return max(ok, key=lambda c: c.get("diam") or 0.0) if ok else otherwise(cs)
    return pick


RULES = {"wm_top1": lambda cs: cs[0], "reader_maxp": maxp, "wm_x_reader": wm_x_reader, "maxp_tiebreak": tiebreak}
for tau in (0.5, 0.7, 0.9, 0.95):
    RULES[f"largest_confirmed{tau:g}"] = largest_confirmed(tau)
for tau in (0.05, 0.1, 0.2, 0.3, 0.5):
    RULES[f"maxp_fallback{tau:g}"] = fallback(tau)
    RULES[f"tiebreak_fallback{tau:g}"] = fallback(tau, tiebreak)

if __name__ == "__main__":
    rows = [r for p in sys.argv[1:] for r in json.load(open(p))]
    diam = {(r["uid"], r["peak"]): r["measurement"].get("equivalent_diameter_mm")
            for r in map(json.loads, open("data/reader_ctx/train.jsonl")) if r["source"] == "wm"}
    rows = [dict(r, diam=diam.get((r["uid"], r["peak"]))) for r in rows]
    big = set(json.load(open("results/pool_scans_with_big.json")))
    by = defaultdict(list)
    for r in rows:
        if r["uid"] in big:
            by[r["uid"]].append(r)
    for K in (5, 10):
        cases = [sorted(v, key=lambda r: r["peak"])[:K] for v in by.values()]
        print(f"K={K} n={len(cases)} oracle {np.mean([any(c['label'] for c in cs) for cs in cases]):.3f}")
        for name, pick in RULES.items():
            print(f"  {name:22s} {np.mean([pick(cs)['label'] for cs in cases]):.3f}")
