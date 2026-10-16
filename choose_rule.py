"""Choose the selection rule on the training pool (never on the evaluation sets): top-1 hit rate of each rule,
over scans that contain a nodule >= 6 mm (as every evaluation scan does), from reader scores on pool rows.

    python choose_rule.py results/reader_<name>_train.json results/reader_<name>_val.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

RULES = {
    "wm_top1": lambda cs: cs[0],
    "reader_maxp": lambda cs: max(cs, key=lambda c: c["p"]),
    "wm_x_reader": lambda cs: max(cs, key=lambda c: c["wm"] / cs[0]["wm"] * c["p"]),
}

if __name__ == "__main__":
    rows = [r for path in sys.argv[1:] for r in json.load(open(path)) if r["source"] == "wm"]
    big = set(json.load(open("results/pool_scans_with_big.json")))
    by = defaultdict(list)
    for r in rows:
        if r["uid"] in big:
            by[r["uid"]].append(r)
    for K in (5, 10):
        cases = [sorted(v, key=lambda r: r["peak"])[:K] for v in by.values()]
        res = {name: np.mean([pick(cs)["label"] for cs in cases]) for name, pick in RULES.items()}
        print(f"K={K:2d} n={len(cases)} " + " | ".join(f"{k} {v:.3f}" for k, v in res.items())
              + f" | oracle {np.mean([any(c['label'] for c in cs) for cs in cases]):.3f}")
