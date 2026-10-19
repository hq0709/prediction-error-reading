"""Choose the liver reading budget K, address list and rule on the 100 held-out Task08 cases (make_liver_kdev.py), then
report the chosen configuration, unchanged, on the MSD evaluation half and the external HCC-TACE-Seg set (hit: the
reported address inside a target tumour >= 10 mm). Ties go to the smaller K, then the fused list, then max-p (the
earlier final rule).

    python collect_liver_kdev.py [reader name]
"""
import json
import sys
from collections import defaultdict

import numpy as np

KS, LISTS, RULES = (10, 20, 30), ("fused", "prior", "learned"), ("maxp", "wm_x_reader")


def load(name, which, lst):
    meta = {(json.loads(l)["uid"], json.loads(l)["peak"]): json.loads(l) for l in open(f"data/liver_k30_{which}_{lst}/eval.jsonl")}
    by = defaultdict(list)
    for r in json.load(open(f"results/reader_{name}_eval_{which}_{lst}.json")):
        by[r["uid"]].append({**r, "label": meta[(r["uid"], r["peak"])]["label"]})
    return by


def hit(by, k, rule):
    h = []
    for cs in by.values():
        cs = sorted(cs, key=lambda c: c["peak"])[:k]
        c = max(cs, key=lambda c: c["p"]) if rule == "maxp" else max(cs, key=lambda c: c["wm"] / max(cs[0]["wm"], 1e-9) * c["p"])
        h.append(c["label"])
    return float(np.mean(h))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "q8_liver_t08h"
    grid = {}
    for lst in LISTS:
        by = load(name, "t08", lst)
        for k in KS:
            for rule in RULES:
                grid[(k, lst, rule)] = hit(by, k, rule)
    order = sorted(grid, key=lambda c: (-grid[c], KS.index(c[0]), LISTS.index(c[1]), RULES.index(c[2])))
    print("Task08 held-out (100 cases), hit by K / list / rule:")
    for c in order[:8]:
        print(f"  K={c[0]:2d} {c[1]:8s} {c[2]:12s} {grid[c]:.3f}")
    best = order[0]
    print(f"chosen: K={best[0]}, {best[1]} list, {best[2]}")
    for which, label in (("msd", "MSD evaluation (53)"), ("ext", "HCC-TACE-Seg external (93)")):
        by = load(name, which, best[1])
        ref = load(name, which, "fused")
        print(f"{label}: chosen {hit(by, best[0], best[2]):.3f}; earlier configuration (K=10, fused, maxp) "
              f"{hit(ref, 10, 'maxp'):.3f}; oracle@K {np.mean([any(c['label'] for c in sorted(v, key=lambda c: c['peak'])[:best[0]]) for v in by.values()]):.3f}")


if __name__ == "__main__":
    main()
