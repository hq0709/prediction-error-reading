"""Follow-up as prediction-error reading: the pair is called progressive if the reader says yes at any of the
no-change world model's top-k error peaks. Pair-level AUROC and balanced accuracy at P(yes) = 0.5.

    python summarize_fu_reader.py results/reader_<name>_fu_pilot.json [...]
"""
import json
import sys
from collections import defaultdict

import numpy as np


def auroc(s, y):
    pos = [a for a, b in zip(s, y) if b]
    neg = [a for a, b in zip(s, y) if not b]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def bal_acc(pred, y):
    y, pred = np.array(y, bool), np.array(pred, bool)
    return 0.5 * (pred[y].mean() + (~pred[~y]).mean()), pred[y].mean(), (~pred[~y]).mean()


if __name__ == "__main__":
    for path in sys.argv[1:]:
        by = defaultdict(list)
        for r in json.load(open(path)):
            by[r["uid"]].append(r)
        cases = [sorted(v, key=lambda r: r["peak"]) for v in by.values()]
        y = [cs[0]["label"] for cs in cases]
        print(f"{path}  n={len(cases)} (PD {sum(y)})")
        for K in sorted({1, 3, max(len(cs) for cs in cases)}):
            wm = [cs[0]["wm"] for cs in cases]
            maxp = [max(c["p"] for c in cs[:K]) for cs in cases]
            ratio = lambda c, cs: c["wm"] / cs[0]["wm"] if cs[0]["wm"] > 0 else 1.0
            fuse = [max(ratio(c, cs) * c["p"] for c in cs[:K]) for cs in cases]
            ba, se, sp = bal_acc([m > 0.5 for m in maxp], y)
            print(f"  K={K}  AUROC: WM max change {auroc(wm, y):.3f} | reader max-p {auroc(maxp, y):.3f} | "
                  f"WM x reader {auroc(fuse, y):.3f}   reader @0.5: bal.acc {ba:.2f} (sens {se:.2f}, spec {sp:.2f})")
