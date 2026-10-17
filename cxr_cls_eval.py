"""MIMIC-CXR test split: AUROC per finding of reader scores P(yes) (rows: one image x finding question each; uncertain
labels were already left out), the mean over the 13 findings and over the 5 CheXpert competition findings.

    python cxr_cls_eval.py results/reader_cxr_global_test.json results/reader_cxr_wm_test.json ...
"""
import json
import sys
from collections import defaultdict

import numpy as np

from perr.cxr import FINDINGS

LABELS = [f for f in FINDINGS if f != "No Finding"]
COMPETITION = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")


def auroc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def per_finding(path):
    by = defaultdict(lambda: ([], []))
    for r in json.load(open(path)):
        f = LABELS[r["peak"] - 1]
        by[f][0].append(r["p"]); by[f][1].append(r["label"])
    return {f: auroc(*by[f]) for f in LABELS if f in by}


if __name__ == "__main__":
    res = {p: per_finding(p) for p in sys.argv[1:]}
    print(f"{'finding':28s}" + "".join(f"{p.split('reader_')[-1][:18]:>20s}" for p in res))
    for f in LABELS:
        print(f"{f:28s}" + "".join(f"{r.get(f, float('nan')):20.3f}" for r in res.values()))
    print(f"{'mean (13)':28s}" + "".join(f"{np.mean(list(r.values())):20.3f}" for r in res.values()))
    print(f"{'mean (5 competition)':28s}" + "".join(f"{np.mean([r[f] for f in COMPETITION]):20.3f}" for r in res.values()))
