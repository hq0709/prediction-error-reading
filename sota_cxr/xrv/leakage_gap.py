"""Memorisation check for the reconstructed xrv splits (leakage.py): do the xrv models score better on official test
images of patients their training part contained than on those it held out, beyond what a model trained on the
official train split shows on the same two groups?

Per group: mean AUROC over the 11 findings xrv predicts (score.py convention), and a pooled AUROC over all
(image, finding) pairs after turning each finding's scores into percentiles over the whole test split (robust for
small groups).  Run from anywhere:  python leakage_gap.py densenet121 xrv_mimic_ch xrv_mimic_nb xrv_all
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import FINDINGS, OUT, split_table, targets  # noqa: E402
from score import auroc  # noqa: E402

XRV11 = [f for f in FINDINGS if f not in ("Pleural Other", "Support Devices")]


def main():
    methods = sys.argv[1:]
    gt = split_table("test")
    y, keep = targets(gt)
    cols = [FINDINGS.index(f) for f in XRV11]
    y, keep = y[:, cols], keep[:, cols]
    preds = {}
    for m in methods:
        p = gt[["key"]].merge(pd.read_csv(f"{OUT}/{m}.csv"), on="key", how="left")
        preds[m] = p[XRV11].to_numpy(float)
    groups = {}
    for h in ("H2", "H1"):
        for part in ("trainpat", "heldpat"):
            groups[f"{h}_{part}"] = set(open(f"{HERE}/test_keys_{h}_{part}.txt").read().split())
    for part in ("trainimg", "heldimg"):
        groups[f"H2_{part}"] = set(open(f"{HERE}/test_keys_H2_{part}.txt").read().split())
    print(f"{'group':14s} {'n':>5s}  " + "  ".join(f"{m:>22s}" for m in methods))
    for g, keys in groups.items():
        sel = gt.key.isin(keys).to_numpy()
        row = []
        for m in methods:
            s = preds[m]
            per = [auroc(y[sel & keep[:, j], j], s[sel & keep[:, j], j]) for j in range(len(XRV11))]
            pct = np.column_stack([pd.Series(s[:, j]).rank(pct=True).to_numpy() for j in range(len(XRV11))])
            mk = keep & sel[:, None]
            pooled = auroc(y[mk], pct[mk])
            row.append(f"{np.nanmean(per):.3f} / pooled {pooled:.3f}")
        print(f"{g:14s} {int(sel.sum()):5d}  " + "  ".join(f"{r:>22s}" for r in row))


if __name__ == "__main__":
    main()
