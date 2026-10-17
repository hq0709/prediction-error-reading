"""Score chest X-ray classifiers on the official MIMIC-CXR-JPG test split (frontal images), per-finding AUROC.

    python sota_cxr/score.py <method> [<method> ...] [--subset KEYS.txt] [--partial]

Reads results/sota_cxr/<method>.csv: a column "key" (as in results/cxr_mimic.csv), then one probability column per
finding name.  Ground truth: the test rows of results/cxr_mimic.csv (3,403 frontal images).  Per finding, y = 1 if
the CheXpert label is 1.0, y = 0 if it is 0.0 or missing, and uncertain (-1.0) images are left out of that finding.
A finding the method does not predict (column absent or all NaN) gets no AUROC; the 13-finding mean is then
reported over the findings it does predict, with the count.  When several methods are given, the summary also
shows the mean over the findings that every listed method predicts.

Every test image must have a row (otherwise the method is not scored unless --partial is given).
--subset restricts scoring to the test keys listed in a file (one key per line), e.g. for leakage analyses.
Needs only numpy and pandas.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import COMPETITION, FINDINGS, OUT, split_table, targets  # noqa: E402


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def score(method: str, gt: pd.DataFrame, partial: bool) -> dict | None:
    path = f"{OUT}/{method}.csv"
    if not os.path.exists(path):
        print(f"{method}: {path} not found")
        return None
    p = pd.read_csv(path)
    if p.key.duplicated().any():
        print(f"{method}: {int(p.key.duplicated().sum())} duplicated keys; not scored")
        return None
    extra = int((~p.key.isin(gt.key)).sum())
    d = gt[["key"]].merge(p, on="key", how="left", indicator=True)
    missing = int((d._merge == "left_only").sum())
    print(f"\n{method}: {len(gt) - missing}/{len(gt)} test images scored" + (f" ({extra} rows outside the set ignored)" if extra else ""))
    if missing and not partial:
        print(f"  {missing} test images have no row; not scored (pass --partial to score the covered images only)")
        return None
    y, keep = targets(gt)
    res = {}
    print(f"  {'finding':28s} {'n':>5s} {'pos':>5s}  AUROC")
    for j, f in enumerate(FINDINGS):
        s = d[f].to_numpy(float) if f in d else np.full(len(d), np.nan)
        m = keep[:, j] & ~np.isnan(s)
        if f not in d or np.isnan(s).all():
            res[f] = float("nan")
            print(f"  {f:28s} {'-':>5s} {'-':>5s}  not predicted")
            continue
        res[f] = auroc(y[m, j], s[m])
        print(f"  {f:28s} {int(m.sum()):5d} {int(y[m, j].sum()):5d}  {res[f]:.4f}")
    have = [f for f in FINDINGS if not np.isnan(res[f])]
    res["mean13"] = float(np.mean([res[f] for f in have])) if have else float("nan")
    res["n13"] = len(have)
    comp = [res[f] for f in COMPETITION]
    res["mean5"] = float(np.mean(comp)) if not np.isnan(comp).any() else float("nan")
    print(f"  mean over {'the 13' if len(have) == 13 else f'{len(have)} predicted'} findings: {res['mean13']:.4f}")
    print(f"  mean over the 5 competition findings: {res['mean5']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--subset", help="file with the test keys to score (one per line)")
    ap.add_argument("--partial", action="store_true", help="score methods that miss some test images")
    a = ap.parse_args()
    gt = split_table("test")
    if a.subset:
        keys = set(open(a.subset).read().split())
        gt = gt[gt.key.isin(keys)].reset_index(drop=True)
        print(f"subset {a.subset}: {len(gt)} test images")
    out = {m: score(m, gt, a.partial) for m in a.methods}
    out = {m: r for m, r in out.items() if r is not None}
    if len(out) > 1:
        common = [f for f in FINDINGS if all(not np.isnan(r[f]) for r in out.values())]
        print(f"\n{'method':28s} {'mean13':>7s} {'(n)':>4s} {'mean5':>7s} {'common':>7s} ({len(common)} findings predicted by all)")
        for m, r in out.items():
            print(f"{m:28s} {r['mean13']:7.4f} {r['n13']:4d} {r['mean5']:7.4f} {np.mean([r[f] for f in common]):7.4f}")


if __name__ == "__main__":
    main()
