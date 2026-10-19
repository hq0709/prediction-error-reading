"""A new site needs only its normal radiographs: the 512 world model trained on NIH "No Finding" radiographs
(build_nih_normals.py, runs_wm/cxr_nll_nih, addresses "wmn") against the one trained on MIMIC "No Finding" radiographs
(runs_wm/cxr_nll, addresses "wm"), read by the same reader (runs_reader/cxr_focal, trained on ChestX-Det boxes at the
MIMIC model's addresses), on ChestX-Det test and on the NIH boxes (mean over findings of presence AUROC, pointing hit,
LROC; top 30 addresses, max-p). MIMIC report labels: score_cxr_focal_mimic.py densenet121_60k cxr_focal.
Second table: the final method (both MIMIC world models, 60 addresses, final reader) with and without the NIH-normal
world model's 30 addresses added (read by the same final reader; max-p over all addresses).

    python collect_cxr_site.py
"""
import json
import os

import pandas as pd

from cxr_recall import boxes
from score_cxr_focal import TYPES, evaluate

R = "results/cxr_focal/cxr_focal"


def main():
    out = []
    for which, split in (("cxdet", "test"), ("nih", None)):
        b = boxes(which)
        if split:
            b = b[b.split == split]
        types = [t for t in TYPES if t in set(b.finding)]
        for pre, label in (("wm", "world model trained on MIMIC normals"), ("wmn", "world model trained on NIH normals"),
                           ("wma", "world model trained on MIMIC images of any label")):
            path = f"{R}/{pre}_{which}_test.jsonl"
            if not os.path.exists(path):
                continue
            res = evaluate([json.loads(l) for l in open(path)], b, types, ks=(30,), rules=("max-p",))
            out.append({"test set": which, "addresses": label, **{m: res[m].mean() for m in ("AUROC", "hit", "LROC")},
                        **{f"{t} hit": h for t, h in zip(res.finding, res.hit)}})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(out).to_string(index=False, float_format="%.3f", na_rep=""))
    F = "results/cxr_focal/cxr_focal_wmu"
    out = []
    for which, split in (("cxdet", "test"), ("nih", None)):
        b = boxes(which)
        if split:
            b = b[b.split == split]
        types = [t for t in TYPES if t in set(b.finding)]
        base = [json.loads(l) for l in open(f"{F}/wmu_{which}_test.jsonl")]
        sets = [("final method (MIMIC world models)", base, 60)]
        if os.path.exists(f"{F}/wmn_{which}_test.jsonl"):
            add = [dict(json.loads(l)) for l in open(f"{F}/wmn_{which}_test.jsonl")]
            for r in add:
                r["rank"] = r["rank"] + 60
            sets.append(("  + world model trained on NIH normals", base + add, 90))
        for label, rows, k in sets:
            res = evaluate(rows, b, types, ks=(k,), rules=("max-p",))
            out.append({"test set": which, "addresses": label, **{m: res[m].mean() for m in ("AUROC", "hit", "LROC")},
                        **{f"{t} hit": h for t, h in zip(res.finding, res.hit)}})
    print()
    print(pd.DataFrame(out).to_string(index=False, float_format="%.3f", na_rep=""))


if __name__ == "__main__":
    main()
