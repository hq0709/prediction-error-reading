"""Lung label-efficiency curve (fresh1 = cases 60-159, fresh2 = 160-259; hit = within radius + 3 mm of a nodule >= 6 mm):
our reader trained on the nodule labels of b% of the training scans (results/lung_train_uids_<b>.txt) and reading the
addresses of the world model trained without any lesion annotation (data/reader_ctxfree; WM x reader over the top 10),
against MONAI's 3-D RetinaNet trained on the boxes of the same scans (train_ct_retinanet.py; rule and checkpoint chosen
on validation).

    python collect_ct_curve.py
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

from perr.score import score_case
from perr.study import load_luna

BUDGETS = (5, 10, 25, 50, 100)
SETS = ((60, 160), (160, 260))


def reader_hit(name, lo, hi, data="data/reader_ctxfree", k=10, rule="wm_x_reader"):
    path = f"results/reader_{name}_eval_{lo}_{hi}.json"
    if not os.path.exists(path):
        return None
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"{data}/eval_{lo}_{hi}.jsonl"))}
    by = defaultdict(list)
    for r in json.load(open(path)):
        by[r["uid"]].append({**r, "zyx": meta[(r["uid"], r["peak"])]["zyx"]})
    hits = []
    for u, cs in by.items():
        cs = sorted(cs, key=lambda c: c["peak"])[:k]
        c = cs[0] if rule == "wm_top1" else max(cs, key=lambda c: c["wm"] / cs[0]["wm"] * c["p"])
        hits.append(score_case(load_luna(u), {"present": True, "zyx": c["zyx"]})["hit6"])
    return float(np.mean(hits))


def detector_hit(run, lo, hi):
    log = f"logs/{run}.log"
    if not os.path.exists(log):
        return None
    m = re.search(rf"eval {lo}-{hi}: hit \(rule chosen on validation, \w+\) ([0-9.]+)", open(log).read())
    return float(m.group(1)) if m else None


def main():
    rows = [("PER (ours), no labels: world model's top address", 0,
             [reader_hit("nm_on_ctxfree", lo, hi, rule="wm_top1") for lo, hi in SETS])]
    for b in BUDGETS:
        ours = "q8_ctxfree_nomeas" if b == 100 else (f"q8_ctxfree2_b{b}" if b != 5 else "q8_ctxfree_b5")  # per-pass checkpoints
        rows.append(("PER (ours), world model without lesion labels", b, [reader_hit(ours, lo, hi) for lo, hi in SETS]))
    for prefix, label in (("ct_retina3", "3-D RetinaNet (MedicalNet ResNet-50)"), ("ct_retina2", "3-D RetinaNet (from scratch)")):
        for b in BUDGETS:
            rows.append((label, b, [detector_hit(f"{prefix}_b{b}", lo, hi) for lo, hi in SETS]))
    print(f"{'method':50s} {'scans %':>8s}  fresh1  fresh2")
    for label, b, h in rows:
        if all(v is None for v in h):
            continue
        print(f"{label:50s} {b:8d}  " + "  ".join("  –  " if v is None else f"{v:.2f}" for v in h))


if __name__ == "__main__":
    main()
