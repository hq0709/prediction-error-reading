"""World models compared at the granularity SliceWorld's state supports, a slice: on the lung evaluation sets (fresh1 =
cases 60-159, fresh2 = 160-259), the reported slice is a hit when it lies within radius + 3 mm (along z) of an annotated
nodule >= 6 mm. Our rows report the slice of the address the pipeline reports (final reader, WM x reader over the top
10) or of the world model's top address; their 3-D hit (perr/score.py) is given alongside. SliceWorld rows
(train_sliceworld.py) report the slice with the highest lesion-occupancy probability (trained on nodule labels), the
highest one-step prediction error, or that error over the predicted uncertainty; SliceWorld has no in-plane location.

    python collect_sliceworld.py [runs_sw/lung]
"""
import json
import sys
from collections import defaultdict

import numpy as np

from perr.score import score_case
from perr.study import load_luna
from train_sliceworld import slice_hit, top_slice

SETS = ((60, 160), (160, 260))


def ours(lo, hi, rule):
    res = json.load(open(f"results/reader_q8_ctx_nomeas_eval_{lo}_{hi}.json"))
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"data/reader_ctx/eval_{lo}_{hi}.jsonl"))}
    by = defaultdict(list)
    for r in res:
        by[r["uid"]].append({**r, "zyx": meta[(r["uid"], r["peak"])]["zyx"]})
    sh, h3 = [], []
    for u, cs in by.items():
        cs = sorted(cs, key=lambda c: c["peak"])[:10]
        c = cs[0] if rule == "wm" else max(cs, key=lambda c: c["wm"] / cs[0]["wm"] * c["p"])
        s = load_luna(u)
        z_mm = c["zyx"][0] * s.spacing[0]
        sh.append(any(n.diameter_mm >= 6 and abs(z_mm - n.zyx[0] * s.spacing[0]) <= n.diameter_mm / 2 + 3 for n in s.nodules))
        h3.append(score_case(s, {"present": True, "zyx": c["zyx"]})["hit6"])
    return float(np.mean(sh)), float(np.mean(h3))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs_sw/lung"
    rows = []
    for label, rule in (("PER (ours): reported address", "final"), ("our world model alone: top address", "wm")):
        r = [ours(lo, hi, rule) for lo, hi in SETS]
        rows.append((label, [x[0] for x in r], [x[1] for x in r]))
    for label, key in (("SliceWorld: lesion occupancy (nodule labels)", "occ"),
                       ("SliceWorld: prediction error", "err"),
                       ("SliceWorld: prediction error / predicted uncertainty", "surprise")):
        sh, h3 = [], []
        for lo, hi in SETS:
            r = json.load(open(f"{run}/eval_{lo}_{hi}.json"))
            sh.append(float(np.mean([slice_hit(u, top_slice(v, key)) for u, v in r.items()])))
            try:                                   # eval_sliceworld_3d.py: occupancy slice + gradient x input peak
                r3 = json.load(open(f"{run}/eval3d_{lo}_{hi}.json")) if key == "occ" else None
            except OSError:
                r3 = None
            h3.append(float(np.mean([v["hit6"] for v in r3.values()])) if r3 else None)
        rows.append((label, sh, h3))
    print(f"{'method':55s} slice hit f1 / f2    3-D hit f1 / f2")
    for label, sh, h3 in rows:
        h = " / ".join("–" if v is None else f"{v:.2f}" for v in h3)
        print(f"{label:55s} {sh[0]:.2f} / {sh[1]:.2f}          {h}")


if __name__ == "__main__":
    main()
