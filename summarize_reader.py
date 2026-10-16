"""Prediction-error reading as a pipeline: the world model proposes its top-K addresses, the reader scores each
strip, the chosen address is reported with the harness measurement, and the report is scored exactly like an
agent's final report (hit >= 6 mm, hit on the largest nodule, diameter error, side).

    python summarize_reader.py results/reader_<name>_eval_60_160.json [...] [--data data/reader_ctx]
"""
import json
import sys
from collections import defaultdict

import numpy as np

from perr.score import patient_side, score_case
from perr.study import load_luna

RULES = {
    "wm_top1": lambda cs: cs[0],
    "reader_maxp": lambda cs: max(cs, key=lambda c: c["p"]),
    "wm_x_reader": lambda cs: max(cs, key=lambda c: c["wm"] / cs[0]["wm"] * c["p"]),
}


def evaluate(path, ks=(5, 10), data="data/reader"):
    res = json.load(open(path))
    stem = path.split("reader_", 1)[1].rsplit(".", 1)[0]
    ev = stem[stem.index("eval_"):]
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"{data}/{ev}.jsonl"))}
    by = defaultdict(list)
    for r in res:
        by[r["uid"]].append({**r, **{k: meta[(r["uid"], r["peak"])][k] for k in ("zyx", "measurement")}})
    studies = {u: load_luna(u) for u in by}
    out = []
    for K in ks:
        for rule, pick in RULES.items():
            sc = []
            for u, cs in by.items():
                c = pick(sorted(cs, key=lambda c: c["peak"])[:K])
                final = {"present": True, "zyx": c["zyx"], "diameter_mm": c["measurement"].get("equivalent_diameter_mm"),
                         "side": patient_side(studies[u], c["zyx"])}
                sc.append(score_case(studies[u], final))
            hits = [s["diam_err_mm"] for s in sc if s["diam_err_mm"] is not None]
            out.append(f"{stem:40s} K={K:2d} {rule:12s} hit>=6 {np.mean([s['hit6'] for s in sc]):.2f}  largest "
                       f"{np.mean([s['hit_largest'] for s in sc]):.2f}  diamMAE {np.mean(hits):.1f}  side "
                       f"{np.mean([s['side_ok'] for s in sc]):.2f}")
    return "\n".join(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    data = "data/reader"
    if "--data" in args:
        i = args.index("--data")
        data = args[i + 1]
        args = args[:i] + args[i + 2:]
    for p in args:
        print(evaluate(p, data=data), flush=True)
