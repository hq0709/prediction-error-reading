"""Score baseline predictions exactly like our reports.

Each method writes results/sota/<method>_luna.jsonl and/or results/sota/<method>_liver.jsonl, one line per case:
    {"case": id, "present": bool, "zyx": [z, y, x] or null, "diameter_mm": float or null, ...}
zyx is a continuous voxel index into the cached array (cache/luna/<uid>.vol.npy, cache/liver/<case>.vol.npy:
[z, y, x], x -> patient left, y -> posterior, z -> superior). A case with no line counts as "absent".

LUNA16 (fresh1 / fresh2): hit = the point lies within radius + 3 mm of a nodule >= 6 mm; largest = of the largest
nodule; diameter MAE over hits. Liver: hit = the point lies inside a tumour >= 10 mm grown by 3 mm.

    python score_sota.py <method> [<method> ...]
"""
import json
import os
import sys

import numpy as np

from perr.liver import label_at
from perr.score import score_case
from perr.study import load_luna


def read(path):
    return {r["case"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else None


def luna(method):
    pred = read(f"results/sota/{method}_luna.jsonl")
    if pred is None:
        return None
    out = []
    for name in ("fresh1", "fresh2"):
        sc = []
        for u in open(f"sota/eval_luna_{name}.txt").read().split():
            p = pred.get(u)
            final = None if p is None else {"present": bool(p.get("present")), "zyx": p.get("zyx"),
                                            "diameter_mm": p.get("diameter_mm"), "side": None}
            sc.append(score_case(load_luna(u), final or {"present": False}))
        d = [s["diam_err_mm"] for s in sc if s["diam_err_mm"] is not None]
        out.append(f"{name} hit {np.mean([s['hit6'] for s in sc]):.2f} largest {np.mean([s['hit_largest'] for s in sc]):.2f} "
                   f"diamMAE {np.mean(d) if d else float('nan'):.1f} answered {sum(u in pred for u in open(f'sota/eval_luna_{name}.txt').read().split())}")
    return " | ".join(out)


def liver(method):
    pred = read(f"results/sota/{method}_liver.jsonl")
    if pred is None:
        return None
    cases = open("sota/eval_liver.txt").read().split()
    hits = [bool(p and p.get("present") and p.get("zyx") is not None and label_at(c, p["zyx"]) == "big")
            for c, p in ((c, pred.get(c)) for c in cases)]
    return f"liver hit {np.mean(hits):.2f} (n={len(cases)}, answered {sum(c in pred for c in cases)})"


if __name__ == "__main__":
    for m in sys.argv[1:]:
        for f in (luna, liver):
            r = f(m)
            if r:
                print(f"{m:24s} {r}")
