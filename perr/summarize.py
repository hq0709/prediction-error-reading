"""Aggregate pilot runs into one table (plus the world-model-only baseline, which needs no model).

    python -m perr.summarize runs/pilot
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np

from .score import score_peaks
from .study import load_luna, pilot_cases
from .worldmodels import ShapePrior


def rows(root: str):
    out = []
    for model_dir in sorted(glob.glob(f"{root}/*")):
        for cond_dir in sorted(glob.glob(f"{model_dir}/*")):
            recs = [json.load(open(f)) for f in sorted(glob.glob(f"{cond_dir}/*.json"))]
            if not recs:
                continue
            sc = [r["score"] for r in recs]
            tr = [r["trace"] for r in recs]
            hits = [s for s in sc if s["diam_err_mm"] is not None]
            out.append({
                "model": recs[0]["model"], "cond": recs[0]["cond"], "n": len(recs),
                "answered": np.mean([s["answered"] for s in sc]),
                "hit6": np.mean([s["hit6"] for s in sc]),
                "hit_largest": np.mean([s["hit_largest"] for s in sc]),
                "diam_mae": np.mean([s["diam_err_mm"] for s in hits]) if hits else float("nan"),
                "side": np.mean([bool(s["side_ok"]) for s in sc]),
                "calls": np.mean([t["calls"] for t in tr]),
                "views": np.mean([t["views_rendered"] for t in tr]),
                "ktok": np.mean([(t["usage"]["prompt"] + t["usage"]["completion"]) / 1e3 for t in tr]),
                "invalid": np.mean([t["invalid"] for t in tr]),
                "errors": sum("error" in t for t in tr),
            })
    return out


def wm_only(n: int):
    wm = ShapePrior()
    hits = []
    for uid in pilot_cases(n):
        s = load_luna(uid)
        hits.append(score_peaks(s, wm.peaks(s), 1))
    return float(np.mean(hits))


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "runs/pilot"
    rs = rows(root)
    n = max((r["n"] for r in rs), default=50)
    print(f"{'model':14s} {'condition':12s} {'n':>3s} {'answer':>6s} {'hit>=6':>6s} {'largest':>7s} "
          f"{'diamMAE':>7s} {'side':>5s} {'calls':>5s} {'views':>5s} {'ktok':>6s} {'bad':>4s} {'err':>3s}")
    print(f"{'(none)':14s} {'wm_top1':12s} {n:3d} {'':>6s} {wm_only(n):6.2f}")
    for r in rs:
        print(f"{r['model']:14s} {r['cond']:12s} {r['n']:3d} {r['answered']:6.2f} {r['hit6']:6.2f} {r['hit_largest']:7.2f} "
              f"{r['diam_mae']:7.1f} {r['side']:5.2f} {r['calls']:5.1f} {r['views']:5.1f} {r['ktok']:6.1f} "
              f"{r['invalid']:4.1f} {r['errors']:3d}")
