"""World model + mechanical measurement, no reader: among the top-k peaks, report the one whose seed-grown
structure is largest (ignoring outlines that leak). Scored like the agents."""
import json, sys
import numpy as np
from multiprocessing import Pool
from perr.study import load_luna, pilot_cases
from perr.harness import Session
from perr.score import score_case

def run(uid, k=5):
    s = load_luna(uid); S = Session(s, "harness_wm", "norm1000", 512); S.overview()
    best = None
    for n in range(1, min(k, len(S.peaks)) + 1):
        out = json.loads(S.call("measure", {"peak": n})[0])
        d = out.get("equivalent_diameter_mm")
        if d and "warning" not in out and (best is None or d > best[1]):
            best = (n, d)
    if best is None:
        S.final = {"present": False, "zyx": None, "views": [], "diameter_mm": None, "side": None}
    else:
        z, y, x, _ = S.peaks[best[0] - 1]
        S.final = {"present": True, "zyx": (z, y, x), "views": [], "diameter_mm": best[1], "side": None}
    return score_case(s, S.final)

if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    with Pool(10) as p: sc = p.map(run, pilot_cases(hi)[lo:hi])
    print("WM+measure rule (top-5): hit6 %.2f, largest %.2f, diam MAE %.1f (n=%d)" % (
        np.mean([s["hit6"] for s in sc]), np.mean([s["hit_largest"] for s in sc]),
        np.mean([s["diam_err_mm"] for s in sc if s["diam_err_mm"] is not None] or [np.nan]), len(sc)))
