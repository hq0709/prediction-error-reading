"""Follow-up pilot table: balanced accuracy, sensitivity, specificity and evidence at 3 mm and 10 mm.

    python -m perr.summarize_fu runs/fu_pilot
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np
from scipy import ndimage as ndi

from .lumiere import load_pair, pilot_pairs


def evidence(fu, zyx, mm: int) -> bool:
    if zyx is None:
        return False
    z, y, x = (int(round(c)) for c in zyx)
    m = ndi.binary_dilation(fu.seg_current > 0, iterations=mm)
    Z, Y, X = m.shape
    return bool(0 <= z < Z and 0 <= y < Y and 0 <= x < X and m[z, y, x])


def main(root):
    pairs = {f"{p['patient']}_{p['prior']}_{p['current']}": p for p in pilot_pairs(70)}
    print(f"{'model':12s} {'condition':11s} {'n':>3s} {'bal_acc':>7s} {'sens':>5s} {'spec':>5s} {'PD+ev3':>6s} "
          f"{'PD+ev10':>7s} {'calls':>5s} {'ktok':>6s}")
    for d in sorted(glob.glob(f"{root}/*/*")):
        recs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json"))]
        if not recs:
            continue
        pd_ = [r for r in recs if r["rating"] == "PD"]
        np_ = [r for r in recs if r["rating"] != "PD"]
        sens = np.mean([r["score"]["pred_pd"] is True for r in pd_]) if pd_ else np.nan
        spec = np.mean([r["score"]["pred_pd"] is False for r in np_]) if np_ else np.nan
        ev = {3: [], 10: []}
        for r in pd_:
            fin = r["trace"]["final"]
            ok = fin is not None and fin.get("progression") and fin.get("zyx") is not None
            fu = load_pair(pairs[r["case"]]) if ok else None
            for mm in ev:
                ev[mm].append(bool(ok and evidence(fu, fin["zyx"], mm)))
        calls = np.mean([r["trace"]["calls"] for r in recs])
        ktok = np.mean([(r["trace"]["usage"]["prompt"] + r["trace"]["usage"]["completion"]) / 1e3 for r in recs])
        print(f"{recs[0]['model']:12s} {recs[0]['cond']:11s} {len(recs):3d} {(sens + spec) / 2:7.2f} {sens:5.2f} "
              f"{spec:5.2f} {np.mean(ev[3]):6.2f} {np.mean(ev[10]):7.2f} {calls:5.1f} {ktok:6.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/fu_pilot")
