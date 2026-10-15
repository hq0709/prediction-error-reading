"""Evidence-gated scoring against LUNA16 consensus annotations.

A reported location is a hit when it lies within (radius + 3 mm) of an annotated nodule centre, in
millimetres. Fixed before any run: the tolerance, the 6 mm threshold, and the use of the largest
annotated nodule for the size and side questions.
"""

from __future__ import annotations

import numpy as np

TOL_MM = 3.0
MIN_DIAM = 6.0


def _mm(zyx, spacing):
    return np.array(zyx, float) * np.array(spacing, float)


def patient_side(study, zyx) -> str:
    """LPS: x grows toward the patient's left, so a point left of the lung midline (smaller x) is right-sided."""
    xs = np.nonzero(np.asarray(study.lung).any(axis=(0, 1)))[0]
    mid = (xs.min() + xs.max()) / 2 if len(xs) else study.shape[2] / 2
    return "right" if zyx[2] < mid else "left"


def score_case(study, final: dict | None) -> dict:
    nods = study.nodules
    big = [n for n in nods if n.diameter_mm >= MIN_DIAM]
    largest = max(nods, key=lambda n: n.diameter_mm)
    out = {"answered": final is not None, "present": None, "hit6": False, "hit_largest": False, "hit_any": False,
           "dist_mm": None, "diam_err_mm": None, "side_ok": None, "gt_largest_mm": round(largest.diameter_mm, 1)}
    if final is None:
        return out
    out["present"] = final.get("present")
    out["side_ok"] = (final.get("side") == patient_side(study, largest.zyx))
    if not final.get("present") or final.get("zyx") is None:
        return out
    p = _mm(final["zyx"], study.spacing)
    best, best_d = None, 1e9
    for n in nods:
        d = float(np.linalg.norm(p - _mm(n.zyx, study.spacing)))
        if d <= n.diameter_mm / 2 + TOL_MM:
            out["hit_any"] = True
            if n.diameter_mm >= MIN_DIAM:
                out["hit6"] = True
            if n is largest:
                out["hit_largest"] = True
        if d < best_d:
            best, best_d = n, d
    out["dist_mm"] = round(best_d, 1)
    if final.get("diameter_mm") is not None:
        target = largest if out["hit_largest"] else (best if out["hit_any"] else None)
        if target is not None:
            out["diam_err_mm"] = round(abs(final["diameter_mm"] - target.diameter_mm), 1)
    return out


def score_peaks(study, peaks, k: int) -> bool:
    """World model alone: does any of the top-k continuity peaks hit a nodule >= 6 mm?"""
    for z, y, x, _ in peaks[:k]:
        p = _mm((z, y, x), study.spacing)
        for n in study.nodules:
            if n.diameter_mm >= MIN_DIAM and np.linalg.norm(p - _mm(n.zyx, study.spacing)) <= n.diameter_mm / 2 + TOL_MM:
                return True
    return False
