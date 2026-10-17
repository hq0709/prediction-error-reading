"""Answer rule for LesionLocator on the disease-course task (fixed before the evaluation run; reads model outputs only).

For each pair, every baseline lesion that LesionLocator segmented on the baseline (box prompt from the baseline mask)
and then tracked into the follow-up gives one candidate:
  growth   = follow-up segmentation volume / baseline segmentation volume (both LesionLocator's own masks, ml);
  location = centroid of the follow-up segmentation, [z, y, x] voxel index on the native follow-up grid.
A lesion counts as tracked when its follow-up mask is non-empty. The answer is the tracked lesion with the largest
growth (ties: baseline piece order, then lesion order); the progression score is that growth. No tracked lesion ->
present false, zyx null, score 0. All tracked lesions are listed in "candidates", largest growth first.

    python answers.py plan_eval.json _work/eval /raid/home/CAMCA/hj880/perr/results/sota/lesionlocator_course.jsonl
"""
import json
import sys

import numpy as np


def pair_answer(pair, work):
    cands, untracked, n = [], [], 0
    for bl in pair["bl"]:
        d = json.load(open(f"{work}/{pair['key']}/bl{bl['piece']:02d}/done.json"))
        annot = bl["lesions"]
        for L in d["lesions"]:
            if L.get("bl_seg_file") is None and L.get("bl_lesion_id") is None:
                continue        # empty prompt slot: lesion id absent from this BL piece's mask (no prompt, no output)
            n += 1
            lid = L.get("bl_lesion_id")
            bl_ml = L.get("bl_seg", {}).get("ml", 0.0)
            fu = L.get("fu_seg", {})
            if bl_ml <= 0 or fu.get("nvox", 0) == 0:
                untracked.append({"bl_piece": bl["piece"], "bl_lesion_id": lid, "track_index": L["track_index"],
                                  "bl_ml": bl_ml, "reason": "no baseline segmentation" if bl_ml <= 0 else
                                  "empty follow-up segmentation"})
                continue
            fu_ml = fu["ml"]
            cands.append({"zyx": fu["centroid_zyx"], "growth": fu_ml / bl_ml, "fu_ml": fu_ml, "bl_ml": bl_ml,
                          "bl_ml_annot": annot.get(str(lid), {}).get("ml") if not bl["binary_mode"] else None,
                          "fu_diam_mm": float((6.0 * fu_ml * 1000.0 / np.pi) ** (1.0 / 3.0)),
                          "fu_ncomp": fu["ncomp"], "bl_piece": bl["piece"], "bl_lesion_id": lid,
                          "track_index": L["track_index"]})
    cands.sort(key=lambda c: -c["growth"])   # stable: ties keep piece / lesion order
    top = cands[0] if cands else None
    return {"case": pair["key"], "present": top is not None, "zyx": top["zyx"] if top else None,
            "score": float(top["growth"]) if top else 0.0, "candidates": cands,
            "n_bl_lesions": n, "n_tracked": len(cands), "untracked": untracked,
            "bl_pieces": [bl["piece"] for bl in pair["bl"]]}


if __name__ == "__main__":
    plan_path, work, out = sys.argv[1:4]
    plan = json.load(open(plan_path))
    rows = [pair_answer(p, work) for p in plan]
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} pairs -> {out}: present {sum(r['present'] for r in rows)}, "
          f"lesions {sum(r['n_bl_lesions'] for r in rows)}, tracked {sum(r['n_tracked'] for r in rows)}")
