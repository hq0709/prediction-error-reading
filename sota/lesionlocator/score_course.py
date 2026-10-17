"""Score a disease-course answer file on the evaluation pairs (sota/eval_course.txt), as for our own reports.

Answer file: one JSON line per pair {"case": key, "present": bool, "zyx": [z, y, x] or null, "score": float, ...};
zyx is a voxel index on the native follow-up grid (perr.autopet.load_pair(key).fu). A pair without a line is absent
with score 0.

Targets: selected_targets (perr/course_candidates.py): new lesions, or matched lesions grown to >= x1.728 volume, with
follow-up equivalent diameter >= 10 mm. Hit: the reported point, rounded to the nearest voxel and inside the grid, lies
within 3 mm of a target's voxels (fu_lab of a selected target, grown by a Euclidean distance transform in mm).
  pair hit rate      over pairs with at least one target;
  progression AUROC  over all pairs: label = the pair has a target, score = the answer's score (ties count 1/2).
Diagnostic only: where the answers land when they miss (another lesion within 3 mm, or no lesion).

    python score_course.py /raid/home/CAMCA/hj880/perr/results/sota/lesionlocator_course.jsonl
"""
import json
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr")
from perr.autopet import load_pair  # noqa: E402
from perr.course_candidates import selected_targets  # noqa: E402

EVAL = "/raid/home/CAMCA/hj880/perr/sota/eval_course.txt"
MIN_DIAM, TOL = 10.0, 3.0


def near(mask, spacing, i):
    """Is voxel i within TOL mm of a voxel of mask? (EDT on a box around i; same result as on the full grid.)"""
    r = [int(np.ceil(TOL / s)) + 1 for s in spacing]
    sl = tuple(slice(max(0, i[d] - r[d]), i[d] + r[d] + 1) for d in range(3))
    sub = mask[sl]
    if not sub.any():
        return False
    j = tuple(i[d] - sl[d].start for d in range(3))
    return bool(ndi.distance_transform_edt(~sub, sampling=spacing)[j] <= TOL)


def main(path):
    pred = {r["case"]: r for r in map(json.loads, open(path))}
    keys = open(EVAL).read().split()
    y, s, hits, where = [], [], [], {"target": 0, "other lesion": 0, "no lesion": 0, "absent": 0}
    matched_only = 0   # pairs whose targets include a matched (grown) lesion, i.e. reachable by tracking alone
    for key in keys:
        P = load_pair(key)
        t = selected_targets(P.targets, MIN_DIAM)
        labs = sorted(int(v) for v in t.fu_label) if len(t) else []
        matched_only += int(bool(len(t)) and bool((t.topology_class != "NEWLYAPPEARING").any()))
        p = pred.get(key, {"present": False, "zyx": None, "score": 0.0})
        y.append(int(bool(labs)))
        s.append(float(p.get("score") or 0.0) if p.get("present") else 0.0)
        lab = np.asarray(P.fu_lab)
        hit, lesion = False, False
        if p.get("present") and p.get("zyx") is not None:
            i = tuple(int(round(c)) for c in p["zyx"])
            if all(0 <= i[d] < lab.shape[d] for d in range(3)):
                hit = bool(labs) and near(np.isin(lab, labs), P.spacing, i)
                lesion = near(lab > 0, P.spacing, i)
        if labs:
            hits.append(hit)
            where["target" if hit else "absent" if not p.get("present") else
                  "other lesion" if lesion else "no lesion"] += 1
    y, s = np.array(y), np.array(s)
    pos, neg = s[y == 1], s[y == 0]
    auc = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]) if len(pos) and len(neg) else float("nan")
    print(f"{path}")
    print(f"  pairs {len(keys)} (answered {sum(k in pred for k in keys)}, present "
          f"{sum(bool(pred.get(k, {}).get('present')) for k in keys)})")
    print(f"  pair hit rate {np.mean(hits):.3f} ({sum(hits)}/{len(hits)} pairs with a target)")
    print(f"  progression AUROC {auc:.3f} ({len(pos)} pairs with a target vs {len(neg)} without)")
    print(f"  answers on pairs with a target: {where}")
    print(f"  diagnostic: pairs with a target that include a grown matched lesion (reachable by tracking): "
          f"{matched_only}/{len(hits)}")


if __name__ == "__main__":
    main(sys.argv[1])
