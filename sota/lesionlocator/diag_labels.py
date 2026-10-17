"""Post-run interpretation of the LesionLocator evaluation run (uses evaluation labels; run only after scoring, never
fed back into the method or the answer rule).

For every prompted baseline lesion: its CSV topology class, whether LesionLocator tracked it (non-empty follow-up mask),
and for one-to-one matched lesions owned by this follow-up piece, the distance between LesionLocator's follow-up
centroid and the CSV follow-up centroid (CPM@25-style matching, as in the paper).

    python diag_labels.py plan_eval.json _work/eval
"""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr")
from perr import autopet as A  # noqa: E402


def main(plan_path, work):
    plan = json.load(open(plan_path))
    tracked, total, dists, small = Counter(), Counter(), [], Counter()
    for pair in plan:
        meta = np.load(f"{A.CACHE}/{pair['key']}.meta.npy", allow_pickle=True).item()
        rows = {r["lesion_id"]: r for r in meta["lesions"]}
        sp = np.array(meta["spacing"])
        for bl in pair["bl"]:
            d = json.load(open(f"{work}/{pair['key']}/bl{bl['piece']:02d}/done.json"))
            for L in d["lesions"]:
                if L.get("bl_seg_file") is None or bl["binary_mode"]:
                    continue
                r = rows.get(L.get("bl_lesion_id"))
                if r is None:
                    continue
                cls = r["topology_class"] + ("" if r["topology_class"] == "DISAPPEARING" else
                                             (" (this FU piece)" if r["owned"] else " (other FU piece)"))
                tr = L.get("fu_seg", {}).get("nvox", 0) > 0
                total[cls] += 1
                tracked[cls] += tr
                small[(cls, r["bl_ml"] < 0.5)] += 1
                if tr and r["topology_class"] == "UNCHANGED" and r["owned"] and not r["linking_unclear"]:
                    c = np.array(L["fu_seg"]["centroid_zyx"])
                    dists.append(float(np.linalg.norm((c - [r["fu_z"], r["fu_y"], r["fu_x"]]) * sp)))
    print("tracked / prompted baseline lesions by CSV topology class:")
    for k in sorted(total):
        print(f"  {k:32s} {tracked[k]:4d} / {total[k]:4d} = {tracked[k] / total[k]:.2f}")
    d = np.array(dists)
    n_unch = total.get("UNCHANGED (this FU piece)", 0)
    print(f"matched (UNCHANGED, this FU piece) and tracked: {len(d)}; centroid error median {np.median(d):.1f} mm, "
          f"<= 25 mm {np.mean(d <= 25):.2f}; CPM@25 over all {n_unch} such lesions {np.sum(d <= 25) / max(n_unch, 1):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
