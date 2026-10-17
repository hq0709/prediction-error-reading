"""Coordinate sanity check on training pairs (labels of training pairs only).

For each LesionLocator-tracked baseline lesion that the CSV links one-to-one to a follow-up lesion (UNCHANGED, owned
by this FU piece, not linking_unclear): distance (mm) between LesionLocator's follow-up centroid and the CSV follow-up
centroid (perr.autopet lesion table, [z, y, x] on the native FU grid), Dice with the follow-up annotation, and volumes.
Also checks that every output mask has the FU image's exact geometry.

    python sanity_train.py plan_sanity.json _work/sanity
"""
import json
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr")
from perr.autopet import load_pair  # noqa: E402


def main(plan_path, work):
    plan = json.load(open(plan_path))
    dists = []
    for pair in plan:
        P = load_pair(pair["key"])
        L = P.lesions.set_index("lesion_id")
        fu_ref = sitk.ReadImage(pair["fu"])
        lab = np.asarray(P.fu_lab)
        sp = np.array(P.spacing)
        print(f"\n{pair['key']}  FU shape {P.shape} spacing {tuple(round(v, 2) for v in P.spacing)}")
        for bl in pair["bl"]:
            d = json.load(open(f"{work}/{pair['key']}/bl{bl['piece']:02d}/done.json"))
            print(f"  BL piece {bl['piece']}: {d['seconds']}")
            for rec in d["lesions"]:
                lid = rec.get("bl_lesion_id")
                fu = rec.get("fu_seg", {})
                row = L.loc[lid] if lid in L.index else None
                line = f"    lesion {lid}: "
                if row is None:
                    print(line + "not in CSV")
                    continue
                line += f"{row.topology_class:15s} CSV bl {row.bl_ml:7.2f} ml fu {row.fu_ml:7.2f} ml ratio {row.ratio:5.2f} | "
                line += f"LL bl {rec.get('bl_seg', {}).get('ml', 0):7.2f} ml "
                if not fu.get("nvox"):
                    print(line + "| not tracked (empty FU mask)")
                    continue
                img = sitk.ReadImage(rec["fu_seg_file"])
                geo = (img.GetSize() == fu_ref.GetSize() and np.allclose(img.GetSpacing(), fu_ref.GetSpacing())
                       and np.allclose(img.GetOrigin(), fu_ref.GetOrigin())
                       and np.allclose(img.GetDirection(), fu_ref.GetDirection()))
                m = sitk.GetArrayFromImage(img) > 0
                line += f"fu {fu['ml']:7.2f} ml growth {fu['ml'] / rec['bl_seg']['ml']:5.2f} ncomp {fu['ncomp']} geo {'ok' if geo else 'MISMATCH'}"
                if row.topology_class == "UNCHANGED" and row.owned and not row.linking_unclear:
                    gt = lab == int(row.fu_label)
                    dice = 2 * (m & gt).sum() / (m.sum() + gt.sum())
                    dmm = float(np.linalg.norm((np.array(fu["centroid_zyx"]) - [row.fu_z, row.fu_y, row.fu_x]) * sp))
                    dists.append(dmm)
                    line += f" | centroid error {dmm:6.1f} mm, Dice {dice:.2f}"
                print(line)
    d = np.array(dists)
    print(f"\nmatched lesions {len(d)}: centroid error median {np.median(d):.1f} mm, max {d.max():.1f} mm, "
          f"<= 10 mm {np.mean(d <= 10):.2f}, <= 25 mm {np.mean(d <= 25):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
