"""Label-free diagnostics of the LesionLocator run (no follow-up labels, no targets).

* tracked fraction of baseline lesions;
* for each tracked lesion: distance (mm) between LesionLocator's follow-up centroid and the location our image-only
  registration gives the same baseline lesion (centroid of its warped baseline label, blw_* in the pair's lesion
  table; BL-derived);
* whether TrackNet's z-window search was active (BL and FU z-extents differ by > 20 slices at its 3 mm spacing).

    python diag_tracking.py plan_eval.json _work/eval
"""
import json
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr")
from perr import autopet as A  # noqa: E402


def nz3mm(path):
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return r.GetSize()[2] * r.GetSpacing()[2] / 3.0


def main(plan_path, work):
    plan = json.load(open(plan_path))
    rows = []
    for pair in plan:
        meta = np.load(f"{A.CACHE}/{pair['key']}.meta.npy", allow_pickle=True).item()
        blw = {r["lesion_id"]: (r["blw_z"], r["blw_y"], r["blw_x"]) for r in meta["lesions"]}
        sp = np.array(meta["spacing"])
        fu_slices = nz3mm(pair["fu"])
        for bl in pair["bl"]:
            d = json.load(open(f"{work}/{pair['key']}/bl{bl['piece']:02d}/done.json"))
            zsearch = abs(nz3mm(bl["img"]) - fu_slices) > 20
            for L in d["lesions"]:
                if L.get("bl_seg_file") is None:
                    continue
                fu = L.get("fu_seg", {})
                tracked = fu.get("nvox", 0) > 0
                dist = np.nan
                lid = L.get("bl_lesion_id")
                if tracked and not bl["binary_mode"] and lid in blw and np.isfinite(blw[lid][0]):
                    dist = float(np.linalg.norm((np.array(fu["centroid_zyx"]) - np.array(blw[lid])) * sp))
                rows.append((pair["key"], bl["piece"], zsearch, tracked, dist))
    tr = np.array([r[3] for r in rows])
    zs = np.array([r[2] for r in rows])
    dist = np.array([r[4] for r in rows])
    ok = np.isfinite(dist)
    print(f"baseline lesions with a baseline segmentation: {len(rows)}; tracked {tr.sum()} ({tr.mean():.2f})")
    print(f"  tracked fraction: z-window search active {tr[zs].mean():.2f} (n={zs.sum()}), "
          f"inactive {tr[~zs].mean():.2f} (n={(~zs).sum()})")
    print(f"tracked lesions vs our registration's location of the same baseline lesion (n={ok.sum()}): "
          f"median {np.nanmedian(dist):.1f} mm, <= 25 mm {np.mean(dist[ok] <= 25):.2f}, > 100 mm {np.mean(dist[ok] > 100):.2f}")
    for flag, name in ((True, "active"), (False, "inactive")):
        m = ok & (zs == flag)
        if m.any():
            print(f"  z-window search {name}: n={m.sum()}, median {np.median(dist[m]):.1f} mm, "
                  f"<= 25 mm {np.mean(dist[m] <= 25):.2f}")
    pieces = {(r[0], r[1]) for r in rows}
    print(f"BL pieces with z-window search active: {sum(1 for p in pieces if any(r[2] for r in rows if (r[0], r[1]) == p))}"
          f"/{len(pieces)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
