#!/usr/bin/env python
"""Smoke-test report for TRAINING-POOL cases only (never eval cases).

LUNA16: annotated nodules come from annotations.csv (world mm) and are mapped to the cached index with SimpleITK on
the LPS-reoriented original .mhd, independently of the cache. Liver: pool-case tumour components from the MSD label.
For each predicted candidate: distance to the nearest annotated lesion and whether it would count as a hit.

    python smoke_check.py luna|liver results.jsonl
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

LUNA = "/raid/home/CAMCA/hj880/data_ext/luna16"
EVAL_LUNA = set(open("/raid/home/CAMCA/hj880/perr/sota/eval_luna_fresh1.txt").read().split()
                + open("/raid/home/CAMCA/hj880/perr/sota/eval_luna_fresh2.txt").read().split())
EVAL_LIVER = set(open("/raid/home/CAMCA/hj880/perr/sota/eval_liver.txt").read().split())


def luna_nodules(uid):
    assert uid not in EVAL_LUNA, "eval case: annotations must not be read"
    img = sitk.DICOMOrient(sitk.ReadImage(glob.glob(f"{LUNA}/subset[0-9]/{uid}.mhd")[0]), "LPS")
    sx, sy, sz = img.GetSpacing()
    ann = pd.read_csv(f"{LUNA}/annotations.csv")
    out = []
    for r in ann[ann.seriesuid == uid].itertuples():
        ix, iy, iz = img.TransformPhysicalPointToContinuousIndex((r.coordX, r.coordY, r.coordZ))
        out.append(((iz, iy, ix), r.diameter_mm))
    return out, (sz, sy, sx)


def main():
    ds, path = sys.argv[1], sys.argv[2]
    for rec in map(json.loads, open(path)):
        c = rec["case"]
        if ds == "luna":
            nods, sp = luna_nodules(c)
            print(f"{c[-12:]} nodules: " + ", ".join(f"{d:.1f}mm@{np.round(z, 1).tolist()}" for z, d in nods))
            for k, cand in enumerate(rec["candidates"]):
                p = np.array(cand["zyx"]) * sp
                dists = [np.linalg.norm(p - np.array(z) * sp) for z, _ in nods]
                j = int(np.argmin(dists)) if dists else -1
                hit = j >= 0 and dists[j] <= nods[j][1] / 2 + 3 and nods[j][1] >= 6
                print(f"   #{k} score {cand['score']:.3f} d {cand['diameter_mm']:.1f}mm zyx {cand['zyx']} "
                      f"nearest {dists[j] if j >= 0 else float('nan'):.1f}mm (nod {nods[j][1] if j >= 0 else 0:.1f}mm) "
                      f"{'HIT' if hit else ''}")
        else:
            assert c not in EVAL_LIVER, "eval case: labels must not be read"
            lab = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(
                f"/raid/home/CAMCA/hj880/msd/Task03_Liver/labelsTr/{c}.nii.gz"), "LPS"))
            from scipy import ndimage as ndi
            sp = np.array(np.load(f"/raid/home/CAMCA/hj880/perr/cache/liver/{c}.meta.npy", allow_pickle=True).item()["spacing"])
            comp, n = ndi.label(lab == 2, structure=np.ones((3, 3, 3)))
            sizes = ndi.sum(np.ones_like(comp), comp, range(1, n + 1)) * sp.prod()
            diam = (6 * sizes / np.pi) ** (1 / 3)
            print(f"{c}: {n} tumour components, diameters " + ", ".join(f"{d:.0f}" for d in sorted(diam, reverse=True)[:6]))
            for k, cand in enumerate(rec["candidates"]):
                z, y, x = (int(round(v)) for v in cand["zyx"])
                lbl = int(lab[z, y, x])
                cid = comp[z, y, x]
                print(f"   #{k} score {cand['score']:.3f} d {cand['diameter_mm']:.1f}mm zyx {cand['zyx']} label {lbl}"
                      f"{f' (tumour {diam[cid - 1]:.0f}mm)' if cid else ''}")


if __name__ == "__main__":
    main()
