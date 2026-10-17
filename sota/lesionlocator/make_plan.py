"""Inputs for the LesionLocator runs, one entry per pair "<pid>_<NN>" (FU piece NN on its native grid).

Run with the perr venv. Reads only baseline-side information and image geometry:
  * the original FU image inputsTr/<pid>_FU_img_NN.nii.gz (LesionLocator's input and output grid);
  * the BL pieces that our image-only registration matched to this FU piece (meta["bl_order"]; no labels involved),
    i.e. the BL series whose annotated lesions form the prior findings our own method receives for this pair;
  * for each such BL piece: the original image, its lesion mask (inputsTr/<pid>_BL_mask_QQ.nii.gz) and the lesion ids
    and voxel counts in that mask (baseline annotations).
Never reads FU masks, FU lesion tables or targets.

    python make_plan.py eval  ../eval_course.txt  plan_eval.json
    python make_plan.py train <key> [<key> ...]   plan_sanity.json
"""
import json
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr")
from perr import autopet as A  # noqa: E402


def entry(key: str) -> dict:
    pid, nn = key.split("_")
    meta = np.load(f"{A.CACHE}/{key}.meta.npy", allow_pickle=True).item()
    fu_path = f"{A.ROOT}/inputsTr/{pid}_FU_img_{nn}.nii.gz"
    r = sitk.ImageFileReader()
    r.SetFileName(fu_path)
    r.ReadImageInformation()
    size, sp, direction = r.GetSize(), r.GetSpacing(), r.GetDirection()
    assert np.allclose(direction, np.eye(3).ravel()), (key, direction)   # LPS identity: array [z, y, x] == cached grid
    assert tuple(meta["shape"]) == tuple(size[::-1]), (key, meta["shape"], size)
    bl = []
    for q in meta["bl_order"]:
        img = f"{A.ROOT}/inputsTr/{pid}_BL_img_{q:02d}.nii.gz"
        msk = f"{A.ROOT}/inputsTr/{pid}_BL_mask_{q:02d}.nii.gz"
        m = sitk.ReadImage(msk)
        a = sitk.GetArrayFromImage(m)
        ids, cnt = np.unique(a[a > 0], return_counts=True)
        vox_ml = float(np.prod(m.GetSpacing())) / 1000.0
        bl.append({"piece": int(q), "img": img, "mask": msk,
                   "lesions": {str(int(i)): {"nvox": int(c), "ml": float(c * vox_ml)} for i, c in zip(ids, cnt)},
                   "binary_mode": bool(len(ids) == 1 and ids[0] == 1)})   # LesionLocator splits a label-1-only mask by cc3d
    return {"key": key, "fu": fu_path, "fu_shape_zyx": list(size[::-1]), "fu_spacing_zyx": list(sp[::-1]),
            "bl_pieces_all": A.pieces(pid, "BL"), "bl": bl}


if __name__ == "__main__":
    split, *rest = sys.argv[1:]
    out = rest[-1]
    keys = open(rest[0]).read().split() if split == "eval" else rest[:-1]
    plan = [entry(k) for k in keys]
    json.dump(plan, open(out, "w"), indent=1)
    n = sum(len(b["lesions"]) for p in plan for b in p["bl"])
    print(f"{len(plan)} pairs, {sum(len(p['bl']) for p in plan)} BL pieces, {n} BL lesions -> {out}")
    for p in plan:
        dropped = sorted(set(p["bl_pieces_all"]) - {b["piece"] for b in p["bl"]})
        if dropped or any(b["binary_mode"] for b in p["bl"]):
            print(p["key"], "dropped BL pieces", dropped, "binary-mode pieces", [b["piece"] for b in p["bl"] if b["binary_mode"]])
