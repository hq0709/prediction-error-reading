#!/usr/bin/env python
"""README rule 3 coordinate check for SAT-Pro.

A training-pool LUNA16 scan gets a 10 mm sphere of +300 HU at a known cached index (deep inside the lung, from the
pool scan's lung mask). Original and synthetic arrays both go through common.write_nifti and SAT's official
load_image (data/inference_dataset.py); the sphere is isolated as the difference of the two processed images, then
passed through the same mask -> answer rule and affine back-mapping that run_sat.py uses. The recovered index must lie
within 1 voxel of the known index.
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import affine_mapper, load_cached, mask_to_answer, write_nifti  # noqa: E402
from run_sat import PROMPT, add_sphere  # noqa: E402

os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
sys.path.insert(0, "/raid/home/CAMCA/hj880/ext_src/SAT")
import data.inference_dataset as ids  # noqa: E402

_capt = {}
_orig = ids.Normalization


def _norm(img, t):
    _capt["affine"] = np.asarray(img.affine, dtype=np.float64)
    return _orig(img, t)


ids.Normalization = _norm


def known_index(ds, uid, vol, sp):
    """LUNA16 pool scan: deepest voxel of its lung mask. liver_ext (eval cases only, so no label or mask is read):
    a fixed fractional position of the array, (0.50, 0.45, 0.40) x (shape - 1), rounded."""
    if ds == "luna":
        lung = np.load(f"/raid/home/CAMCA/hj880/perr/cache/luna/{uid}.lung.npy")  # pool scan: allowed
        dist = ndi.distance_transform_edt(lung, sampling=sp)
        return np.array(np.unravel_index(np.argmax(dist), dist.shape), dtype=float)
    return np.array([round(f * (n - 1)) for f, n in zip((0.50, 0.45, 0.40), vol.shape)], dtype=float)


def main():
    args = sys.argv[1:]
    ds = "luna"
    if args[0] == "--ds":
        ds, args = args[1], args[2:]
    uid = args[0]
    work = f"{HERE}/_work/coord"
    os.makedirs(work, exist_ok=True)
    vol, sp = load_cached(ds, uid)
    known = known_index(ds, uid, vol, sp)
    syn = add_sphere(vol, sp, known)
    res = {}
    for name, arr in (("orig", vol), ("syn", syn)):
        p = f"{work}/{uid}_{name}.nii"
        write_nifti(arr, sp, p)
        img, *_ = ids.load_image({"image": p, "label": [PROMPT[ds]], "modality": "ct", "dataset": "coord"})
        res[name] = (img[0].numpy(), _capt["affine"].copy())
        os.remove(p)
    (a, A1), (b, A2) = res["orig"], res["syn"]
    assert a.shape == b.shape and np.allclose(A1, A2), (a.shape, b.shape)
    diff = np.abs(b - a)
    mask = diff > 0.5 * diff.max()
    ans = mask_to_answer(mask, diff / diff.max(), abs(np.linalg.det(A1[:3, :3])), affine_mapper(A1, sp), vol.shape)
    got = np.array(ans["zyx"])
    err_vox = got - known
    err_mm = err_vox * np.array(sp)
    print(f"uid {uid}\nspacing zyx {sp}  cached shape {vol.shape}\nprocessed grid {a.shape} affine\n{np.round(A1, 3)}")
    print(f"known zyx {known.tolist()}\nrecovered zyx {got.tolist()}  ({ans['n_components']} component(s), "
          f"equivalent diameter {ans['diameter_mm']} mm)")
    print(f"error voxels {np.round(err_vox, 3).tolist()}  error mm {np.round(err_mm, 3).tolist()}")
    print("PASS" if np.all(np.abs(err_vox) <= 1.0) else "FAIL")


if __name__ == "__main__":
    main()
