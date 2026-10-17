#!/usr/bin/env python
"""README rule 3 coordinate check for VISTA3D.

A training-pool LUNA16 scan gets a 10 mm sphere of +300 HU at a known cached index (deep inside the lung). Original
and synthetic arrays go through common.write_nifti and the bundle's own preprocessing; the sphere is isolated as the
difference of the two processed images, then sent through the exact postprocessing chain of run_vista3d.py
(VistaPostTransformd, Invertd back to the input grid, nan_to_num), transposed to [z, y, x] and reduced with the same
mask -> answer rule. The recovered index must lie within 1 voxel of the known index.
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_cached, mask_to_answer, write_nifti  # noqa: E402
from run_vista3d import LABEL, add_sphere, build  # noqa: E402


def main():
    import torch
    from monai.transforms import reset_ops_id

    args = sys.argv[1:]
    ds = "luna"
    if args[0] == "--ds":
        ds, args = args[1], args[2:]
    uid = args[0]
    work = f"{HERE}/_work/coord"
    os.makedirs(work, exist_ok=True)
    vol, sp = load_cached(ds, uid)
    if ds == "luna":
        lung = np.load(f"/raid/home/CAMCA/hj880/perr/cache/luna/{uid}.lung.npy")  # pool scan: allowed
        dist = ndi.distance_transform_edt(lung, sampling=sp)
        known = np.array(np.unravel_index(np.argmax(dist), dist.shape), dtype=float)
    else:  # liver_ext: eval cases only, so no label or mask is read; fixed fractional position of the array
        known = np.array([round(f * (n - 1)) for f, n in zip((0.50, 0.45, 0.40), vol.shape)], dtype=float)
    lab = LABEL[ds]
    po, ps = f"{work}/{uid}_orig.nii", f"{work}/{uid}_syn.nii"
    write_nifti(vol, sp, po)
    write_nifti(add_sphere(vol, sp, known), sp, ps)
    parser, pre, _ = build(work, [{"image": ps, "label_prompt": [lab]}])
    post = parser.get_parsed_content("postprocessing")
    a = pre({"image": po, "label_prompt": [lab]})["image"]
    b = pre({"image": ps, "label_prompt": [lab]})["image"]
    print(f"uid {uid}\nspacing zyx {sp}  cached shape {vol.shape}\nprocessed grid {tuple(b.shape)} affine\n"
          f"{np.round(np.asarray(b.affine), 3)}")
    diff = (b - a).abs()
    mask = (diff > 0.5 * diff.max()).float()
    # `mask` stays a MetaTensor, like the inferer output in the real run (Invertd skips plain tensors)
    data = {"image": reset_ops_id(b), "pred": mask, "label_prompt": torch.tensor([[lab]], device=b.device)}
    out = post(data)
    m = out["pred"][0].cpu().numpy().transpose(2, 1, 0)
    assert m.shape == vol.shape, (m.shape, vol.shape)
    ans = mask_to_answer((m > 0) & (m != 255), out["prob"][0].float().cpu().numpy().transpose(2, 1, 0),
                         float(np.prod(sp)), lambda idx: np.asarray(idx, dtype=np.float64), vol.shape)
    got = np.array(ans["zyx"])
    err_vox = got - known
    print(f"known zyx {known.tolist()}\nrecovered zyx {got.tolist()}  ({ans['n_components']} component(s), "
          f"equivalent diameter {ans['diameter_mm']} mm)")
    print(f"error voxels {np.round(err_vox, 3).tolist()}  error mm {np.round(err_vox * np.array(sp), 3).tolist()}")
    print("PASS" if np.all(np.abs(err_vox) <= 1.0) else "FAIL")
    os.remove(po)
    os.remove(ps)


if __name__ == "__main__":
    main()
