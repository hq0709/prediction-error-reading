#!/usr/bin/env python
"""README rule 3 coordinate check for SegVol.

A 10 mm sphere of +300 HU is put at a known cached index:
  * luna: training-pool scan, deepest voxel of its lung mask (pool scan: allowed);
  * liver: training-pool MSD case, deepest voxel of its liver mask (pool case: allowed);
  * liver_ext: eval cases only, so no label or mask is read: fixed fraction (0.50, 0.45, 0.40) of the array shape.
I/O path: original and synthetic arrays both go through run_segvol.write_nifti_ras and the official SegVol
preprocessing (run_segvol.preprocess = CT branch of preprocess_ct_gt + zoom_transform); the sphere is isolated as the
difference of the two processed images and reduced with the run's mask -> answer rule and back-mapping
(processed_to_cache). PASS = recovered index within 1 voxel of the known index on every axis (exit code 1 otherwise).
End to end: SegVol forward_test on both volumes with the dataset's prompt; reports the rank-1 answer, the component
nearest the sphere and the fraction of sphere voxels inside the mask.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_cached, mask_to_answer  # noqa: E402
from run_segvol import (PROMPT, add_sphere, build_model, load_segvol_module, preprocess,  # noqa: E402
                        processed_to_cache, write_nifti_ras)

EVAL = {"luna": ["eval_luna_fresh1.txt", "eval_luna_fresh2.txt"], "liver": ["eval_liver.txt"], "liver_ext": []}


def known_index(ds, case, vol, sp):
    if ds == "luna":
        m = np.load(f"/raid/home/CAMCA/hj880/perr/cache/luna/{case}.lung.npy")  # pool scan: allowed
    elif ds == "liver":
        m = np.load(f"/raid/home/CAMCA/hj880/perr/cache/liver/{case}.liver.npy")  # pool case: allowed
    else:
        return np.array([round(f * (n - 1)) for f, n in zip((0.50, 0.45, 0.40), vol.shape)], dtype=float)
    dist = ndi.distance_transform_edt(m, sampling=sp)
    return np.array(np.unravel_index(np.argmax(dist), dist.shape), dtype=float)


def comp_report(ans, known, sp, sphere_mask, mask_cache):
    out = {"n_components": ans["n_components"], "rank1": ans["candidates"][0] if ans["present"] else None}
    if ans["present"]:
        d = [float(np.linalg.norm((np.array(c["zyx"]) - known) * np.array(sp))) for c in ans["candidates"]]
        j = int(np.argmin(d))
        out["nearest_in_top10"] = {"rank": j, "dist_mm": round(d[j], 2), **ans["candidates"][j]}
    out["frac_sphere_voxels_in_mask"] = round(float(mask_cache[sphere_mask].mean()), 3)
    return out


def main():
    args = [x for x in sys.argv[1:] if x != "--io_only"]
    ds = "luna"
    if args[0] == "--ds":
        ds, args = args[1], args[2:]
    case = args[0]
    for f in EVAL[ds]:
        assert case not in open(f"/raid/home/CAMCA/hj880/perr/sota/{f}").read().split(), "use a training-pool case"
    work = f"{HERE}/_work/coord"
    os.makedirs(work, exist_ok=True)
    vol, sp = load_cached(ds, case)
    known = known_index(ds, case, vol, sp)
    syn = add_sphere(vol, sp, known)
    sphere_mask = syn != vol
    processor = load_segvol_module().SegVolProcessor(spatial_size=[32, 256, 256])
    items = {}
    for name, arr in (("orig", vol), ("syn", syn)):
        p = f"{work}/{case}_{name}.nii"
        write_nifti_ras(arr, sp, p)
        items[name] = preprocess(processor, p)
        os.remove(p)
    a, b = items["orig"], items["syn"]
    sa, sb = np.asarray(a["foreground_start_coord"]), np.asarray(b["foreground_start_coord"])
    assert a["image"].shape == b["image"].shape and np.array_equal(sa, sb), (a["image"].shape, b["image"].shape, sa, sb)
    diff = np.abs(b["image"][0].numpy() - a["image"][0].numpy())
    mask = diff > 0.5 * diff.max()
    ans = mask_to_answer(mask, diff / diff.max(), float(np.prod(sp)), processed_to_cache(sa, vol.shape), vol.shape)
    got = np.array(ans["zyx"])
    err = got - known
    ok = bool(np.all(np.abs(err) <= 1.0))
    res = {"ds": ds, "case": case, "spacing_zyx": list(sp), "shape": list(vol.shape), "known_zyx": known.tolist(),
           "sphere": "10 mm, +300 HU", "processed_grid": list(a["image"].shape[1:]), "crop_start": sa.tolist(),
           "io": {"recovered_zyx": got.tolist(), "err_vox": np.round(err, 3).tolist(),
                  "err_mm": np.round(err * np.array(sp), 3).tolist(), "n_components": ans["n_components"],
                  "equivalent_diameter_mm": ans["diameter_mm"], "pass": ok}}
    print(json.dumps(res["io"]), flush=True)
    if "--io_only" in sys.argv:
        print("PASS" if ok else "FAIL", "(I/O path only)", flush=True)
        sys.exit(0 if ok else 1)

    # end to end with the model (GPU)
    import torch

    device = torch.device("cuda", 0)
    _, model = build_model(device)
    res["model"] = {"prompt": PROMPT[ds]}
    for name in ("syn", "orig"):
        it = items[name]
        with torch.no_grad():
            logits = model.forward_test(image=it["image"].unsqueeze(0).to(device),
                                        zoomed_image=it["zoom_out_image"].unsqueeze(0).to(device),
                                        text_prompt=[PROMPT[ds]], use_zoom=True)
        prob = torch.sigmoid(logits[0, 0].float()).numpy()
        m = prob > 0.5
        st = np.asarray(it["foreground_start_coord"])
        ans_m = mask_to_answer(m, prob, float(np.prod(sp)), processed_to_cache(st, vol.shape), vol.shape)
        # probability and mask on the cached grid: uncrop to the (k, j, i) grid, then flip j and i (y = ny-1-j, x = nx-1-i)
        d, h, w = m.shape
        pad = [(st[0], vol.shape[0] - st[0] - d), (st[1], vol.shape[1] - st[1] - h), (st[2], vol.shape[2] - st[2] - w)]
        prob_cache = np.pad(prob, pad)[:, ::-1, ::-1]
        res["model"][name] = comp_report(ans_m, known, sp, sphere_mask, prob_cache > 0.5)
        res["model"][name]["max_prob_in_sphere"] = round(float(prob_cache[sphere_mask].max()), 4)
        print(name, json.dumps(res["model"][name]), flush=True)
    res["pass"] = ok
    json.dump(res, open(f"{HERE}/coord_check_{ds}.json", "w"), indent=1)
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
