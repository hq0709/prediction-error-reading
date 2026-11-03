#!/usr/bin/env python
"""README rule 3 coordinate check for the CLIP-Driven Universal Model.

A 10 mm sphere of +300 HU is put at a known cached index:
  * luna: training-pool scan, deepest voxel of its lung mask (pool scan: allowed);
  * liver: training-pool MSD case, deepest voxel of its liver mask (pool case: allowed);
  * liver_ext: eval cases only, so no label or mask is read: fixed fraction (0.50, 0.45, 0.40) of the array shape.
I/O path: original and synthetic arrays both go through common.write_nifti and the official transforms
(get_loader_without_gt) via the run's DataLoader; the sphere is isolated as the difference of the two processed images,
inverted with the run's Invertd (nearest) back to the input grid, transposed to [z, y, x] and reduced with the same
mask -> answer rule. PASS = recovered index within 1 voxel on every axis (exit code 1 otherwise).
End to end: the model on both volumes (the run's predict + invert); reports the rank-1 answer, the component nearest
the sphere and the fraction of sphere voxels inside the mask.
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
from run_clip_universal import (TARGET, add_sphere, build_model, import_official, invert_to_cache,  # noqa: E402
                                make_loader, official_args, predict)

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


def comp_report(ans, known, sp, sphere_mask, mask_cache, prob_cache):
    out = {"n_components": ans["n_components"], "rank1": ans["candidates"][0] if ans["present"] else None}
    if ans["present"]:
        d = [float(np.linalg.norm((np.array(c["zyx"]) - known) * np.array(sp))) for c in ans["candidates"]]
        j = int(np.argmin(d))
        out["nearest_in_top10"] = {"rank": j, "dist_mm": round(d[j], 2), **ans["candidates"][j]}
    out["frac_sphere_voxels_in_mask"] = round(float(mask_cache[sphere_mask].mean()), 3)
    out["max_prob_in_sphere"] = round(float(prob_cache[sphere_mask].max()), 4)
    return out


def main():
    io_only = "--io_only" in sys.argv
    args = [x for x in sys.argv[1:] if x != "--io_only"]
    ds = "luna"
    if args[0] == "--ds":
        ds, args = args[1], args[2:]
    case = args[0]
    for f in EVAL[ds]:
        assert case not in open(f"/raid/home/CAMCA/hj880/perr/sota/{f}").read().split(), "use a training-pool case"
    vol, sp = load_cached(ds, case)
    known = known_index(ds, case, vol, sp)
    sphere_mask = add_sphere(vol, sp, known) != vol
    target = TARGET[ds]
    dl, uu, Universal_model = import_official()
    batches = {}
    for name, sph in (("orig", None), ("syn", tuple(known))):
        ndir = f"{HERE}/_work/coord_{name}_{os.getpid()}"
        os.makedirs(ndir, exist_ok=True)
        _, val_transforms = dl.get_loader_without_gt(official_args(ndir))
        batches[name] = (next(iter(make_loader(ds, [case], ndir, val_transforms, 0, sphere=sph))), val_transforms)
        os.remove(f"{ndir}/{case}.nii")
        os.rmdir(ndir)
    (ba, ta), (bb, tb) = batches["orig"], batches["syn"]
    a, b = ba["image"].float(), bb["image"].float()
    assert a.shape == b.shape, (a.shape, b.shape)
    diff = (b - a).abs()
    mask = (diff > 0.5 * diff.max()).float()
    inv = invert_to_cache(bb, {"m": mask, "p": diff / diff.max()}, tb)
    assert inv["m"].shape == vol.shape, (inv["m"].shape, vol.shape)
    ans = mask_to_answer(inv["m"] > 0.5, inv["p"], float(np.prod(sp)), lambda idx: np.asarray(idx, dtype=np.float64),
                         vol.shape)
    got = np.array(ans["zyx"])
    err = got - known
    ok = bool(np.all(np.abs(err) <= 1.0))
    res = {"ds": ds, "case": case, "spacing_zyx": list(sp), "shape": list(vol.shape), "known_zyx": known.tolist(),
           "sphere": "10 mm, +300 HU", "processed_grid": list(b.shape[2:]),
           "processed_affine": np.round(np.asarray(bb["image_meta_dict"]["affine"][0]), 3).tolist(),
           "io": {"recovered_zyx": got.tolist(), "err_vox": np.round(err, 3).tolist(),
                  "err_mm": np.round(err * np.array(sp), 3).tolist(), "n_components": ans["n_components"],
                  "equivalent_diameter_mm": ans["diameter_mm"], "pass": ok}}
    print(json.dumps({k: res[k] for k in ("processed_grid", "processed_affine")}), flush=True)
    print(json.dumps(res["io"]), flush=True)
    if io_only:
        print("PASS" if ok else "FAIL", "(I/O path only)", flush=True)
        sys.exit(0 if ok else 1)

    model = build_model(Universal_model, uu.NUM_CLASS)
    args_ns = official_args(None)
    organ_name = uu.ORGAN_NAME[target - 1]
    res["model"] = {"target": target, "class": organ_name}
    for name in ("syn", "orig"):
        batch, vt = batches[name]
        m, p = predict(model, uu, batch, target, args_ns)
        inv = invert_to_cache(batch, {organ_name: m, "prob": p}, vt)
        binm = inv[organ_name] > 0.5
        ans_m = mask_to_answer(binm, inv["prob"], float(np.prod(sp)), lambda idx: np.asarray(idx, dtype=np.float64),
                               vol.shape)
        res["model"][name] = comp_report(ans_m, known, sp, sphere_mask, binm, inv["prob"])
        print(name, json.dumps(res["model"][name]), flush=True)
    res["pass"] = ok
    json.dump(res, open(f"{HERE}/coord_check_{ds}.json", "w"), indent=1)
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
