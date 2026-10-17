#!/usr/bin/env python
"""README rule 3 coordinate check for OmniAbnorm-CT.

LUNA16 (default): a training-pool scan gets a 10 mm sphere of +300 HU at a known cached index (the deepest lung
voxel of the pool scan's lung mask, the same scan and point as sota/sat_pro and sota/vista3d).
External liver set (`--ds liver_ext --case HCC_002`): every cached case is an eval case, so nothing but the image is
read; the sphere goes to the middle slice, at the body's centre row and 30 % of the body width from the patient's
right edge (right upper quadrant: the liver on these liver-centred scans); the HU there is recorded.

(a) I/O path. Original and synthetic arrays go through the run's own preprocessing (`SliceSource` = window +
    uint8 + z-score, `slice_sample` = inference.py `preprocess_input`) for every key slice near the sphere. The sphere is isolated as
    the difference of the two 9-channel U-Net inputs; channel c of key slice z is mapped back with the official
    `recover_single_slice` to cached slice z + 4 - c (the cranio-caudal context order of `slice_sample`), the 3D
    difference is reduced with the run's mask -> answer rule, and the result must lie within 1 voxel of the known
    index. The same is done for the VLM image (key slice only).
(b) End to end. The run's `sweep_cases` (DataLoader workers -> VLM <SEG> -> U-Net -> recover -> stack) on the
    synthetic and the original array over the slices around the sphere; reports the component nearest the sphere.

    python coord_check.py                                  # LUNA16 pool scan -> coord_check.json
    python coord_check.py --ds liver_ext --case HCC_002    # -> coord_check_liver_ext.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_omniabnorm as R  # noqa: E402
from common import add_sphere, load_cached, mask_to_answer  # noqa: E402

UID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275"  # results/train_uids.txt (pool)


def known_from_lung_mask(case, vol, sp):
    assert case in open("/raid/home/CAMCA/hj880/perr/results/train_uids.txt").read().split(), "pool scans only"
    lung = np.load(f"/raid/home/CAMCA/hj880/perr/cache/luna/{case}.lung.npy")  # pool scan: allowed
    dist = ndi.distance_transform_edt(lung, sampling=sp)
    return np.array(np.unravel_index(np.argmax(dist), dist.shape), dtype=float)


def known_from_image(vol):
    """Image only (eval case): middle slice; body = largest connected region with HU > -500; centre row of the body,
    30 % of its width from the patient's right edge (x grows toward the patient's left)."""
    z = vol.shape[0] // 2
    lab, n = ndi.label(vol[z] > -500)
    body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    ys, xs = np.nonzero(body)
    return np.array([z, round(ys.mean()), round(xs.min() + 0.3 * (xs.max() - xs.min()))], dtype=float)


def io_check(ds, vol, syn, sp, known):
    za, zb = R.SliceSource(vol, R.WINDOWS[ds], cache=1000), R.SliceSource(syn, R.WINDOWS[ds], cache=1000)
    kz = int(round(known[0]))
    acc = np.zeros(vol.shape, np.float64)
    cnt = np.zeros(vol.shape[0], np.float64)
    vlm = np.zeros(vol.shape, np.float64)
    per_channel = []
    for z in range(max(0, kz - 8), min(vol.shape[0], kz + 9)):
        (ra, sa, ta), (rb, sb, tb) = R.slice_sample(za, z, R.PROMPTS[ds]), R.slice_sample(zb, z, R.PROMPTS[ds])
        d = (sb - sa).abs().numpy()
        for c in range(9):
            zz = z + R.CONTEXT - c
            if 0 <= zz < vol.shape[0]:
                acc[zz] += R.oa.recover_single_slice(d[c].astype(np.float32), ta)
                cnt[zz] += 1
        per_channel.append([round(float(d[c].max()), 3) for c in range(9)])
        pa = np.asarray(ra[1]["content"][0]["image"], np.float64)  # the PIL image handed to the VLM
        pb = np.asarray(rb[1]["content"][0]["image"], np.float64)
        vlm[z] = np.abs(pb - pa)
    acc[cnt > 0] /= cnt[cnt > 0][:, None, None]
    out = {}
    for name, diff in (("unet_input_all_channels", acc), ("vlm_image", vlm)):
        m = diff > 0.5 * diff.max()
        ans = mask_to_answer(m, diff / diff.max(), sp)
        got = np.array(ans["zyx"])
        out[name] = {"recovered_zyx": got.tolist(), "err_vox": np.round(got - known, 3).tolist(),
                     "err_mm": np.round((got - known) * np.array(sp), 3).tolist(), "n_components": ans["n_components"],
                     "pass": bool(np.all(np.abs(got - known) <= 1.0))}
    out["max_abs_diff_per_channel_for_key_slices_kz-8..kz+8"] = per_channel
    return out


def model_check(M, ds, vol, syn, sp, known, half=12):
    kz = int(round(known[0]))
    zs = list(range(min(vol.shape[0] - 1, kz + half), max(-1, kz - half - 1), -1))
    arrays = {"synthetic": syn, "original": vol}
    res = {}
    for name, acc in R.sweep_cases(M, ds, list(arrays), {k: v.shape for k, v in arrays.items()}, 32, 8,
                                   arrays=arrays, key_slices={k: zs for k in arrays}):
        mask, prob, diag = acc.mask, acc.prob, acc.diag()
        ans = mask_to_answer(mask, prob, sp, topk=1000)
        near = None
        for rank, c in enumerate(ans["candidates"]):
            dmm = float(np.linalg.norm((np.array(c["zyx"]) - known) * np.array(sp)))
            if near is None or dmm < near["dist_mm"]:
                near = {"rank": rank, "dist_mm": round(dmm, 2), **{k: c[k] for k in ("zyx", "diameter_mm", "score", "n_vox")}}
        sphere_vox = mask[add_sphere(np.zeros(vol.shape, np.int16), sp, known, hu=1) > 0]
        res[name] = {"key_slices": [zs[-1], zs[0]], "n_seg": diag["n_seg"], "n_mask": diag["n_mask"],
                     "n_components": ans["n_components"], "nearest_component": near,
                     "frac_sphere_voxels_in_mask": round(float(sphere_vox.mean()), 3),
                     "rank1": ans["candidates"][0] if ans["present"] else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="luna", choices=["luna", "liver_ext"])
    ap.add_argument("--case", default=UID)
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args()
    vol, sp = load_cached(args.ds, args.case)
    if args.ds == "luna":
        known, where = known_from_lung_mask(args.case, vol, sp), "deepest voxel of the pool scan's lung mask"
    else:
        known, where = known_from_image(vol), "image only: middle slice, body centre row, 30 % of body width from the right"
    syn = add_sphere(vol, sp, known)
    zi, yi, xi = (int(v) for v in known)
    out = {"ds": args.ds, "case": args.case, "spacing_zyx": sp, "shape": list(vol.shape), "known_zyx": known.tolist(),
           "placement": where, "hu_at_point": int(vol[zi, yi, xi]),
           "hu_mean_3x3x3": round(float(vol[max(zi - 1, 0):zi + 2, yi - 1:yi + 2, xi - 1:xi + 2].mean()), 1),
           "sphere": "10 mm, +300 HU", "io": io_check(args.ds, vol, syn, sp, known)}
    print(json.dumps({k: v for k, v in out.items() if k != "io"}), flush=True)
    print(json.dumps(out["io"], indent=1), flush=True)
    if not args.no_model:
        M, info = R.load_model()
        out["model"] = model_check(M, args.ds, vol, syn, sp, known)
        print(json.dumps(out["model"], indent=1), flush=True)
    out["pass"] = bool(out["io"]["unet_input_all_channels"]["pass"] and out["io"]["vlm_image"]["pass"])
    name = "coord_check.json" if args.ds == "luna" else f"coord_check_{args.ds}.json"
    with open(f"{HERE}/{name}", "w") as f:
        json.dump(out, f, indent=1)
    print("PASS" if out["pass"] else "FAIL")


if __name__ == "__main__":
    main()
