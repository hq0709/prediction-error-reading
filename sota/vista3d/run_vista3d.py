#!/usr/bin/env python
"""VISTA3D (He et al., CVPR 2025) on the cached LUNA16 / MSD-liver arrays via the official MONAI bundle.

Bundle: MONAI/vista3d 0.5.11 (HF snapshot c6dbe159), configs/inference.json, models/model.pt. Everything that touches
the image comes from that config: LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd 1.5 mm, CropForegroundd,
VistaPreTransformd, ScaleIntensityRanged [-963.8, 1053.7] -> [0, 1], Orientationd RAS; network vista3d132 with the
bundle checkpoint; scripts.inferer.Vista3dInferer (sliding window 128^3, overlap 0.3, replicate padding);
scripts.evaluator.Vista3dEvaluator (amp, user_prompt mode, automatic segmentation from a class label prompt, no points).

Postprocessing = the bundle's chain (VistaPostTransformd, Invertd with nearest interpolation back to the input image
grid, nan_to_num) without SaveImaged, plus one side branch: a copy of the logits passed through a sigmoid and the
same Invertd, which gives the per-voxel probability used to score components. The inverted mask lives on the grid of
the NIfTI written from the cached array; transposing (x, y, z) -> (z, y, x) gives the cached index directly.
"""
import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.dont_write_bytecode = True  # do not write __pycache__ into the HF snapshot when importing the bundle's scripts
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_cached, load_spacing, mask_to_answer, write_nifti  # noqa: E402

BUNDLE = "/raid/home/CAMCA/hj880/hf_home/hub/models--MONAI--vista3d/snapshots/c6dbe159632a4767696e09f91d74d729b82e73e6"
# class prompts in VISTA3D's vocabulary (configs/metadata.json): 23 "lung tumor"; 26 "hepatic tumor".
# 130 "liver tumor" is a legacy index the bundle rejects without point prompts ("already have hepatic tumor").
LABEL = {"luna": 23, "liver": 26, "liver_ext": 26}


def add_sphere(vol, sp, zyx, diam=10.0, hu=300):
    vol = vol.copy()
    zz, yy, xx = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    d2 = ((zz - zyx[0]) * sp[0]) ** 2 + ((yy - zyx[1]) * sp[1]) ** 2 + ((xx - zyx[2]) * sp[2]) ** 2
    vol[d2 <= (diam / 2) ** 2] = hu
    return vol


def _write(job):
    ds, case, path, sphere = job
    if not os.path.exists(path):
        vol, sp = load_cached(ds, case)
        if sphere is not None:
            vol = add_sphere(vol, sp, sphere)
        write_nifti(vol, sp, path + ".tmp.nii")
        os.replace(path + ".tmp.nii", path)
    return path


def build(work, data_list):
    """Official bundle config with three overrides: bundle_root, the data list, and the postprocessing chain
    (official entries reused verbatim, SaveImaged dropped, probability branch added)."""
    from monai.bundle import ConfigParser

    sys.path.insert(0, BUNDLE)  # the config does "$import scripts"
    parser = ConfigParser()
    parser.read_config(f"{BUNDLE}/configs/inference.json")
    parser["bundle_root"] = BUNDLE
    parser["output_dir"] = work
    parser["dataset#data"] = data_list
    off = copy.deepcopy(parser["postprocessing"]["transforms"])
    targets = [t["_target_"] for t in off]
    assert targets == ["ToDeviced", "monai.apps.vista3d.transforms.VistaPostTransformd", "Invertd", "Lambdad",
                       "SaveImaged"], targets
    assert off[0].get("_disabled_") is True
    parser["postprocessing"] = {"_target_": "Compose", "transforms": [
        {"_target_": "CopyItemsd", "keys": "pred", "names": "prob"},
        {"_target_": "Lambdad", "keys": "prob", "func": "$lambda x: torch.sigmoid(x.float())"},
        off[1],                                  # VistaPostTransformd (logit > 0 -> label value)
        {**off[2], "keys": ["pred", "prob"]},    # Invertd, nearest_interp=True, to the input image grid
        off[3],                                  # nan_to_num(pred, 255)
    ]}
    evaluator = parser.get_parsed_content("evaluator")
    preprocessing = parser.get_parsed_content("preprocessing")
    return parser, preprocessing, evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["luna", "liver", "liver_ext"], required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=f"{HERE}/_work")
    ap.add_argument("--label", type=int, default=None, help="override the class prompt (pool smoke tests only)")
    ap.add_argument("--sphere", default=None, help="z,y,x: insert a synthetic sphere (coordinate check only)")
    ap.add_argument("--save_npz", default=None)
    ap.add_argument("--nproc", type=int, default=16, help="processes writing the NIfTI inputs")
    a = ap.parse_args()
    label = a.label if a.label is not None else LABEL[a.ds]

    cases = open(a.cases).read().split()
    done = {json.loads(l)["case"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
    todo = [c for c in cases if c not in done]
    print(f"{len(cases)} cases, {len(done)} done, {len(todo)} to run; label prompt {label}", flush=True)
    if not todo:
        return
    sphere = tuple(float(v) for v in a.sphere.split(",")) if a.sphere else None
    ndir = f"{a.work}/nii_{'sphere' if sphere else a.ds}"
    os.makedirs(ndir, exist_ok=True)
    paths = {c: f"{ndir}/{c}.nii" for c in todo}
    t = time.time()
    with Pool(a.nproc) as p:
        list(p.imap(_write, [(a.ds, c, paths[c], sphere) for c in todo]))
    print(f"wrote {len(todo)} NIfTI files in {time.time() - t:.0f}s", flush=True)

    import torch
    from ignite.engine import Events

    parser, _, evaluator = build(a.work, [{"image": paths[c], "label_prompt": [label]} for c in todo])
    parser.get_parsed_content("initialize")  # set_determinism(seed=123) + CheckpointLoader(models/model.pt)
    by_path = {v: k for k, v in paths.items()}
    fout = open(a.out, "a")
    if a.save_npz:
        os.makedirs(a.save_npz, exist_ok=True)
    state = {"t": time.time()}

    def collect(engine):
        out = engine.state.output[0]
        pred, prob = out["pred"], out["prob"]
        case = by_path[str(pred.meta["filename_or_obj"])]
        vol_shape = np.load(f"/raid/home/CAMCA/hj880/perr/cache/{a.ds}/{case}.vol.npy", mmap_mode="r").shape
        sp = load_spacing(a.ds, case)
        m = pred[0].detach().cpu().numpy().transpose(2, 1, 0)
        pr = prob[0].detach().float().cpu().numpy().transpose(2, 1, 0)
        assert m.shape == tuple(vol_shape), (m.shape, vol_shape)
        n255 = int((m == 255).sum())
        mask = (m > 0) & (m != 255)
        ans = mask_to_answer(mask, pr, sp[0] * sp[1] * sp[2], lambda idx: np.asarray(idx, dtype=np.float64), vol_shape)
        dt = time.time() - state["t"]
        rec = {"case": case, **{k: ans[k] for k in ("present", "zyx", "diameter_mm", "score", "candidates")},
               "raw": json.dumps({"label_prompt": label, "n_components": ans["n_components"], "mask_vox": int(mask.sum()),
                                  "max_prob": round(float(pr.max()), 4), "nan_vox": n255,
                                  "clipped": ans.get("clipped", False), "t_case_s": round(dt, 1)})}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if a.save_npz:
            np.savez_compressed(f"{a.save_npz}/{case}.npz", prob=pr.astype(np.float16), mask=mask)
        try:
            os.remove(paths[case])
        except FileNotFoundError:
            pass
        print(f"{case} present={ans['present']} ncomp={ans['n_components']} zyx={ans['zyx']} d={ans['diameter_mm']} "
              f"s={ans['score']} {dt:.1f}s", flush=True)
        state["t"] = time.time()

    evaluator.add_event_handler(Events.ITERATION_COMPLETED, collect)
    t = time.time()
    evaluator.run()
    print(f"total {time.time() - t:.0f}s for {len(todo)} cases", flush=True)


if __name__ == "__main__":
    main()
