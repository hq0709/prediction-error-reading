#!/usr/bin/env python
"""SegVol (Du et al., NeurIPS 2024) on the cached LUNA16 / MSD-liver / HCC-TACE-Seg arrays: official HF quickstart
code (BAAI/SegVol snapshot 516701d5, model_segvol_single.py) and weights, text prompt only.

Per case:
  cached LPS array -> NIfTI stored in RAS order (write_nifti_ras: i -> patient right, j -> anterior, k -> superior),
     the orientation the official demo pipeline asks for (GitHub data_process/demo_data_process.py starts with
     Orientationd(axcodes="RAS")) and the orientation of the official demo case; SegVol's own loader does not reorient
  -> official SegVolProcessor, CT branch of preprocess_ct_gt verbatim (MONAI LoadImage, squeeze, channel dim,
     ForegroundNorm: clip to the 0.05/99.95 percentiles of the voxels above the volume mean, z-score with their
     mean/std), then zoom_transform (DimTranspose, MinMaxNormalization, CropForegroundd, ToTensord; zoom-out view =
     Resized to 32x256x256, nearest-exact). No label exists: an all-zero one-category mask fills the label slot (the
     transforms only crop/resize it; it is never read).
  -> official SegVolModel.forward_test(image, zoomed_image, text_prompt=[prompt], use_zoom=True): zoom-out inference
     on the 32x256x256 view, ROI = bounding box of its sigmoid > 0.5 voxels padded to the 32x256x256 window, zoom-in
     sliding-window inference on the ROI at native resolution (window 32x256x256, overlap 0.5, constant blending);
     outside the ROI the nearest-upsampled zoom-out logits remain. Prompt template (inside the model):
     "A computerized tomography of a {prompt}."
  -> probability = sigmoid(logits), mask = probability > 0.5 (the official save_preds threshold)
  -> fixed mask -> answer rule (common.mask_to_answer) on the processed grid, which is the cached voxel grid cropped
     and flipped (no resampling): voxel volume = sz*sy*sx; centroids map back through the crop offset and the two flips.

Weights are read with torch.load(weights_only=True) into SegVolModel(config) with strict=True (all 475 tensors), instead
of AutoModel.from_pretrained, whose transformers 4.18 loader unpickles without weights_only. Same module, same weights.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True  # nothing is written into the downloaded code directory
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_cached, load_spacing, mask_to_answer  # noqa: E402

HF_DIR = "/raid/home/CAMCA/hj880/ext_src/segvol_hf"  # BAAI/SegVol @ 516701d5bef430604a13f5db7ce896149d075663
CACHE = "/raid/home/CAMCA/hj880/perr/cache"
# SegVol vocabulary (paper Table 7 / internal validation categories): "lung tumor" (MSD-lung) is its only lung-lesion
# class (no nodule class in any of its 25 training datasets); "liver tumor" (MSD-liver, MSD-hepatic-vessel, 3D-IRCADb).
PROMPT = {"luna": "lung tumor", "liver": "liver tumor", "liver_ext": "liver tumor"}


def load_segvol_module():
    spec = importlib.util.spec_from_file_location("model_segvol_single", f"{HF_DIR}/model_segvol_single.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_nifti_ras(vol_zyx, spacing_zyx, path):
    """Cached LPS array [z, y, x] -> NIfTI whose stored axes (i, j, k) point to the patient's right (i = nx-1-x),
    anterior (j = ny-1-y) and superior (k = z), with the matching RAS+ affine diag(sx, sy, sz)."""
    import nibabel as nib

    sz, sy, sx = spacing_zyx
    aff = np.diag([sx, sy, sz, 1.0])
    data = np.ascontiguousarray(np.asarray(vol_zyx).transpose(2, 1, 0)[::-1, ::-1, :])
    img = nib.Nifti1Image(data, aff)
    img.set_qform(aff, code=1)
    img.set_sform(aff, code=1)
    img.header.set_xyzt_units("mm")
    nib.save(img, path)
    return aff


def processed_to_cache(start, cache_shape):
    """Processed-grid index (d, h, w) -> cached [z, y, x]. Processed = DimTranspose of the stored (i, j, k), i.e.
    (k, j, i), cropped at `start`; then k = z, j = ny-1-y, i = nx-1-x."""
    start = np.asarray(start, dtype=np.float64)
    ny, nx = cache_shape[1], cache_shape[2]

    def f(idx):
        kji = np.asarray(idx, dtype=np.float64) + start
        return np.stack([kji[:, 0], (ny - 1) - kji[:, 1], (nx - 1) - kji[:, 2]], axis=1)

    return f


def add_sphere(vol, sp, zyx, diam=10.0, hu=300):
    vol = vol.copy()
    zz, yy, xx = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    d2 = ((zz - zyx[0]) * sp[0]) ** 2 + ((yy - zyx[1]) * sp[1]) ** 2 + ((xx - zyx[2]) * sp[2]) ** 2
    vol[d2 <= (diam / 2) ** 2] = hu
    return vol


def preprocess(processor, ct_path):
    """CT branch of SegVolProcessor.preprocess_ct_gt (verbatim) + zoom_transform with an empty label."""
    ct_voxel_ndarray, _ = processor.img_loader(ct_path)
    ct_voxel_ndarray = np.array(ct_voxel_ndarray).squeeze()
    ct_shape = ct_voxel_ndarray.shape
    ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray, axis=0)
    ct_voxel_ndarray = processor.ForegroundNorm(ct_voxel_ndarray)
    gt_npy = np.zeros((1,) + tuple(ct_shape), dtype=np.int32)
    return processor.zoom_transform(ct_voxel_ndarray, gt_npy)


class CaseDataset:
    """Writes the RAS NIfTI from the cached array, runs the official preprocessing (in DataLoader workers)."""

    def __init__(self, ds, cases, ndir, sphere=None, keep_nifti=False):
        self.ds, self.cases, self.ndir, self.sphere, self.keep = ds, cases, ndir, sphere, keep_nifti
        self.processor = None

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        t0 = time.time()
        if self.processor is None:
            self.processor = load_segvol_module().SegVolProcessor(spatial_size=[32, 256, 256])
        case = self.cases[i]
        vol, sp = load_cached(self.ds, case)
        if self.sphere is not None:  # coordinate check only
            vol = add_sphere(vol, sp, self.sphere)
        path = f"{self.ndir}/{case}.nii"
        write_nifti_ras(vol, sp, path + ".tmp.nii")
        os.replace(path + ".tmp.nii", path)
        item = preprocess(self.processor, path)
        if not self.keep:
            os.remove(path)
        return {"case": case, "image": item["image"], "zoom_out_image": item["zoom_out_image"],
                "start": np.asarray(item["foreground_start_coord"]), "end": np.asarray(item["foreground_end_coord"]),
                "cache_shape": vol.shape, "spacing": sp, "t_prep": time.time() - t0}


def build_model(device):
    import torch
    from transformers import AutoTokenizer

    mod = load_segvol_module()
    config = mod.SegVolConfig.from_pretrained(HF_DIR, test_mode=True)
    model = mod.SegVolModel(config)
    sd = torch.load(f"{HF_DIR}/pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.model.text_encoder.tokenizer = AutoTokenizer.from_pretrained(HF_DIR)
    model.eval()
    model.to(device)
    return mod, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["luna", "liver", "liver_ext"], required=True)
    ap.add_argument("--cases", required=True, help="whitespace-separated case ids (file)")
    ap.add_argument("--out", required=True, help="output jsonl (appended; finished cases are skipped)")
    ap.add_argument("--work", default=f"{HERE}/_work")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--sphere", default=None, help="z,y,x: insert a synthetic sphere (coordinate check only)")
    ap.add_argument("--save_npz", default=None, help="dir: keep prob per case (smoke tests / coordinate check only)")
    a = ap.parse_args()

    cases = open(a.cases).read().split()
    done = {json.loads(l)["case"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
    todo = [c for c in cases if c not in done]
    prompt = PROMPT[a.ds]
    print(f"{len(cases)} cases, {len(done)} done, {len(todo)} to run; prompt '{prompt}'", flush=True)
    if not todo:
        return
    sphere = tuple(float(v) for v in a.sphere.split(",")) if a.sphere else None
    ndir = f"{a.work}/nii_{'sphere' if sphere else a.ds}_{os.getpid()}"
    os.makedirs(ndir, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda", 0)
    t0 = time.time()
    mod, model = build_model(device)
    roi = {}
    _orig_roi = mod.logits2roi_coor

    def _roi_record(spatial_size, logits):  # records the zoom-in ROI (diagnostics only), returns it unchanged
        r = _orig_roi(spatial_size, logits)
        roi["box"] = None if r[0] is None else [int(v) for v in r]
        return r

    mod.logits2roi_coor = _roi_record
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    loader = DataLoader(CaseDataset(a.ds, todo, ndir, sphere, keep_nifti=False), batch_size=1, shuffle=False,
                        num_workers=a.num_workers, collate_fn=lambda b: b[0], prefetch_factor=2 if a.num_workers else None)
    if a.save_npz:
        os.makedirs(a.save_npz, exist_ok=True)
    fout = open(a.out, "a")
    t_all = time.time()
    t_wait = time.time()
    for item in loader:
        t_load = time.time() - t_wait
        case = item["case"]
        tm = time.time()
        roi.clear()
        image = item["image"].unsqueeze(0).to(device)
        zoomed = item["zoom_out_image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.forward_test(image=image, zoomed_image=zoomed, text_prompt=[prompt], use_zoom=True)
        prob = torch.sigmoid(logits[0, 0].float()).numpy()
        mask = prob > 0.5
        t_model = time.time() - tm
        sp = item["spacing"]
        assert tuple(sp) == tuple(load_spacing(a.ds, case))
        ans = mask_to_answer(mask, prob, float(sp[0] * sp[1] * sp[2]), processed_to_cache(item["start"], item["cache_shape"]),
                             item["cache_shape"])
        rec = {"case": case, **{k: ans[k] for k in ("present", "zyx", "diameter_mm", "score", "candidates")},
               "raw": json.dumps({"prompt": prompt, "n_components": ans["n_components"], "mask_vox": int(mask.sum()),
                                  "max_prob": round(float(prob.max()), 4), "clipped": ans.get("clipped", False),
                                  "grid": list(prob.shape), "crop_start": [int(v) for v in item["start"]],
                                  "zoom_in_roi": roi.get("box"), "t_model_s": round(t_model, 1),
                                  "t_prep_s": round(item["t_prep"], 1), "t_load_s": round(t_load, 1)})}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if a.save_npz:
            np.savez_compressed(f"{a.save_npz}/{case}.npz", prob=prob.astype(np.float16), start=item["start"],
                                image=item["image"][0].numpy().astype(np.float16))
        print(f"{case} present={ans['present']} ncomp={ans['n_components']} zyx={ans['zyx']} d={ans['diameter_mm']} "
              f"s={ans['score']} grid={list(prob.shape)} roi={roi.get('box')} model={t_model:.1f}s load={t_load:.1f}s",
              flush=True)
        del image, zoomed, logits
        t_wait = time.time()
    print(f"total {time.time() - t_all:.0f}s for {len(todo)} cases", flush=True)
    try:
        os.rmdir(ndir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
