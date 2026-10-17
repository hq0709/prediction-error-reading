"""M3D-LaMed-Phi-3-4B on the benchmark: native 3D box output (positioning / REC task).

Input path = the repo's own positioning-data path (Data/process/m3d_seg_data_preprocess.py + m3d_seg_data_prepare.py,
which produced the M3D-Seg volumes the box task was trained on):
  original voxels in the file's stored order [k, j, i] (no reorientation, as in the M3D-Seg preprocessing)
  -> SegVol normalize (clip to the 0.05/99.95 percentiles of voxels above the volume mean, z-score)
  -> min-max to [0, 1] -> CropForeground (voxels > 0) -> Resize to 32 x 256 x 256 (MONAI 'bilinear' = trilinear).
Prompt: '<im_patch>' * 256 + ' ' + question, no chat template (Bench/dataset/multi_dataset.py PosRECDataset);
greedy decoding, max_new_tokens 256, fp32 weights as in Bench/eval/eval_pos.py.
Output: '<bx_start>[d1, h1, w1, d2, h2, w2]<bx_end>' with each coordinate = voxel index / axis size in the 32x256x256
volume (LaMed/src/utils/utils.py mask2box on nearest-resized masks). Inverse used here (exact inverse of that label
path): centre_src = (o_min + o_max) / 2 * s - 0.5 and extent = (o_max - o_min + 1) * s voxels, s = cropped / resized size.

  python run_m3d.py check            # synthetic-sphere coordinate check (pool scans)
  python run_m3d.py smoke            # 3 LUNA16 + 3 liver training-pool scans
  python run_m3d.py eval --ds luna   # writes results/sota/m3d_lamed_luna.jsonl
  python run_m3d.py check --ds liver_ext --model && python run_m3d.py eval --ds liver_ext   # HCC-TACE-Seg
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, add_sphere, clip_zyx, eval_cases, label_free_center, load_lps, lps_to_raw,  # noqa: E402
                    prompt_key, raw_axes, raw_index_to_lps)

import monai.transforms as mtf  # noqa: E402
from monai.data import set_track_meta  # noqa: E402

set_track_meta(False)

MODEL = ("/raid/home/CAMCA/hj880/hf_cache/models--GoodBaiBai88--M3D-LaMed-Phi-3-4B/snapshots/"
         "329bed510c2cff0ad3fe6f2a1a7075f45e2706e9")
OUT_SIZE = np.array([32, 256, 256])
QUESTION = {"luna": "Where is the largest lung nodule of at least 6 mm in this image? Please output the box.",
            "liver": "Where is the largest liver tumor of at least 10 mm in this image? Please output the box."}
BOX_RE = re.compile(r"\[([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), "
                    r"([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+)\]")  # Bench/utils.py
NO_RE = re.compile(r"\b(no|not|sorry|unfortunately|couldn't|can not|cannot|isn't)\b", re.I)
SMOKE = {"luna": ["1.3.6.1.4.1.14519.5.2.1.6279.6001.123697637451437522065941162930",
                  "1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217",
                  "1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275"],
         "liver": ["liver_8", "liver_48", "liver_1"]}


# ----------------------------------------------------------------------------------------------- preprocessing
def segvol_normalize(ct: np.ndarray) -> np.ndarray:
    """Data/process/m3d_seg_data_preprocess.py normalize(), verbatim logic."""
    flat = ct.flatten()
    thred = np.mean(flat)
    vf = flat[flat > thred]
    upper, lower = np.percentile(vf, 99.95), np.percentile(vf, 00.05)
    mean, std = np.mean(vf), np.std(vf)
    ct = np.clip(ct, lower, upper)
    return (ct - mean) / max(std, 1e-8)


def preprocess(raw: np.ndarray):
    """raw: [D, H, W] original voxels in stored order. Returns fed (1, 32, 256, 256) float32, crop start, crop end."""
    x = segvol_normalize(raw[None].astype(np.float32))
    x = x - x.min()                                              # m3d_seg_data_prepare.py normalization
    x = x / np.clip(x.max(), a_min=1e-8, a_max=None)
    cropped, start, end = mtf.CropForeground(return_coords=True)(x)   # select_fn x > 0, margin 0
    fed = mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")(cropped)
    return np.asarray(fed, dtype=np.float32), np.asarray(start), np.asarray(end)


def label_box(mask_raw: np.ndarray, start, end) -> list[float]:
    """The repo's own label path for a mask: same crop, nearest resize, mask2box (utils.py)."""
    m = mask_raw[None, start[0]:end[0], start[1]:end[1], start[2]:end[2]].astype(np.float32)
    m = np.asarray(mtf.Resize(spatial_size=[32, 256, 256], mode="nearest")(m))[0] > 0.5
    nz = np.nonzero(m)
    if len(nz[0]) == 0:
        return None                                              # the object fell between nearest-resize samples
    lo = [int(i.min()) for i in nz]
    hi = [int(i.max()) for i in nz]
    return [round(lo[a] / OUT_SIZE[a], 3) for a in range(3)] + [round(hi[a] / OUT_SIZE[a], 3) for a in range(3)]


# ----------------------------------------------------------------------------------------------- mapping
def box_to_answer(box, start, end, raw_shape, axes, sp_lps, lps_shape):
    b = np.clip(np.asarray(box, float), 0.0, 1.0)
    lo, hi = np.minimum(b[:3], b[3:]), np.maximum(b[:3], b[3:])
    o_lo, o_hi = lo * OUT_SIZE, hi * OUT_SIZE                    # voxel indices in the 32x256x256 volume
    s = (end - start) / OUT_SIZE                                 # cropped voxels per resized voxel
    c_raw = (o_lo + o_hi) / 2.0 * s - 0.5 + start                # inverse of nearest resize + mask2box
    zyx = clip_zyx(raw_index_to_lps(c_raw, axes, raw_shape), lps_shape)
    sp_raw = np.array([sp_lps[m] for m, _ in axes])              # spacing along the stored axes
    ext_mm = (o_hi - o_lo + 1.0) * s * sp_raw
    return zyx, float(np.mean(ext_mm)), ext_mm.tolist()


def fed_index_to_lps(o, start, end, raw_shape, axes, lps_shape):
    """Continuous index in the fed (trilinear-resized) volume -> cached LPS index (align_corners=False)."""
    s = (end - start) / OUT_SIZE
    c_raw = (np.asarray(o, float) + 0.5) * s - 0.5 + start
    return raw_index_to_lps(c_raw, axes, raw_shape)


def parse(text: str):
    m = BOX_RE.search(text)
    if m:
        return [float(g) for g in m.groups()], "ok"
    return None, ("absent" if NO_RE.search(text) else "unparseable")


# ----------------------------------------------------------------------------------------------- model
class M3D:
    def __init__(self, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(MODEL, model_max_length=512, padding_side="right", use_fast=False,
                                                 trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True).to(self.device).eval()
        self.dtype = next(self.model.parameters()).dtype

    def ask(self, fed: np.ndarray, question: str) -> str:
        prompt = "<im_patch>" * 256 + " " + question
        ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.device)
        assert ids[0, 0].item() == self.tok.bos_token_id and (ids[0, 1:257] == 32011).all(), ids[0, :3]
        img = torch.from_numpy(fed).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            gen = self.model.generate(img, ids, max_new_tokens=256, do_sample=False, top_p=None, temperature=1.0)
        return self.tok.batch_decode(gen, skip_special_tokens=True)[0]


def run_case(m3d: M3D, ds: str, case: str, vol=None):
    t0 = time.time()
    if vol is None:
        vol, sp = load_lps(ds, case)
    else:
        sp = load_lps(ds, case)[1]
    axes = raw_axes(ds, case)
    raw = lps_to_raw(vol, axes)
    fed, start, end = preprocess(raw)
    text = m3d.ask(fed, QUESTION[prompt_key(ds)])
    box, status = parse(text)
    row = {"case": case, "present": False, "zyx": None, "diameter_mm": None, "score": None, "candidates": [],
           "raw": text, "parse_status": status,
           "fed": {"raw_axes": axes, "raw_shape": list(raw.shape), "crop_start": start.tolist(),
                   "crop_end": end.tolist(), "resized": OUT_SIZE.tolist()}}
    if box is not None:
        zyx, d, ext = box_to_answer(box, start, end, raw.shape, axes, sp, vol.shape)
        row.update(present=True, zyx=zyx, diameter_mm=d, box=box, box_extent_mm=ext)
    row["sec"] = round(time.time() - t0, 2)
    return row


# ----------------------------------------------------------------------------------------------- modes
CHECK = [("luna", "1.3.6.1.4.1.14519.5.2.1.6279.6001.111258527162678142285870245028"),   # identity direction
         ("luna", "1.3.6.1.4.1.14519.5.2.1.6279.6001.282512043257574309474415322775"),   # x/y-flipped direction
         ("liver", "liver_3")]


def pick_center(ds, case, vol, sp, start, end, axes, raw_shape):
    """A point deep inside the lung (LUNA lung mask) or liver (pool liver mask), moved onto a resized-voxel centre."""
    from scipy import ndimage as ndi
    if ds == "liver_ext":                                         # evaluation set: image-only location, no labels
        c = label_free_center(vol)
    else:
        dom = np.load(f"{ROOT}/cache/{ds}/{case}.{'lung' if ds == 'luna' else 'liver'}.npy")
        dist = ndi.distance_transform_edt(dom, sampling=sp)
        c = np.array(np.unravel_index(int(np.argmax(dist)), dom.shape), float)
    # snap to the centre of the nearest fed (trilinear) voxel so the fed image samples the sphere centre exactly
    raw_c = np.array([c[axes[n][0]] if axes[n][1] > 0 else raw_shape[n] - 1 - c[axes[n][0]] for n in range(3)])
    s = (end - start) / OUT_SIZE
    o = np.round((raw_c - start + 0.5) / s - 0.5)
    raw_c = (o + 0.5) * s - 0.5 + start
    return np.array(raw_index_to_lps(raw_c, axes, raw_shape))


def check(args):
    m3d = M3D() if args.model else None
    report = []
    checks = [("liver_ext", args.case or "HCC_002")] if args.ds == "liver_ext" else CHECK
    for ds, case in checks:
        vol, sp = load_lps(ds, case)
        axes = raw_axes(ds, case)
        raw0 = lps_to_raw(vol, axes)
        # the stored-order array must equal the original file read without reorientation
        import SimpleITK as sitk
        from common import original_path
        assert np.array_equal(raw0, sitk.GetArrayFromImage(sitk.ReadImage(original_path(ds, case)))), case
        _, start0, end0 = preprocess(raw0)
        center = pick_center(ds, case, vol, sp, start0, end0, axes, raw0.shape)
        syn, mask = add_sphere(vol, sp, center)
        raw, mraw = lps_to_raw(syn, axes), lps_to_raw(mask, axes)
        fed, start, end = preprocess(raw)
        fed0, _, _ = preprocess(raw0)
        assert np.array_equal(start, start0) and np.array_equal(end, end0)
        # (a) where does the sphere sit in the tensor the model sees?  intensity-weighted centroid of the change
        from scipy import ndimage as ndi
        diff = np.clip(fed[0] - fed0[0], 0, None)
        cen = np.array(ndi.center_of_mass(diff * (diff > 0.5 * diff.max())))
        back_a = np.array(fed_index_to_lps(cen, start, end, raw.shape, axes, vol.shape))
        # (b) the model's own label path: the sphere mask -> nearest resize -> mask2box -> our inverse
        s = (end - start) / OUT_SIZE
        rep = {"ds": ds, "case": case, "center_zyx": center.round(3).tolist(), "raw_axes": axes,
               "crop": [start.tolist(), end.tolist()], "cropped_per_resized_voxel": s.round(3).tolist(),
               "fed_centroid": cen.round(3).tolist(), "fed_centroid_back_zyx": back_a.round(3).tolist(),
               "err_fed_centroid_vox": (back_a - center).round(3).tolist()}
        box = label_box(mraw, start, end)
        rep["label_box"] = box
        if box is not None:
            pbox, _ = parse("<bx_start>" + str(box) + "<bx_end>")
            back_b, dmm, ext = box_to_answer(pbox, start, end, raw.shape, axes, sp, vol.shape)
            rep.update(label_box_back_zyx=np.round(back_b, 3).tolist(),
                       err_label_box_vox=(np.array(back_b) - center).round(3).tolist(),
                       label_box_diameter_mm=round(dmm, 2), label_box_extent_mm=np.round(ext, 2).tolist())
        if m3d is not None:
            t = m3d.ask(fed, QUESTION[prompt_key(ds)])
            b, st = parse(t)
            rep["model_raw"] = t
            if b is not None:
                z, d, _ = box_to_answer(b, start, end, raw.shape, axes, sp, vol.shape)
                rep["model_zyx"] = np.round(z, 2).tolist()
                rep["model_dist_mm"] = round(float(np.linalg.norm((np.array(z) - center) * np.array(sp))), 1)
                rep["model_diameter_mm"] = round(d, 1)
        print(json.dumps(rep), flush=True)
        report.append(rep)
    out = "coord_check_liver_ext.json" if args.ds == "liver_ext" else "coord_check.json"
    json.dump(report, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), out), "w"),
              indent=1)


def smoke(args):
    m3d = M3D()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke.jsonl")
    with open(out, "w") as f:
        for ds in ("luna", "liver"):
            for case in SMOKE[ds]:
                row = run_case(m3d, ds, case)
                row["ds"] = ds
                print(json.dumps({k: row[k] for k in ("ds", "case", "present", "zyx", "diameter_mm", "raw",
                                                       "parse_status", "sec")}), flush=True)
                f.write(json.dumps(row) + "\n")


def evaluate(args):
    m3d = M3D()
    out = f"{ROOT}/results/sota/m3d_lamed_{args.ds}.jsonl"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["case"] for l in open(out)}
    t0 = time.time()
    cases = eval_cases(args.ds)
    for i, case in enumerate(cases):
        if case in done:
            continue
        row = run_case(m3d, args.ds, case)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(cases)}] {case} {row['parse_status']} {row['raw'][:120]!r} {row['sec']}s", flush=True)
    print(f"done {args.ds} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "smoke", "eval"])
    ap.add_argument("--ds", choices=["luna", "liver", "liver_ext"])
    ap.add_argument("--case", default=None, help="check --ds liver_ext: the case to use")
    ap.add_argument("--model", action="store_true", help="check: also run the model on the synthetic volumes")
    a = ap.parse_args()
    {"check": check, "smoke": smoke, "eval": evaluate}[a.mode](a)
