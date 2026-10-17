"""Hulu-Med-14B on the benchmark: 3D volume fed as axial slices through the documented NIfTI path.

Documented path (README "3D Medical Image Example", HF remote code processing_hulumed.py load_nii):
  conversation content {"type": "3d", "3d": {"image_path": <nii>, "nii_num_slices": 180, "nii_axis": 2}}
  -> load_nii: nibabel get_fdata, slices volume[:, :, k], np.linspace(0, N-1, 180, dtype=int) if N > 180 else all,
     no windowing (default), per-slice min-max to uint8, 3 channels -> treated as a video (merge size 2) and resized
     by the processor to fit the 16384-token budget.
The NIfTI is written from the cached LPS volume (original voxel values) with an explicit affine such that each
volume[:, :, k] is the axial slice in radiological display orientation (rows anterior->posterior, columns patient
right->left) and k runs inferior->superior: volume[i, j, k] = vol_lps[k, i, j].
Coordinates: slice index into the fed slice list, x = column and y = row in the fed W x H slice image.
Inverse: z = fed_slice_indices[slice], y = y, x = x (cached LPS index).
Generation: add_system_prompt=True, add_generation_prompt=True, greedy (do_sample=False), max_new_tokens=1024,
bf16 + flash_attention_2 as in the README; other generation_config.json fields (repetition_penalty 1.05) unchanged.

  python run_hulumed.py check [--model]
  python run_hulumed.py smoke
  python run_hulumed.py eval --ds luna
  python run_hulumed.py check --ds liver_ext --model && python run_hulumed.py eval --ds liver_ext   # HCC-TACE-Seg
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, add_sphere, clip_zyx, eval_cases, label_free_center, load_lps,  # noqa: E402
                    parse_json_answer, prompt_key)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = ("/raid/home/CAMCA/hj880/hf_home/hub/models--ZJU-AI4H--Hulu-Med-14B/snapshots/"
         "b30d9161b8c23a79e20e1eca3891f63697531904")
NUM_SLICES = 180
TMP = os.path.join(HERE, "_tmp_nii")
PROMPT = {
    "luna": ("This is a chest CT scan given as {n} axial slices, ordered from inferior (slice 0) to superior "
             "(slice {nm1}), about {dz:.1f} mm apart. Each slice is a {w} x {h} pixel image with {px:.2f} mm pixels; "
             "x is the column (0 at the left edge, {wm1} at the right edge) and y is the row (0 at the top edge, {hm1} "
             "at the bottom edge). Find the largest lung nodule that is at least 6 mm in diameter. Answer with only a "
             "JSON object: {{\"present\": true or false, \"slice\": <slice index>, \"x\": <x>, \"y\": <y>, "
             "\"diameter_mm\": <diameter in mm>}}. If there is no lung nodule of at least 6 mm, set present to false "
             "and the other fields to null."),
    "liver": ("This is a contrast-enhanced abdominal CT scan given as {n} axial slices, ordered from inferior "
              "(slice 0) to superior (slice {nm1}), about {dz:.1f} mm apart. Each slice is a {w} x {h} pixel image "
              "with {px:.2f} mm pixels; x is the column (0 at the left edge, {wm1} at the right edge) and y is the row "
              "(0 at the top edge, {hm1} at the bottom edge). Find the largest liver tumor that is at least 10 mm in "
              "diameter. Answer with only a JSON object: {{\"present\": true or false, \"slice\": <slice index>, "
              "\"x\": <x>, \"y\": <y>, \"diameter_mm\": <diameter in mm>}}. If there is no liver tumor of at least "
              "10 mm, set present to false and the other fields to null."),
}
SMOKE = {"luna": ["1.3.6.1.4.1.14519.5.2.1.6279.6001.123697637451437522065941162930",
                  "1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217",
                  "1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275"],
         "liver": ["liver_8", "liver_48", "liver_1"]}


# ----------------------------------------------------------------------------------------------- I/O
def fed_indices(n: int) -> np.ndarray:
    """load_nii slice selection, verbatim."""
    if NUM_SLICES is not None and NUM_SLICES < n:
        return np.linspace(0, n - 1, NUM_SLICES, dtype=int)
    return np.arange(n)


def write_nifti(vol_lps: np.ndarray, sp, path: str) -> None:
    sz, sy, sx = sp
    arr = np.transpose(vol_lps, (1, 2, 0))                        # volume[i, j, k] = vol_lps[k, i, j]
    aff = np.array([[0.0, -sx, 0.0, 0.0],                         # j -> patient left  (RAS x decreases)
                    [-sy, 0.0, 0.0, 0.0],                         # i -> posterior     (RAS y decreases)
                    [0.0, 0.0, sz, 0.0],                          # k -> superior
                    [0.0, 0.0, 0.0, 1.0]])
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr), aff), path)


def make_prompt(ds: str, shape, sp, idx) -> str:
    n = len(idx)
    dz = (idx[-1] - idx[0]) / max(n - 1, 1) * sp[0]
    h, w = shape[1], shape[2]
    return PROMPT[prompt_key(ds)].format(n=n, nm1=n - 1, dz=dz, w=w, h=h, px=sp[2], wm1=w - 1, hm1=h - 1)


def answer_to_zyx(f: dict, idx, shape):
    """Fed (slice, x, y) -> cached LPS (z, y, x). Out-of-range coordinates -> None."""
    s, x, y = f.get("slice"), f.get("x"), f.get("y")
    if s is None or x is None or y is None:
        return None, "no_coordinates"
    n, h, w = len(idx), shape[1], shape[2]
    if not (-0.5 <= s <= n - 0.5 and -0.5 <= x <= w - 0.5 and -0.5 <= y <= h - 0.5):
        return None, "out_of_range"
    z = float(np.interp(min(max(s, 0), n - 1), np.arange(n), idx))
    return clip_zyx([z, y, x], shape), "ok"


# ----------------------------------------------------------------------------------------------- model
class Hulu:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype="bfloat16",
                                                          device_map="auto",
                                                          attn_implementation="flash_attention_2").eval()
        self.proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    def inputs(self, nii: str, prompt: str):
        conv = [{"role": "user", "content": [
            {"type": "3d", "3d": {"image_path": nii, "nii_num_slices": NUM_SLICES, "nii_axis": 2}},
            {"type": "text", "text": prompt}]}]
        return self.proc(conversation=conv, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt")

    def ask(self, nii: str, prompt: str):
        inp = self.inputs(nii, prompt)
        n_frames = int(inp["grid_sizes"][0][0])
        inp = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=1024, do_sample=False)
        full = self.proc.batch_decode(out, skip_special_tokens=True, use_think=True)[0].strip()
        ans = self.proc.batch_decode(out, skip_special_tokens=True, use_think=False)[0].strip()
        return ans, full, n_frames, int(inp["input_ids"].shape[1])


def run_case(hulu: Hulu, ds: str, case: str, vol=None):
    t0 = time.time()
    v0, sp = load_lps(ds, case)
    vol = v0 if vol is None else vol
    idx = fed_indices(vol.shape[0])
    os.makedirs(TMP, exist_ok=True)
    nii = os.path.join(TMP, f"{case}.nii")
    write_nifti(vol, sp, nii)
    prompt = make_prompt(ds, vol.shape, sp, idx)
    try:
        ans, full, n_frames, n_tok = hulu.ask(nii, prompt)
    finally:
        os.remove(nii)
    assert n_frames == len(idx), (n_frames, len(idx))
    f, status = parse_json_answer(ans)
    row = {"case": case, "present": False, "zyx": None, "diameter_mm": None, "score": None, "candidates": [],
           "raw": full, "parse_status": status, "parsed": f, "prompt": prompt,
           "fed": {"slice_indices": idx.tolist(), "image_hw": [int(vol.shape[1]), int(vol.shape[2])],
                   "orientation": "rows=y(A->P), cols=x(R->L), slices z inferior->superior", "n_tokens": n_tok}}
    if f is not None:
        zyx, st = answer_to_zyx(f, idx, vol.shape)
        present = f.get("present")
        if present is None:
            present = zyx is not None
        row["present"] = bool(present)
        if present and zyx is not None:
            row["zyx"] = zyx
            row["diameter_mm"] = f.get("diameter_mm")
        elif present:
            row["parse_status"] = st
    row["sec"] = round(time.time() - t0, 2)
    return row


# ----------------------------------------------------------------------------------------------- modes
CHECK = [("luna", "1.3.6.1.4.1.14519.5.2.1.6279.6001.111258527162678142285870245028"),
         ("luna", "1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275"),   # N < 180: all slices
         ("liver", "liver_3")]


def frames_from_pixels(inp) -> np.ndarray:
    """Invert the image processor's patch flattening: pixel_values -> (t, C, H, W) frames."""
    pv = inp["pixel_values"].float().numpy()
    t, gh, gw = (int(v) for v in inp["grid_sizes"][0])
    m, p = int(inp["merge_sizes"][0]), 14
    x = pv.reshape(t, gh // m, gw // m, m, m, 3, p, p).transpose(0, 5, 1, 3, 6, 2, 4, 7)
    return x.reshape(t, 3, gh * p, gw * p)


def check(args):
    from scipy import ndimage as ndi
    from PIL import Image
    from transformers import AutoProcessor
    hulu = Hulu() if args.model else None
    proc = hulu.proc if hulu else AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    os.makedirs(TMP, exist_ok=True)
    report = []
    checks = [("liver_ext", args.case or "HCC_002")] if args.ds == "liver_ext" else CHECK
    for ds, case in checks:
        vol, sp = load_lps(ds, case)
        idx = fed_indices(vol.shape[0])
        if ds == "liver_ext":                                     # evaluation set: image-only location, no labels
            c = label_free_center(vol)
        else:
            dom = np.load(f"{ROOT}/cache/{ds}/{case}.{'lung' if ds == 'luna' else 'liver'}.npy")
            dist = ndi.distance_transform_edt(dom, sampling=sp)
            c = np.array(np.unravel_index(int(np.argmax(dist)), dom.shape), float)
        k = int(np.argmin(np.abs(idx - c[0])))                    # put the centre on a fed slice
        center = np.array([float(idx[k]), round(c[1]), round(c[2])])
        syn, _ = add_sphere(vol, sp, center)
        prompt = make_prompt(ds, vol.shape, sp, idx)
        fr = {}
        for tag, v in (("orig", vol), ("syn", syn)):
            nii = os.path.join(TMP, f"check_{tag}.nii")
            write_nifti(v, sp, nii)
            conv = [{"role": "user", "content": [
                {"type": "3d", "3d": {"image_path": nii, "nii_num_slices": NUM_SLICES, "nii_axis": 2}},
                {"type": "text", "text": prompt}]}]
            inp = proc(conversation=conv, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt")
            fr[tag] = frames_from_pixels(inp)
            os.remove(nii)
        diff = np.clip(fr["syn"] - fr["orig"], 0, None).mean(1)  # (t, Hr, Wr)
        t_hat = int(np.argmax(diff.reshape(len(diff), -1).sum(1)))
        d2 = diff[t_hat]
        cy, cx = ndi.center_of_mass(d2 * (d2 > 0.5 * d2.max()))
        hr, wr = d2.shape
        y_fed = (cy + 0.5) * vol.shape[1] / hr - 0.5
        x_fed = (cx + 0.5) * vol.shape[2] / wr - 0.5
        back, _ = answer_to_zyx({"slice": t_hat, "x": x_fed, "y": y_fed}, idx, vol.shape)
        rep = {"ds": ds, "case": case, "center_zyx": center.tolist(), "n_fed_slices": len(idx),
               "frame_hw_seen_by_model": [hr, wr], "sphere_frame": t_hat, "expected_frame": k,
               "sphere_xy_fed": [round(x_fed, 2), round(y_fed, 2)], "back_zyx": np.round(back, 2).tolist(),
               "err_vox": (np.array(back) - center).round(2).tolist()}
        # a picture of the fed frame (orientation check)
        g = fr["syn"][t_hat].transpose(1, 2, 0)
        Image.fromarray(np.clip((g * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)).save(
            os.path.join(HERE, f"check_frame_{ds}.png"))
        if hulu is not None:
            row = run_case(hulu, ds, case, vol=syn)
            rep["model_raw"] = row["raw"]
            if row["zyx"] is not None:
                rep["model_zyx"] = np.round(row["zyx"], 1).tolist()
                rep["model_dist_mm"] = round(float(np.linalg.norm((np.array(row["zyx"]) - center) * np.array(sp))), 1)
        print(json.dumps(rep), flush=True)
        report.append(rep)
    out = "coord_check_liver_ext.json" if args.ds == "liver_ext" else "coord_check.json"
    json.dump(report, open(os.path.join(HERE, out), "w"), indent=1)


def smoke(args):
    hulu = Hulu()
    with open(os.path.join(HERE, "smoke.jsonl"), "w") as f:
        for ds in ("luna", "liver"):
            for case in SMOKE[ds]:
                row = run_case(hulu, ds, case)
                row["ds"] = ds
                print(json.dumps({k: row[k] for k in ("ds", "case", "present", "zyx", "diameter_mm", "raw",
                                                       "parse_status", "sec")}), flush=True)
                f.write(json.dumps(row) + "\n")


def evaluate(args):
    hulu = Hulu()
    out = f"{ROOT}/results/sota/hulumed14b_{args.ds}.jsonl"
    done = {json.loads(l)["case"] for l in open(out)} if os.path.exists(out) else set()
    t0 = time.time()
    cases = eval_cases(args.ds)
    for i, case in enumerate(cases):
        if case in done:
            continue
        row = run_case(hulu, args.ds, case)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(cases)}] {case} {row['parse_status']} {row['raw'][:160]!r} {row['sec']}s", flush=True)
    print(f"done {args.ds} in {time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "smoke", "eval"])
    ap.add_argument("--ds", choices=["luna", "liver", "liver_ext"])
    ap.add_argument("--case", default=None, help="check --ds liver_ext: the case to use")
    ap.add_argument("--model", action="store_true")
    a = ap.parse_args()
    {"check": check, "smoke": smoke, "eval": evaluate}[a.mode](a)
