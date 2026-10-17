"""MedGemma 1.5 4B on the benchmark: CT as a multi-image prompt, following the official CT notebook
(google-health/medgemma notebooks/high_dimensional_ct_hugging_face.ipynb):
  * slices ordered as the DICOM series is read, here superior -> inferior; if more than 85, sample 85 with the
    notebook's rule  [slices[int(round(i / 85 * (N - 1)))] for i in range(1, 86)];
  * each slice windowed into RGB = (-1024, 1024), (-135, 215), (0, 80) HU, scaled to 0-255, rounded to uint8;
  * each slice JPEG-encoded into a data URL; content = instruction, then for every slice an image followed by the
    text "SLICE n", then the query; AutoProcessor(use_fast=True).apply_chat_template, bf16, greedy decoding
    (do_sample=False), max_new_tokens=2000.
In-plane orientation: the cached LPS slice vol[z] is exactly the DICOM pixel layout (rows anterior->posterior,
columns patient right->left). Location format: MedGemma's own localization convention (CXR localization notebook):
x and y normalized to [0, 1000] across the image width / height. Inverse: z = fed_z[SLICE - 1],
x = x / 1000 * W - 0.5, y = y / 1000 * H - 0.5 (cached LPS index).

  python run_medgemma.py check [--model]
  python run_medgemma.py smoke
  python run_medgemma.py eval --ds luna [--part 0/2]    # then: python run_medgemma.py merge --ds luna
  python run_medgemma.py check --ds liver_ext --model; eval --ds liver_ext; merge --ds liver_ext   # HCC-TACE-Seg
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, add_sphere, clip_zyx, eval_cases, label_free_center, load_lps,  # noqa: E402
                    parse_json_answer, prompt_key)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = ("/raid/home/CAMCA/hj880/hf_home/hub/models--unsloth--medgemma-1.5-4b-it/snapshots/"
         "de0155051f541fb5016589feeec59e9cd078347d")
MAX_SLICE = 85
WINDOWS = [(-1024, 1024), (-135, 215), (0, 80)]
INSTRUCTION = {
    "luna": ("You are a radiologist reviewing a chest CT scan. You are given {n} axial CT slices ordered from superior "
             "(SLICE 1) to inferior (SLICE {n}), about {dz:.1f} mm apart; each slice image covers {fov:.0f} mm x "
             "{fov:.0f} mm ({px:.2f} mm per pixel). Please review the slices provided below carefully."),
    "liver": ("You are a radiologist reviewing a contrast-enhanced abdominal CT scan. You are given {n} axial CT slices "
              "ordered from superior (SLICE 1) to inferior (SLICE {n}), about {dz:.1f} mm apart; each slice image "
              "covers {fov:.0f} mm x {fov:.0f} mm ({px:.2f} mm per pixel). Please review the slices provided below "
              "carefully."),
}
# Query: MedGemma's own localization prompt structure (notebooks/cxr_anatomy_localization_with_hugging_face.ipynb:
# "Instructions: ... You MUST output a single parseable json ... Now answer to the user query. / Query: ... Don't give a
# final answer without reasoning. Output the final answer in the format "Final Answer: X" ... Answer:"), adapted from
# boxes to a point (SLICE, x, y) plus a diameter. Placed after the slices, where the CT notebook puts its query.
_LOC = ("\n\nInstructions:\nThe following user query will require outputting the location of a finding in the CT "
        "slices above. The location is the SLICE number on which the finding appears largest and the x and y "
        "coordinates of its centre on that slice. Always normalize the x and y coordinates to the range [0, 1000], "
        "meaning that a point at 15% of the image width would be associated with an x coordinate of 150, with x "
        "measured from the left edge and y from the top edge of the image. You MUST output a single parseable json "
        "object enclosed in ```json...``` brackets, with the keys \"present\" (true or false), \"slice\" (SLICE "
        "number), \"x\", \"y\" and \"diameter_mm\" (diameter in millimetres). Now answer to the user query.\n\n"
        "Query:\nWhere is the largest {target} that is at least {floor} mm in diameter? Don't give a final answer "
        "without reasoning. Output the final answer in the format \"Final Answer: X\" where X is the JSON object. If "
        "there is no {target} of at least {floor} mm, X is {{\"present\": false, \"slice\": null, \"x\": null, "
        "\"y\": null, \"diameter_mm\": null}}. Answer:")
QUERY = {"luna": _LOC.format(target="lung nodule", floor=6), "liver": _LOC.format(target="liver tumor", floor=10)}
_REASON = ("Don't give a final answer without reasoning. Output the final answer in the format \"Final Answer: X\" "
           "where X is the JSON object.")
_DIRECT = ("Do not write a report. Output only the final answer in the format \"Final Answer: X\" where X is the "
           "JSON object.")
INSTRUCTION_TEACH = {
    "luna": ("You are an instructor teaching medical students. You are analyzing {n} axial slices of a chest CT scan, "
             "ordered from superior (SLICE 1) to inferior (SLICE {n}), about {dz:.1f} mm apart; each slice image covers "
             "{fov:.0f} mm x {fov:.0f} mm ({px:.2f} mm per pixel). Please review the slices provided below carefully."),
    "liver": ("You are an instructor teaching medical students. You are analyzing {n} axial slices of a "
              "contrast-enhanced abdominal CT scan, ordered from superior (SLICE 1) to inferior (SLICE {n}), about "
              "{dz:.1f} mm apart; each slice image covers {fov:.0f} mm x {fov:.0f} mm ({px:.2f} mm per pixel). Please "
              "review the slices provided below carefully."),
}
_SPLIT = "\n\nQuery:\n"
INSTRUCTION_D = {k: INSTRUCTION[k] + QUERY[k].split(_SPLIT)[0].replace("in the CT slices above", "in the CT slices below")
                 for k in INSTRUCTION}
QUERY_D = {k: "\n\nQuery:\n" + QUERY[k].split(_SPLIT)[1] for k in QUERY}
INSTRUCTION_F = {
    "luna": ("You are a radiologist. Your task is to locate the largest lung nodule that is at least 6 mm in diameter "
             "in a chest CT scan. You are given {n} axial CT slices ordered from superior (SLICE 1) to inferior (SLICE "
             "{n}), about {dz:.1f} mm apart; each slice image covers {fov:.0f} mm x {fov:.0f} mm ({px:.2f} mm per "
             "pixel). Please review the slices provided below carefully."),
    "liver": ("You are a radiologist. Your task is to locate the largest liver tumor that is at least 10 mm in "
              "diameter in a contrast-enhanced abdominal CT scan. You are given {n} axial CT slices ordered from "
              "superior (SLICE 1) to inferior (SLICE {n}), about {dz:.1f} mm apart; each slice image covers "
              "{fov:.0f} mm x {fov:.0f} mm ({px:.2f} mm per pixel). Please review the slices provided below "
              "carefully."),
}
# Variant G: as D, with the JSON requirement restated in the query after the slices.
_Q_G = ("\n\nQuery:\nWhere is the largest {target} that is at least {floor} mm in diameter? Don't give a final answer "
        "without reasoning. Output the final answer in the format \"Final Answer: X\" where X is a single parseable "
        "json object enclosed in ```json...``` brackets, with the keys \"present\", \"slice\", \"x\", \"y\" and "
        "\"diameter_mm\". If there is no {target} of at least {floor} mm, X is ```json{{\"present\": false, "
        "\"slice\": null, \"x\": null, \"y\": null, \"diameter_mm\": null}}```. Answer:")
QUERY_G = {"luna": _Q_G.format(target="lung nodule", floor=6), "liver": _Q_G.format(target="liver tumor", floor=10)}
VARIANTS = {  # prompt development on training-pool scans only; the chosen one is DEFAULT_VARIANT
    "A": (INSTRUCTION, QUERY),
    "D": (INSTRUCTION_D, QUERY_D),
    "F": (INSTRUCTION_F, QUERY),
    "G": (INSTRUCTION_D, QUERY_G),
    "B": (INSTRUCTION, {k: v.replace(_REASON, _DIRECT) for k, v in QUERY.items()}),
    "C": (INSTRUCTION_TEACH, {k: v.replace(_REASON, _DIRECT) for k, v in QUERY.items()}),
}
DEFAULT_VARIANT = "G"  # chosen on pool scans by machine-readable rate (see NOTES.md)
# First draft (CT-notebook style query), replaced after the pool-scan smoke test: the model answered with a free-text
# report or looped without a 'Final Answer' on pool volumes.
QUERY_DRAFT0 = {
    "luna": ("\n\nBased on the visual evidence in the slices provided above, find the largest lung nodule that is at "
             "least 6 mm in diameter. Report the SLICE number where it appears largest, the position of its centre in "
             "that slice as x and y normalized to the range [0, 1000] (x = 0 at the left edge and 1000 at the right "
             "edge of the image, y = 0 at the top edge and 1000 at the bottom edge), and its diameter in millimetres. "
             "Please provide your reasoning and conclude with 'Final Answer: ' followed by a JSON object "
             "{\"present\": true or false, \"slice\": <SLICE number>, \"x\": <x>, \"y\": <y>, \"diameter_mm\": "
             "<diameter in mm>}. If there is no lung nodule of at least 6 mm, set present to false and the other "
             "fields to null."),
}
SMOKE = {"luna": ["1.3.6.1.4.1.14519.5.2.1.6279.6001.123697637451437522065941162930",
                  "1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217",
                  "1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275"],
         "liver": ["liver_8", "liver_48", "liver_1"]}
SMOKE_EXTRA = {"luna": ["1.3.6.1.4.1.14519.5.2.1.6279.6001.111258527162678142285870245028",
                        "1.3.6.1.4.1.14519.5.2.1.6279.6001.282512043257574309474415322775"],
               "liver": ["liver_3", "liver_26"]}


# ----------------------------------------------------------------------------------------------- I/O
def fed_z(n: int) -> list[int]:
    """Superior -> inferior order (cached z grows toward superior), then the notebook's 85-slice rule."""
    order = list(range(n - 1, -1, -1))
    if len(order) > MAX_SLICE:
        order = [order[int(round(i / MAX_SLICE * (len(order) - 1)))] for i in range(1, MAX_SLICE + 1)]
    return order


def norm(ct: np.ndarray, lo: float, hi: float) -> np.ndarray:
    ct = np.clip(ct, lo, hi).astype(np.float32)
    ct -= lo
    ct /= (hi - lo)
    ct *= 255.0
    return ct


def window(ct_slice: np.ndarray) -> np.ndarray:
    return np.round(np.stack([norm(ct_slice, a, b) for a, b in WINDOWS], axis=-1), 0).astype(np.uint8)


def encode(data: np.ndarray) -> str:
    with io.BytesIO() as buf:
        with Image.fromarray(data) as img:
            img.save(buf, format="jpeg")
        buf.seek(0)
        s = base64.b64encode(buf.getbuffer()).decode("utf-8")
    return f"data:image/jpeg;base64,{s}"


def build_messages(ds: str, vol: np.ndarray, sp, zs, variant: str | None = None):
    instruction, query = VARIANTS[variant or DEFAULT_VARIANT]
    ds = prompt_key(ds)                                           # liver_ext uses the MSD liver prompt
    n = len(zs)
    dz = abs(zs[0] - zs[-1]) / max(n - 1, 1) * sp[0]
    instr = instruction[ds].format(n=n, dz=dz, fov=vol.shape[2] * sp[2], px=sp[2])
    content = [{"type": "text", "text": instr}]
    for k, z in enumerate(zs, 1):
        content.append({"type": "image", "image": encode(window(vol[z]))})
        content.append({"type": "text", "text": f"SLICE {k}"})
    content.append({"type": "text", "text": query[ds]})
    return [{"role": "user", "content": content}], instr + " [SLICE images] " + query[ds]


def answer_to_zyx(f: dict, zs, shape):
    s, x, y = f.get("slice"), f.get("x"), f.get("y")
    if s is None or x is None or y is None:
        return None, "no_coordinates"
    n = len(zs)
    if not (0.5 <= s <= n + 0.5 and 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        return None, "out_of_range"
    z = float(np.interp(min(max(s, 1), n) - 1, np.arange(n), zs))
    return clip_zyx([z, y / 1000.0 * shape[1] - 0.5, x / 1000.0 * shape[2] - 0.5], shape), "ok"


# ----------------------------------------------------------------------------------------------- model
class MedGemma:
    def __init__(self):
        import transformers
        self.proc = transformers.AutoProcessor.from_pretrained(MODEL, use_fast=True)
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="auto").eval()

    def inputs(self, messages):
        return self.proc.apply_chat_template(messages, add_generation_prompt=True, continue_final_message=False,
                                             return_tensors="pt", tokenize=True, return_dict=True)

    def ask(self, messages):
        inp = self.inputs(messages).to(self.model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            gen = self.model.generate(**inp, do_sample=False, max_new_tokens=2000)
        new = gen[0, inp["input_ids"].shape[1]:]
        raw = self.proc.decode(new, skip_special_tokens=False)
        clean = self.proc.decode(new, skip_special_tokens=True)
        return raw, clean, int(inp["input_ids"].shape[1]), int(len(new))


def run_case(mg: MedGemma, ds: str, case: str, vol=None, variant: str | None = None):
    t0 = time.time()
    v0, sp = load_lps(ds, case)
    vol = v0 if vol is None else vol
    zs = fed_z(vol.shape[0])
    messages, prompt = build_messages(ds, vol, sp, zs, variant)
    raw, clean, n_in, n_new = mg.ask(messages)
    f, status = parse_json_answer(raw.replace("<end_of_turn>", ""), final_answer_marker=True)
    row = {"case": case, "present": False, "zyx": None, "diameter_mm": None, "score": None, "candidates": [],
           "raw": raw, "parse_status": status, "parsed": f, "prompt": prompt,
           "fed": {"slice_z": zs, "image_hw": [int(vol.shape[1]), int(vol.shape[2])],
                   "orientation": "DICOM display: rows=y(A->P), cols=x(R->L); SLICE 1 = most superior",
                   "n_input_tokens": n_in, "n_new_tokens": n_new}}
    if f is not None:
        zyx, st = answer_to_zyx(f, zs, vol.shape)
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
         ("liver", "liver_3")]


def check(args):
    from scipy import ndimage as ndi
    import transformers
    mg = MedGemma() if args.model else None
    proc = mg.proc if mg else transformers.AutoProcessor.from_pretrained(MODEL, use_fast=True)
    report = []
    checks = [("liver_ext", args.case or "HCC_002")] if args.ds == "liver_ext" else CHECK
    for ds, case in checks:
        vol, sp = load_lps(ds, case)
        zs = fed_z(vol.shape[0])
        if ds == "liver_ext":                                     # evaluation set: image-only location, no labels
            c = label_free_center(vol)
        else:
            dom = np.load(f"{ROOT}/cache/{ds}/{case}.{'lung' if ds == 'luna' else 'liver'}.npy")
            dist = ndi.distance_transform_edt(dom, sampling=sp)
            c = np.array(np.unravel_index(int(np.argmax(dist)), dom.shape), float)
        k = int(np.argmin(np.abs(np.array(zs) - c[0])))           # centre on a fed slice
        center = np.array([float(zs[k]), round(c[1]), round(c[2])])
        syn, _ = add_sphere(vol, sp, center)
        pv = {}
        for tag, v in (("orig", vol), ("syn", syn)):
            msgs, _ = build_messages(ds, v, sp, zs)
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, continue_final_message=False,
                                           return_tensors="pt", tokenize=True, return_dict=True)
            pv[tag] = inp["pixel_values"].float().numpy()          # (n_images, 3, 896, 896), content order
        diff = np.clip(pv["syn"] - pv["orig"], 0, None).mean(1)
        t_hat = int(np.argmax(diff.reshape(len(diff), -1).sum(1)))
        d2 = diff[t_hat]
        cy, cx = ndi.center_of_mass(d2 * (d2 > 0.5 * d2.max()))
        hr, wr = d2.shape
        x_norm, y_norm = (cx + 0.5) / wr * 1000.0, (cy + 0.5) / hr * 1000.0
        back, _ = answer_to_zyx({"slice": t_hat + 1, "x": x_norm, "y": y_norm}, zs, vol.shape)
        rep = {"ds": ds, "case": case, "center_zyx": center.tolist(), "n_fed_slices": len(zs),
               "image_hw_seen_by_model": [hr, wr], "sphere_SLICE": t_hat + 1, "expected_SLICE": k + 1,
               "sphere_xy_norm1000": [round(x_norm, 1), round(y_norm, 1)], "back_zyx": np.round(back, 2).tolist(),
               "err_vox": (np.array(back) - center).round(2).tolist()}
        g = pv["syn"][t_hat].transpose(1, 2, 0)
        Image.fromarray(np.clip((g * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)).save(
            os.path.join(HERE, f"check_frame_{ds}.png"))
        if mg is not None:
            row = run_case(mg, ds, case, vol=syn)
            rep["model_raw_tail"] = row["raw"][-600:]
            if row["zyx"] is not None:
                rep["model_zyx"] = np.round(row["zyx"], 1).tolist()
                rep["model_dist_mm"] = round(float(np.linalg.norm((np.array(row["zyx"]) - center) * np.array(sp))), 1)
            rep["sec"] = row["sec"]
        print(json.dumps(rep), flush=True)
        report.append(rep)
    out = "coord_check_liver_ext.json" if args.ds == "liver_ext" else "coord_check.json"
    json.dump(report, open(os.path.join(HERE, out), "w"), indent=1)


def smoke(args):
    mg = MedGemma()
    v = args.variant or DEFAULT_VARIANT
    with open(os.path.join(HERE, f"smoke_variant{v}{'_extra' if args.extra_only else ''}.jsonl"), "w") as f:
        for ds in ("luna", "liver"):
            cases = SMOKE_EXTRA[ds] if args.extra_only else SMOKE[ds] + (SMOKE_EXTRA[ds] if args.extra else [])
            for case in cases:
                row = run_case(mg, ds, case, variant=v)
                row["ds"], row["variant"] = ds, v
                print(json.dumps({k: row[k] for k in ("ds", "case", "present", "zyx", "diameter_mm",
                                                       "parse_status", "sec")}), flush=True)
                print("RAW:", row["raw"][-700:], flush=True)
                f.write(json.dumps(row) + "\n")


def evaluate(args):
    """Each part writes its own file under parts/ (no concurrent appends); `merge` assembles results/sota/."""
    mg = MedGemma()
    part, nparts = (int(v) for v in args.part.split("/"))
    os.makedirs(os.path.join(HERE, "parts"), exist_ok=True)
    out = os.path.join(HERE, "parts", f"medgemma15_{args.ds}_part{part}of{nparts}.jsonl")
    done = {json.loads(l)["case"] for l in open(out)} if os.path.exists(out) else set()
    mine = eval_cases(args.ds)[part::nparts]
    t0 = time.time()
    for i, case in enumerate(mine):
        if case in done:
            continue
        row = run_case(mg, args.ds, case)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(mine)}] {case} {row['parse_status']} {row['raw'][-160:]!r} {row['sec']}s", flush=True)
    print(f"done {args.ds} part {args.part} in {time.time() - t0:.0f}s", flush=True)


def merge(args):
    import glob
    rows = {}
    for p in sorted(glob.glob(os.path.join(HERE, "parts", f"medgemma15_{args.ds}_part*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            rows[r["case"]] = r
    cases = eval_cases(args.ds)
    missing = [c for c in cases if c not in rows]
    with open(f"{ROOT}/results/sota/medgemma15_{args.ds}.jsonl", "w") as f:
        for c in cases:
            if c in rows:
                f.write(json.dumps(rows[c]) + "\n")
    print(f"merged {len(cases) - len(missing)}/{len(cases)} rows; missing {missing}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "smoke", "eval", "merge"])
    ap.add_argument("--ds", choices=["luna", "liver", "liver_ext"])
    ap.add_argument("--case", default=None, help="check --ds liver_ext: the case to use")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--extra", action="store_true", help="smoke: also the extra pool scans")
    ap.add_argument("--extra_only", action="store_true", help="smoke: only the extra pool scans")
    a = ap.parse_args()
    {"check": check, "smoke": smoke, "eval": evaluate, "merge": merge}[a.mode](a)
