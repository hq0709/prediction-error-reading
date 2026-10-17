"""Shared I/O for the baseline runs (an identical copy lives in each sota/<method>/ folder).

Reads voxel data, spacing and header geometry only; never labels or annotations of evaluation cases.
Output index space: every zyx is a continuous index into the cached array cache/<ds>/<case>.vol.npy
([z, y, x]; x -> patient left, y -> posterior, z -> superior).
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re

import numpy as np
import SimpleITK as sitk

ROOT = "/raid/home/CAMCA/hj880/perr"
LUNA_RAW = "/raid/home/CAMCA/hj880/data_ext/luna16"
LIVER_RAW = "/raid/home/CAMCA/hj880/msd/Task03_Liver/imagesTr"
LIVER_EXT_RAW = "/raid/home/CAMCA/hj880/data_ext/hcc_tace_seg/nifti/imagesTr"   # HCC-TACE-Seg (external liver set)


def prompt_key(ds: str) -> str:
    """The external liver set uses the MSD liver prompt."""
    return "liver" if ds == "liver_ext" else ds


# ----------------------------------------------------------------------------------------------- cases
def eval_cases(ds: str) -> list[str]:
    if ds == "luna":
        return [u for n in ("fresh1", "fresh2") for u in open(f"{ROOT}/sota/eval_luna_{n}.txt").read().split()]
    if ds == "liver_ext":
        return open(f"{ROOT}/sota/eval_liver_ext.txt").read().split()
    return open(f"{ROOT}/sota/eval_liver.txt").read().split()


def pool_cases(ds: str) -> list[str]:
    """Training-pool cases (smoke tests and coordinate checks only)."""
    if ds == "luna":
        return open(f"{ROOT}/results/train_uids.txt").read().split()
    ev = set(eval_cases("liver"))
    cs = [f[:-7] for f in os.listdir(LIVER_RAW) if f.endswith(".nii.gz") and not f.startswith(".")]
    return sorted((c for c in cs if c not in ev), key=lambda c: int(c.split("_")[1]))


def original_path(ds: str, case: str) -> str:
    if ds == "luna":
        return glob.glob(f"{LUNA_RAW}/subset[0-9]/{case}.mhd")[0]
    if ds == "liver_ext":
        return f"{LIVER_EXT_RAW}/{case}.nii.gz"
    return f"{LIVER_RAW}/{case}.nii.gz"


# ----------------------------------------------------------------------------------------------- volumes
def header_spacing(ds: str, case: str) -> tuple[float, float, float]:
    """(sz, sy, sx) of the LPS-reoriented original, from the file header (same numbers the cache stores)."""
    r = sitk.ImageFileReader()
    r.SetFileName(original_path(ds, case))
    r.ReadImageInformation()
    sp_ijk = r.GetSpacing()
    out = [0.0, 0.0, 0.0]
    for n, (m, _) in enumerate(raw_axes(ds, case)):
        out[m] = float(sp_ijk[2 - n])
    return tuple(out)


def spacing(ds: str, case: str) -> tuple[float, float, float]:
    """(sz, sy, sx) in mm, as documented in sota/README.md (only the 'spacing' key is read). numpy < 1.26 cannot
    unpickle the numpy-2 meta files; those environments fall back to the header (checked equal for all cases)."""
    try:
        meta = np.load(f"{ROOT}/cache/{ds}/{case}.meta.npy", allow_pickle=True).item()
        return tuple(float(s) for s in meta["spacing"])
    except ModuleNotFoundError:
        return header_spacing(ds, case)


def load_lps(ds: str, case: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Volume in the cached LPS index space with the ORIGINAL voxel values, and spacing (sz, sy, sx).

    LUNA16: the cache is a lossless int16 copy of the .mhd voxels. Liver: the cache is clipped to [-1024, 3071],
    so the original NIfTI is reoriented exactly as the cache was built (DICOMOrient 'LPS') and checked against it.
    liver_ext (HCC-TACE-Seg): same path; its NIfTI was itself written clipped to [-1024, 3071].
    """
    cached = np.load(f"{ROOT}/cache/{ds}/{case}.vol.npy")
    if ds == "luna":
        vol = cached
    else:
        img = sitk.DICOMOrient(sitk.ReadImage(original_path(ds, case)), "LPS")
        vol = sitk.GetArrayFromImage(img)
        assert vol.shape == cached.shape, (case, vol.shape, cached.shape)
        assert np.array_equal(np.clip(vol, -1024, 3071).astype(np.int16), cached), case
    return vol, spacing(ds, case)


def raw_axes(ds: str, case: str) -> list[tuple[int, int]]:
    """For each numpy axis n of the original stored array ([k, j, i] order): (cached LPS array axis, sign)."""
    r = sitk.ImageFileReader()
    r.SetFileName(original_path(ds, case))
    r.ReadImageInformation()
    d = np.array(r.GetDirection()).reshape(3, 3)  # column a = direction of index axis a in LPS physical space
    out = []
    for n in range(3):
        a = 2 - n
        p = int(np.argmax(np.abs(d[:, a])))
        assert abs(abs(d[p, a]) - 1.0) < 1e-3, (case, d)
        out.append((2 - p, 1 if d[p, a] > 0 else -1))
    return out


def lps_to_raw(lps: np.ndarray, axes) -> np.ndarray:
    arr = np.transpose(lps, [m for m, _ in axes])
    for n, (_, s) in enumerate(axes):
        if s < 0:
            arr = np.flip(arr, n)
    return np.ascontiguousarray(arr)


def raw_index_to_lps(r, axes, raw_shape) -> list[float]:
    out = [0.0, 0.0, 0.0]
    for n, (m, s) in enumerate(axes):
        out[m] = float(r[n]) if s > 0 else float(raw_shape[n] - 1 - r[n])
    return out


def clip_zyx(zyx, shape) -> list[float]:
    return [float(min(max(c, 0.0), n - 1.0)) for c, n in zip(zyx, shape)]


def add_sphere(vol: np.ndarray, sp, center, diameter_mm: float = 10.0, hu: float = 300.0):
    """Copy of vol with a sphere of the given diameter set to `hu` at continuous index `center` (z, y, x)."""
    out = vol.copy()
    r = diameter_mm / 2.0
    lo = [max(0, int(np.floor(c - r / s)) - 1) for c, s in zip(center, sp)]
    hi = [min(n, int(np.ceil(c + r / s)) + 2) for c, s, n in zip(center, sp, vol.shape)]
    zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    m = ((zz - center[0]) * sp[0]) ** 2 + ((yy - center[1]) * sp[1]) ** 2 + ((xx - center[2]) * sp[2]) ** 2 <= r * r
    sub = out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub[m] = hu
    mask = np.zeros(vol.shape, bool)
    mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = m
    return out, mask


def label_free_center(vol: np.ndarray) -> np.ndarray:
    """Image-only sphere location for the coordinate check on an evaluation set (no labels read): on the middle slice,
    the voxel of the largest soft-tissue region (-100..150 HU) farthest from its boundary."""
    from scipy import ndimage as ndi
    z = vol.shape[0] // 2
    lab, n = ndi.label((vol[z] > -100) & (vol[z] < 150))
    big = lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
    d = ndi.distance_transform_edt(big)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return np.array([float(z), float(y), float(x)])


# ----------------------------------------------------------------------------------------------- answers
_KEYS = {"present": ("present", "exists", "found"),
         "slice": ("slice", "slice_index", "slice_number", "slice_idx", "z"),
         "x": ("x", "x_coordinate", "col", "column"),
         "y": ("y", "y_coordinate", "row"),
         "diameter_mm": ("diameter_mm", "diameter", "size_mm", "size")}


def _json_objects(text: str) -> list[str]:
    """Every balanced {...} substring, ordered by start position (nested ones included, so a JSON object wrapped in
    LaTeX such as $\\boxed{\\{"present": ...\\}}$ is still found)."""
    objs = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    objs.append(text[i:j + 1])
                    break
    return objs


def _loads(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    t = re.sub(r",\s*}", "}", s)
    t = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", t)))
    try:
        v = ast.literal_eval(t)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return _num(v[0])
    if isinstance(v, str):
        m = re.search(r"[-+]?\d*\.?\d+", v)
        return float(m.group()) if m else None
    return None


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "yes", "present", "y"):
            return True
        if t in ("false", "no", "absent", "none", "n"):
            return False
    return None


_NEG = re.compile(r"\b(no|without)\b[^.]{0,60}\b(nodules?|tumou?rs?|lesions?|mass(es)?)\b|\bnot (present|seen|identified|found)\b",
                  re.I)
_POS = re.compile(r"\b(nodules?|tumou?rs?|lesions?|mass(es)?)\b", re.I)


def _text_fallback(seg: str) -> dict:
    """No JSON: coordinates only if stated explicitly (slice N ... x=.. y=..); presence as stated."""
    out = {"present": None, "slice": None, "x": None, "y": None, "diameter_mm": None}
    m_s = re.search(r"\bslice\s*(?:number|index)?\s*[:=#]?\s*(\d+(?:\.\d+)?)", seg, re.I)
    m_x = re.search(r"\bx\s*[:=]\s*(-?\d+(?:\.\d+)?)", seg)
    m_y = re.search(r"\by\s*[:=]\s*(-?\d+(?:\.\d+)?)", seg)
    if m_s and m_x and m_y:
        out.update(slice=float(m_s.group(1)), x=float(m_x.group(1)), y=float(m_y.group(1)), present=True)
        m_d = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", seg, re.I)
        if m_d:
            out["diameter_mm"] = float(m_d.group(1)) * (10.0 if m_d.group(2).lower() == "cm" else 1.0)
    elif _NEG.search(seg):
        out["present"] = False
    elif _POS.search(seg):
        out["present"] = True
    return out


def parse_json_answer(text: str, final_answer_marker: bool = False) -> tuple[dict | None, str]:
    """Extract {present, slice, x, y, diameter_mm} from a VLM answer. Returns (fields or None, status).

    Rule (fixed before any evaluation run): drop a thinking block if present and LaTeX brace escapes; if
    `final_answer_marker`, prefer the first JSON object after the last 'Final Answer'; otherwise take the last JSON
    object in the text that parses. Without any JSON object: status 'text_fallback' with coordinates only when
    slice, x= and y= are all stated explicitly, and presence as stated; otherwise None / 'unparseable'.
    """
    t = text.replace("\\{", "{").replace("\\}", "}")
    for tag in ("<unused95>", "</think>"):
        if tag in t:
            t = t.split(tag, 1)[1]
    segs = []
    if final_answer_marker:
        i = t.lower().rfind("final answer")
        if i >= 0:
            segs.append((t[i:], "first"))
    segs.append((t, "last"))
    for seg, which in segs:
        objs = _json_objects(seg)
        if which == "last":
            objs = objs[::-1]
        for o in objs:
            d = _loads(o)
            if not isinstance(d, dict):
                continue
            low = {str(k).strip().lower(): v for k, v in d.items()}
            out = {}
            for key, aliases in _KEYS.items():
                val = next((low[a] for a in aliases if a in low), None)
                out[key] = _bool(val) if key == "present" else _num(val)
            return out, "ok"
    if final_answer_marker:
        i = t.lower().rfind("final answer")
        seg = t[i:] if i >= 0 else t
        fb = _text_fallback(seg)
        if fb["present"] is not None:
            return fb, "text_fallback"
    return None, "unparseable"
