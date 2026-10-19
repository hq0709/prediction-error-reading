"""Further liver-tumour training data: MSD Task08 Hepatic Vessel (Memorial Sloan Kettering; CC BY-SA 4.0), portal-venous
CT of patients with primary or metastatic liver tumours; 303 labelled training cases with masks of the hepatic vessels
(label 1) and the tumours (label 2). The set has no liver mask, so the liver is TotalSegmentator's (v2, class "liver",
run on each image with --roi_subset liver; segment()), joined with the tumour mask and filled axially like the CRLM
livers. Cached exactly like liver_crlm.load_case (vol int16 [z, y, x] in LPS, liver uint8, tumour components with
equivalent diameter, big / small = components >= / < 10 mm grown by a 3 mm ball). Case names: hepaticvessel_004 ->
T08_004. Train on usable_cases() (LIST): a tumour of at least one voxel and a liver mask of
at least 300 ml. TotalSegmentator's liver often stops at a large tumour (in 8 of the first 180 cases under half of
the tumour voxels fall inside it); the domain is the union, so such tumours stay inside it.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .liver import MIN_DIAM, TOL_MM, has_target  # noqa: F401  (same target definition as the MSD world)
from .study import Nodule, Study

ROOT = "/raid/home/CAMCA/hj880/msd_extra/Task08_HepaticVessel"
SEG = "/raid/home/CAMCA/hj880/data_ext/t08_totalseg"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/liver_t08"
LIST = "/raid/home/CAMCA/hj880/perr/results/t08_cases.txt"
TS = "/raid/home/CAMCA/hj880/venvs/totalseg/bin/TotalSegmentator"
INSIDE_MIN = 0.0


def to_case(name: str) -> str:
    return "T08_" + name.split("_")[1].split(".")[0]


def to_name(case: str) -> str:
    return "hepaticvessel_" + case.split("_")[1]


def cases() -> list[str]:
    return sorted(to_case(f) for f in os.listdir(f"{ROOT}/imagesTr") if f.startswith("hepaticvessel_"))


def segment(case: str, device: str = "gpu") -> str:
    """TotalSegmentator's liver mask of one case (skipped when present)."""
    out = f"{SEG}/{case}"
    if not os.path.exists(f"{out}/liver.nii.gz"):
        os.makedirs(out, exist_ok=True)
        subprocess.run([TS, "-i", f"{ROOT}/imagesTr/{to_name(case)}.nii.gz", "-o", out, "--roi_subset", "liver",
                        "--device", device, "--quiet"], check=True)
    return f"{out}/liver.nii.gz"


def load_case(case: str) -> Study:
    """Study with lung := liver mask (the domain) and nodules := tumour components; caches like liver_crlm.load_case."""
    os.makedirs(CACHE, exist_ok=True)
    stem = f"{CACHE}/{case}"
    if not os.path.exists(stem + ".meta.npy"):
        img = sitk.DICOMOrient(sitk.ReadImage(f"{ROOT}/imagesTr/{to_name(case)}.nii.gz"), "LPS")
        lab = sitk.DICOMOrient(sitk.ReadImage(f"{ROOT}/labelsTr/{to_name(case)}.nii.gz"), "LPS")
        seg = sitk.DICOMOrient(sitk.ReadImage(f"{SEG}/{case}/liver.nii.gz"), "LPS")
        vol = np.clip(sitk.GetArrayFromImage(img), -1024, 3071).astype(np.int16)
        lab = sitk.GetArrayFromImage(lab).astype(np.uint8)
        ts = sitk.GetArrayFromImage(seg) > 0
        assert ts.shape == vol.shape == lab.shape, case
        sx, sy, sz = img.GetSpacing()
        spacing = (sz, sy, sx)
        tum = lab == 2
        liver = np.stack([ndi.binary_fill_holes(s) for s in (ts | tum)])
        comp, n = ndi.label(tum, structure=np.ones((3, 3, 3)))
        nods, big, small = [], np.zeros(tum.shape, bool), np.zeros(tum.shape, bool)
        for i, sl in enumerate(ndi.find_objects(comp), 1):
            m = comp[sl] == i
            v = float(m.sum()) * sz * sy * sx
            d = (6 * v / np.pi) ** (1 / 3)
            c = np.argwhere(m).mean(0) + np.array([s.start for s in sl])
            nods.append(((float(c[0]), float(c[1]), float(c[2])), d))
            (big if d >= MIN_DIAM else small)[sl] |= m
        r = [max(1, int(round(TOL_MM / s))) for s in spacing]
        zz, yy, xx = np.ogrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
        ball = (zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2 <= TOL_MM ** 2
        inside = float((tum & ts).sum()) / max(int(tum.sum()), 1)
        np.save(stem + ".vol.npy", vol)
        np.save(stem + ".liver.npy", liver.astype(np.uint8))
        np.save(stem + ".big.npy", ndi.binary_dilation(big, ball).astype(np.uint8))
        np.save(stem + ".small.npy", ndi.binary_dilation(small, ball).astype(np.uint8))
        np.save(stem + ".meta.npy", {"spacing": spacing, "nodules": nods, "tumour_inside_ts": inside,
                                     "ts_liver_ml": float(ts.sum()) * sz * sy * sx / 1000})
    meta = np.load(stem + ".meta.npy", allow_pickle=True).item()
    return Study(case, np.load(stem + ".vol.npy", mmap_mode="r"), tuple(meta["spacing"]),
                 np.load(stem + ".liver.npy", mmap_mode="r"), "", [Nodule(tuple(z), d) for z, d in meta["nodules"]])


def label_at(case: str, zyx) -> str:
    z, y, x = (int(round(c)) for c in zyx)
    big = np.load(f"{CACHE}/{case}.big.npy", mmap_mode="r")
    if big[z, y, x]:
        return "big"
    return "small" if np.load(f"{CACHE}/{case}.small.npy", mmap_mode="r")[z, y, x] else "none"


def eligibility(case: str) -> list[str]:
    m = np.load(f"{CACHE}/{case}.meta.npy", allow_pickle=True).item()
    why = []
    if not m["nodules"]:
        why.append("no tumour")
    if m["ts_liver_ml"] < 300:
        why.append(f"liver mask {m['ts_liver_ml']:.0f} ml")
    if m["nodules"] and m["tumour_inside_ts"] < INSIDE_MIN:
        why.append(f"{100 * m['tumour_inside_ts']:.0f} % of tumour voxels inside the liver mask")
    return why


def usable_cases(write: bool = False) -> list[str]:
    ok = [c for c in cases() if os.path.exists(f"{CACHE}/{c}.meta.npy") and not eligibility(c)]
    if write:
        with open(LIST, "w") as f:
            f.write("\n".join(ok) + "\n")
    return ok
