"""Lossless study store: raw volumes in LPS voxel order, with spacing and masks, cached as .npy."""

from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import SimpleITK as sitk

LUNA = "/raid/home/CAMCA/hj880/data_ext/luna16"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/luna"


@dataclass
class Nodule:
    zyx: tuple[float, float, float]  # continuous voxel index in the LPS-ordered array
    diameter_mm: float


@dataclass
class Study:
    case_id: str
    vol: np.ndarray            # int16 HU, [z, y, x]; x -> patient left, y -> posterior, z -> superior
    spacing: tuple[float, float, float]  # (sz, sy, sx) in mm
    lung: np.ndarray           # uint8 mask, 1 = lung (both sides), same grid
    sha256: str                # hash of the raw voxels: the study is immutable evidence
    nodules: list[Nodule]

    @property
    def shape(self):
        return self.vol.shape


def _mhd_path(uid: str) -> str:
    hits = glob.glob(f"{LUNA}/subset[0-9]/{uid}.mhd")
    if not hits:
        raise FileNotFoundError(uid)
    return hits[0]


def _load_lps(path: str) -> sitk.Image:
    return sitk.DICOMOrient(sitk.ReadImage(path), "LPS")


def annotations() -> pd.DataFrame:
    return pd.read_csv(f"{LUNA}/annotations.csv")


def load_luna(uid: str) -> Study:
    os.makedirs(CACHE, exist_ok=True)
    stem = f"{CACHE}/{uid}"
    ann = annotations()
    rows = ann[ann.seriesuid == uid]
    if os.path.exists(stem + ".vol.npy"):
        vol = np.load(stem + ".vol.npy", mmap_mode="r")
        lung = np.load(stem + ".lung.npy", mmap_mode="r")
        meta = np.load(stem + ".meta.npy", allow_pickle=True).item()
        spacing, nod_idx, sha = meta["spacing"], meta["nodules"], meta["sha256"]
    else:
        img = _load_lps(_mhd_path(uid))
        vol = sitk.GetArrayFromImage(img).astype(np.int16)
        sx, sy, sz = img.GetSpacing()
        spacing = (sz, sy, sx)
        mimg = _load_lps(f"{LUNA}/seg-lungs-LUNA16/{uid}.mhd")
        m = sitk.GetArrayFromImage(mimg)
        if m.shape != vol.shape:
            raise ValueError(f"mask/volume shape mismatch for {uid}")
        lung = ((m == 3) | (m == 4)).astype(np.uint8)
        nod_idx = []
        for r in rows.itertuples():
            ix, iy, iz = img.TransformPhysicalPointToContinuousIndex((r.coordX, r.coordY, r.coordZ))
            nod_idx.append(((iz, iy, ix), float(r.diameter_mm)))
        sha = hashlib.sha256(vol.tobytes()).hexdigest()
        np.save(stem + ".vol.npy", vol)
        np.save(stem + ".lung.npy", lung)
        np.save(stem + ".meta.npy", {"spacing": spacing, "nodules": nod_idx, "sha256": sha})
    nodules = [Nodule(tuple(z), d) for z, d in nod_idx]
    return Study(uid, np.asarray(vol), tuple(spacing), np.asarray(lung), sha, nodules)


def pilot_cases(n: int = 50, min_diam: float = 6.0, seed: int = 0) -> list[str]:
    """Scans whose largest annotated nodule is >= min_diam, sampled once with a fixed seed."""
    ann = annotations()
    mx = ann.groupby("seriesuid").diameter_mm.max()
    uids = sorted(mx[mx >= min_diam].index)
    rng = np.random.default_rng(seed)
    return [uids[i] for i in rng.permutation(len(uids))[:n]]
