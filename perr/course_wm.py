"""Learned disease-course world model: predict the follow-up CT from the baseline and from the follow-up's own
surroundings, and read the precision-weighted prediction error.

The anatomy world model (context_wm.py) predicts a hidden cube from its spatial context. Here the model also sees
the baseline CT registered onto the follow-up grid, in full: a 96 mm cube of follow-up CT with its central 36 mm
hidden, plus the same cube of registered baseline, and it predicts the hidden follow-up with its own uncertainty.
What the baseline already shows (anatomy, stable lesions) and what registration leaves slightly misplaced is predicted;
a lesion that is new, or larger than the baseline announces, is not. Training cubes never hide a target lesion (new or
grown), so the model learns the course of everything else.

Grid: 1.5 mm isotropic around the follow-up body (autoPET slices are mostly 3 mm), so the 64-voxel cube spans 96 mm
and the hidden 24 voxels span 36 mm. Intensities: HU clipped to [-1000, 1000] -> [0, 1]. Lesions can be brighter
(lung, nodes in fat) or darker (liver) than predicted, so the error is read on both signs.

No torch import at module level.
"""

from __future__ import annotations

import os
import zlib

import numpy as np
from scipy import ndimage as ndi

ISO = "/raid/home/CAMCA/hj880/perr/cache/iso_course"
H = 1.5                                   # mm per voxel
CTX, HOLE = 64, 24                        # voxels
MARGIN = 0
WINDOW = (-1000.0, 1000.0)


def norm(hu):
    lo, hi = WINDOW
    return np.clip((np.asarray(hu, np.float32) - lo) / (hi - lo), 0, 1)


def build_iso(key: str) -> None:
    """Resample a cached autoPET pair to 1.5 mm inside the follow-up body's bounding box: follow-up, registered
    baseline, the domain (body with baseline data), the target exclusion mask (target lesions grown by 6 mm, used only
    to keep them out of training) and training centres."""
    from .autopet import load_pair
    stem = f"{ISO}/{key}"
    if os.path.exists(stem + ".meta.npy"):
        return
    os.makedirs(ISO, exist_ok=True)
    P = load_pair(key)
    sp = np.array(P.spacing, float)
    dom0 = (np.asarray(P.body) > 0) & (np.asarray(P.bl_valid) > 0)
    if not dom0.any():
        np.save(stem + ".meta.npy", {"empty": True})
        return
    idx = [np.nonzero(dom0.any(axis=ax))[0] for ax in ((1, 2), (0, 2), (0, 1))]
    lo = np.array([a.min() for a in idx]) * sp
    hi = np.array([a.max() for a in idx]) * sp
    shape = tuple(int(v) for v in np.floor((hi - lo) / H) + 1)
    mat, off = H / sp, lo / sp
    rs = lambda a, order=1: ndi.affine_transform(np.asarray(a, np.float32), mat, offset=off, output_shape=shape,
                                                 order=order, mode="nearest")
    FU, BL = rs(P.fu), rs(P.bl)
    D = rs(dom0.astype(np.float32)) > 0.5
    tl = set(int(v) for v in P.targets["fu_label"]) if len(P.targets) else set()
    lab = np.asarray(P.fu_lab)
    tmask = np.isin(lab, sorted(tl)) if tl else np.zeros(lab.shape, bool)
    T = rs(tmask.astype(np.float32)) > 0.5
    ex = ndi.distance_transform_edt(~T) * H <= 6 if T.any() else np.zeros(shape, bool)
    np.save(stem + ".fu.npy", np.round(FU).astype(np.int16))
    np.save(stem + ".bl.npy", np.round(BL).astype(np.int16))
    np.save(stem + ".dom.npy", D.astype(np.uint8))
    np.save(stem + ".excl.npy", ex.astype(np.uint8))
    h = CTX // 2
    clear = ndi.maximum_filter(ex, size=HOLE + 1) == 0
    inner = np.zeros(shape, bool)
    if min(shape) > CTX:
        inner[h:shape[0] - h, h:shape[1] - h, h:shape[2] - h] = True
    ok = D & clear & inner
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    allc = np.argwhere(ok)
    dense = np.argwhere(ok & (np.abs(FU - BL) > 50))           # where the two time points differ: the hard cases
    pick = lambda a, n: a[rng.choice(len(a), size=min(n, len(a)), replace=False)] if len(a) else a.reshape(0, 3)
    np.save(stem + ".centres.npy", np.concatenate([pick(allc, 20000), pick(dense, 20000)]).astype(np.int16))
    np.save(stem + ".meta.npy", {"lo_mm": lo, "spacing": sp, "shape": shape, "n_all": len(allc), "n_dense": len(dense)})


def all_centres(key: str):
    """Training centres without the target exclusion, for a world model trained without any lesion annotation: the
    same draw as build_iso (inside the domain, the cube inside the volume, half where the two time points differ) with
    target lesions left in. Cached as <key>.centres_all.npy."""
    stem = f"{ISO}/{key}"
    if not os.path.exists(stem + ".centres_all.npy"):
        meta = np.load(stem + ".meta.npy", allow_pickle=True).item()
        if meta.get("empty"):
            return np.zeros((0, 3), np.int16)
        FU, BL, D, _ = load_iso(key)
        shape, h = D.shape, CTX // 2
        inner = np.zeros(shape, bool)
        if min(shape) > CTX:
            inner[h:shape[0] - h, h:shape[1] - h, h:shape[2] - h] = True
        ok = (np.asarray(D) > 0) & inner
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        allc = np.argwhere(ok)
        dense = np.argwhere(ok & (np.abs(np.asarray(FU, np.float32) - np.asarray(BL, np.float32)) > 50))
        pick = lambda a, n: a[rng.choice(len(a), size=min(n, len(a)), replace=False)] if len(a) else a.reshape(0, 3)  # noqa: E731
        np.save(stem + ".centres_all.npy", np.concatenate([pick(allc, 20000), pick(dense, 20000)]).astype(np.int16))
    return np.load(stem + ".centres_all.npy")


def load_iso(key: str):
    stem = f"{ISO}/{key}"
    meta = np.load(stem + ".meta.npy", allow_pickle=True).item()
    return (np.load(stem + ".fu.npy", mmap_mode="r"), np.load(stem + ".bl.npy", mmap_mode="r"),
            np.load(stem + ".dom.npy", mmap_mode="r"), meta)


def to_original(zyx_iso, meta):
    """1.5 mm grid index -> continuous voxel index of the native follow-up grid."""
    return (np.asarray(zyx_iso, float) * H + meta["lo_mm"]) / meta["spacing"]
