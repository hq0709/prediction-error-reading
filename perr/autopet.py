"""Disease-course world: autoPET IV Longitudinal-CT, baseline -> follow-up CT pairs (CC BY-NC 4.0).

Licence: non-commercial use only; never redistribute this cache or anything derived from it.

A world model predicts the follow-up CT from the baseline registered onto it; new or growing lesions are where the
follow-up departs from that prediction. This module is the data layer only: native follow-up volumes, the baseline
registered onto each follow-up grid from the images alone, body masks, lesion labels and lesion tables.

Source (ROOT): inputsTr/<pid>_{BL,FU}_img_NN.nii.gz, inputsTr/<pid>_BL_mask_NN.nii.gz,
targetsTr/<pid>_FU_mask_NN.nii.gz, inputsTr/<pid>.csv, data_split.json (patients: train 240 / val 30 / test 30).

Conventions, verified on every lesion entry (4079 BL, 3231 FU)
* All images carry identity direction in ITK's LPS frame, so DICOMOrient(img, "LPS") is a no-op; arrays are
  [z, y, x] with x -> patient left, y -> posterior, z -> superior (as in study.py).
* cog_bl, cog_fu, cog_propagated, cog_backpropagated are continuous voxel coordinates "x y z" in ITK (= NIfTI i, j, k)
  order on the piece named by img_id_bl / img_id_fu, with voxel centres at +0.5: cog = centroid(mask == label) + 0.5 on
  every axis. Array index [z, y, x] = (cog_z - 0.5, cog_y - 0.5, cog_x - 0.5). They are not millimetres.
* volume_bl / volume_fu are mm^3 (voxel count x voxel volume); tables here report ml.
* Mask value = lesion_id. In FU masks a MERGING group appears once, under its merged_into id.
* topology_class: UNCHANGED (one-to-one match; the size may change), NEWLYAPPEARING (FU only), DISAPPEARING (BL only),
  MERGING (several BL lesions become one FU lesion, merged_into = FU label). There is no SPLIT class.
* cog_propagated is the organisers' registration of the BL point into FU. It is never used here except as a QA
  reference; registration never sees lesion masks or CSV coordinates.

Pieces (NN). A study can be split into several series of one session: a trunk series plus a head/neck or brain
series, regional series (thorax / abdomen / legs), or re-reconstructions of one region (thin and thick slices, lung and
soft kernels, contrast phases). Every lesion is annotated in exactly one piece. Rule:
* Every FU piece is its own pair "<pid>_<NN>" on its native grid. Its owned lesions are the CSV lesions with
  img_id_fu == NN (centroid and volume exactly as in the CSV). Lesions owned by another FU piece of the same session
  that overlaps this one and shares its world frame are projected onto this grid (nearest neighbour, world
  coordinates) and kept in the label map with owned = False: no visible lesion is left unlabelled, none counted twice.
* The BL side of a pair is assembled from all BL pieces. Each BL piece is registered to the FU piece separately; pieces
  are composited in order of similarity to the FU (canvas NCC after registration), a later piece only filling voxels
  the earlier ones left empty. A secondary piece is used only if its mapping agrees with the anchor piece's (same
  session frame, median <= 15 mm) or it is itself very similar (NCC >= 0.7, e.g. a brain series in its own frame);
  pieces that register to nothing are dropped. bl_valid marks voxels that received BL data.

Registration, BL piece -> FU piece, images only: 4 mm canvases (anti-aliased, HU clipped to [-1024, 1500]) with body
masks; global 3-D translation by masked normalised cross-correlation (FFT over all shifts, >= 30 % overlap) at 8 mm,
plus the main pair's affine as a candidate for secondary pieces (same-session frames); rigid (Euler, Mattes MI) from
each candidate at 8 mm, the best by NCC kept (the frame candidate wins ties within 0.05); rigid -> affine (Mattes MI)
at 8/4 mm; cubic B-spline (60 mm control mesh, coefficients bounded to +-15 mm) at 8/4 mm, LBFGS-B, mean squared
difference of HU clipped to [-1000, 1000]; metric masks = body dilated 8 mm, inside the field of view. The bound keeps
the deformation at posture / breathing scale (no folding; Jacobian 0.36-2.25 over all 379 registrations used):
unbounded, the B-spline folded and pulled baseline anatomy into follow-up effusions and shrunken tumours, i.e. it
explained away the very changes the world model must see (a two-level variant with a 120 mm mesh and +-40-60 mm folded
the same way). The price: breathing differences beyond about 25 mm at the diaphragm are not absorbed and remain as
residual at the lung bases / liver dome. Guards, all image-based: the rigid is kept when the affine lowers the canvas
NCC (MI drift with very different fields of view); the linear transform is kept when the B-spline lowers it or folds;
a B-spline that did not move is reseeded once. The BL is resampled at native FU resolution through a 2 mm displacement
field of the final transform (linear for HU, nearest for labels).

Targets: owned FU lesions that are NEWLYAPPEARING, or matched (UNCHANGED, or a MERGING group against the summed BL
volume of its members) with volume_fu / volume_bl >= 1.2, excluding linking_unclear (NaN counts as not unclear). Each
target keeps its equivalent diameter (6 V / pi)^(1/3) for a size threshold applied later.

Cache (CACHE/<pair>.*, mmap-friendly .npy; <pair>.meta.npy is written last and marks completion):
  fu int16 HU | bl int16 HU (-1024 where no BL) | bl_valid uint8 | body uint8 | fu_lab uint8 | bl_lab uint8 |
  meta (dict: geometry, lesion table, targets, registration record, QA) | bl<QQ>.tfm (FU -> BL piece QQ transform).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import fft as sfft
from scipy import ndimage as ndi

ROOT = "/raid/home/CAMCA/hj880/data_ext/autopet4_longitudinal_ct"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/autopet"
VERSION = 1
GROWTH = 1.2                  # matched lesion is a target when volume_fu / volume_bl >= GROWTH
CANVAS_MM = 4.0               # registration grid; the global search runs at 2 x CANVAS_MM
CLIP_HU = (-1024.0, 1500.0)
BODY_HU = -500.0
MIN_OVERLAP = 0.3             # global search: overlap >= this fraction of the smaller body
MESH_MM = 60.0                # B-spline control-point spacing
BOUND_MM = 15.0               # |B-spline coefficient| <= BOUND_MM: posture/breathing scale only, no folding
FIELD_MM = 2.0                # displacement-field spacing used for native resampling
QA_FLAG_MM = 15.0
SEED = 2026                   # metric sampling seed (SimpleITK treats 0 as 'wall clock')
_ARRAYS = ("fu", "bl", "bl_valid", "body", "fu_lab", "bl_lab")


# --------------------------------------------------------------------------------------------- dataset index

def _split() -> dict:
    with open(f"{ROOT}/data_split.json") as f:
        return json.load(f)


def patients(split: str | None = None) -> list[str]:
    """Patient ids of a split (train / val / test), or of all splits when split is None."""
    s = _split()
    return sorted(s[split]) if split else sorted(p for v in s.values() for p in v)


def split_of(pid: str) -> str:
    for k, v in _split().items():
        if pid in v:
            return k
    raise KeyError(pid)


def pieces(pid: str, side: str = "FU") -> list[int]:
    out, n = [], 0
    while os.path.exists(f"{ROOT}/inputsTr/{pid}_{side}_img_{n:02d}.nii.gz"):
        out.append(n)
        n += 1
    return out


def pair_ids(pid: str) -> list[str]:
    return [f"{pid}_{n:02d}" for n in pieces(pid, "FU")]


def pairs(split: str | None = None) -> list[str]:
    """Pair ids "<pid>_<NN>" (one per FU piece) of a split."""
    return [k for p in patients(split) for k in pair_ids(p)]


def _resolve(key: str) -> tuple[str, int]:
    """Pair id, or a bare pid when that patient has a single FU piece."""
    if "_" in key:
        pid, nn = key.split("_")
        return pid, int(nn)
    fu = pieces(key, "FU")
    if len(fu) != 1:
        raise ValueError(f"{key} has {len(fu)} FU pieces; use one of {pair_ids(key)}")
    return key, 0


def _paths(pid: str, side: str, n: int) -> tuple[str, str]:
    img = f"{ROOT}/inputsTr/{pid}_{side}_img_{n:02d}.nii.gz"
    mask = f"{ROOT}/{'inputsTr' if side == 'BL' else 'targetsTr'}/{pid}_{side}_mask_{n:02d}.nii.gz"
    return img, mask


def csv(pid: str) -> pd.DataFrame:
    D = pd.read_csv(f"{ROOT}/inputsTr/{pid}.csv")
    if "linking_unclear" not in D:      # 38db3aed92 and 8fc4c7ab44 ship without the column: unknown, not unclear
        D["linking_unclear"] = np.nan
    return D


# --------------------------------------------------------------------------------------------- geometry helpers

def _read(path: str) -> tuple[sitk.Image, sitk.Image]:
    """(as stored, LPS-oriented)."""
    raw = sitk.ReadImage(path)
    return raw, sitk.DICOMOrient(raw, "LPS")


def _cog_phys(cog: str, raw: sitk.Image) -> np.ndarray:
    """Physical LPS point (mm) of a CSV cog string on the piece it refers to."""
    c = [float(v) - 0.5 for v in cog.split()]
    return np.array(raw.TransformContinuousIndexToPhysicalPoint(c))


def _phys_to_zyx(p, img: sitk.Image) -> np.ndarray:
    i, j, k = img.TransformPhysicalPointToContinuousIndex([float(v) for v in p])
    return np.array([k, j, i])


def _zyx_to_phys(zyx, img: sitk.Image) -> np.ndarray:
    z, y, x = (float(v) for v in zyx)
    return np.array(img.TransformContinuousIndexToPhysicalPoint((x, y, z)))


def _spacing_zyx(img: sitk.Image) -> tuple[float, float, float]:
    sx, sy, sz = img.GetSpacing()
    return (sz, sy, sx)


def _grid_like(img: sitk.Image, h: float, pixel=sitk.sitkFloat32) -> sitk.Image:
    """Isotropic grid of spacing h covering img's physical extent (voxel boxes, not centres)."""
    sp, size = np.array(img.GetSpacing()), np.array(img.GetSize())
    D = np.array(img.GetDirection()).reshape(3, 3)
    n = np.maximum(1, np.round(size * sp / h)).astype(int)
    start = np.array(img.GetOrigin()) - D @ (0.5 * sp)
    ref = sitk.Image([int(v) for v in n], pixel)
    ref.SetSpacing([h] * 3)
    ref.SetDirection(img.GetDirection())
    ref.SetOrigin(tuple(start + D @ np.full(3, 0.5 * h)))
    return ref


def _ones_like(img: sitk.Image) -> sitk.Image:
    one = sitk.Image(img.GetSize(), sitk.sitkUInt8) + 1
    one.CopyInformation(img)
    return one


# --------------------------------------------------------------------------------------------- body mask

def body_mask(vol: np.ndarray, spacing, valid: np.ndarray | None = None) -> np.ndarray:
    """Patient body on the grid of vol ([z, y, x] HU): air excluded, lungs and bowel gas included, couch removed.

    Couch removal and component selection run on a ~2 mm grid (opening by 2 px, components >= 10 % of the largest
    and >= 20 ml); the native threshold is then kept inside that region (dilated 1 px) and holes are filled per slice.
    """
    sp = np.asarray(spacing, float)
    f = np.maximum(1, np.round(2.0 / sp)).astype(int)
    thr = vol > BODY_HU
    if valid is not None:
        thr &= valid.astype(bool)
    tp = np.pad(thr, [(0, (-s) % k) for s, k in zip(thr.shape, f)])
    Z, Y, X = tp.shape
    c = tp.reshape(Z // f[0], f[0], Y // f[1], f[1], X // f[2], f[2]).sum((1, 3, 5), dtype=np.int32) * 2 >= f.prod()
    del tp
    st = np.zeros((1, 3, 3), bool)
    st[0, 1, :] = st[0, :, 1] = True
    c = ndi.binary_opening(c, st, iterations=max(1, int(round(4.0 / float(min(sp[1:] * f[1:]))))))
    lab, n = ndi.label(c)
    if n == 0:
        return np.zeros(vol.shape, bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    cvox = float(np.prod(sp * f)) / 1000.0  # ml per coarse voxel
    keep = (sizes >= 0.1 * sizes.max()) & (sizes * cvox >= 20.0)
    keep[0] = False
    c = ndi.binary_dilation(keep[lab], st, iterations=1)
    idx = [np.minimum(np.arange(s) // k, cs - 1) for s, k, cs in zip(vol.shape, f, c.shape)]
    m = c[np.ix_(*idx)] & thr
    for z in range(m.shape[0]):
        m[z] = ndi.binary_fill_holes(m[z])
    return m


# --------------------------------------------------------------------------------------------- canvases

@dataclass
class Canvas:
    img: sitk.Image        # float32 HU on the CANVAS_MM grid
    valid: sitk.Image      # uint8, inside the source field of view
    body: sitk.Image       # uint8
    mask: sitk.Image       # uint8, body dilated 8 mm and inside the field of view (metric mask)
    arr: np.ndarray        # float32 [z, y, x] of img
    body_arr: np.ndarray   # bool
    valid_arr: np.ndarray  # bool


def _canvas(img: sitk.Image, h: float = CANVAS_MM) -> Canvas:
    sp = np.array(img.GetSpacing())
    f = sitk.Cast(img, sitk.sitkFloat32)
    sig = [float(max(0.0, 0.4 * (h - s))) for s in sp]
    if max(sig) > 0:
        f = sitk.SmoothingRecursiveGaussian(f, [max(s, 1e-3) for s in sig])
    ref = _grid_like(img, h)
    out = sitk.Resample(f, ref, sitk.Transform(), sitk.sitkLinear, CLIP_HU[0], sitk.sitkFloat32)
    valid = sitk.Resample(_ones_like(img), ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    a = np.clip(sitk.GetArrayFromImage(out), *CLIP_HU).astype(np.float32)
    v = sitk.GetArrayFromImage(valid).astype(bool)
    a[~v] = CLIP_HU[0]
    b = body_mask(a, (h, h, h), v)
    k = max(1, int(round(8.0 / h)))
    m = ndi.binary_dilation(b, iterations=k) & v
    out = sitk.GetImageFromArray(a)
    out.CopyInformation(ref)

    def u8(x):
        im = sitk.GetImageFromArray(x.astype(np.uint8))
        im.CopyInformation(ref)
        return im
    return Canvas(out, valid, u8(b), u8(m), a, b, v)


def _shrink2(c: Canvas):
    """8 mm versions (block means) of a canvas: image, metric mask."""
    img = sitk.BinShrink(c.img, [2, 2, 2])
    m = sitk.BinShrink(sitk.Cast(c.mask, sitk.sitkFloat32), [2, 2, 2]) >= 0.5
    return img, sitk.Cast(m, sitk.sitkUInt8)


# --------------------------------------------------------------------------------------------- registration

def _mncc(f, mf, g, mg, min_frac=MIN_OVERLAP, workers=4, top=2, sep=6):
    """Masked normalised cross-correlation (Padfield 2012) over all integer shifts u ([z, y, x], moving index = fixed
    index + u), with overlap >= min_frac of the smaller mask. Returns up to `top` (ncc, u) maxima >= sep voxels
    apart."""
    f, g = f.astype(np.float64), g.astype(np.float64)
    mf, mg = mf.astype(np.float64), mg.astype(np.float64)
    F, G = f * mf, g * mg
    fs = [sfft.next_fast_len(a + b - 1, real=True) for a, b in zip(f.shape, g.shape)]

    def R(a):
        return sfft.rfftn(a, fs, workers=workers)

    def I(a, b):
        return sfft.irfftn(a * b, fs, workers=workers)
    cF, cF2, cM = np.conj(R(F)), np.conj(R(F * F)), np.conj(R(mf))
    G1, G2, Mg = R(G), R(G * G), R(mg)
    N = np.round(I(cM, Mg))
    Sf, Sg = I(cF, Mg), I(cM, G1)
    Sff, Sgg, Sfg = I(cF2, Mg), I(cM, G2), I(cF, G1)
    nmin = max(min_frac * min(mf.sum(), mg.sum()), 50.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        varf = Sff - Sf * Sf / N
        varg = Sgg - Sg * Sg / N
        ncc = (Sfg - Sf * Sg / N) / np.sqrt(np.clip(varf, 0, None) * np.clip(varg, 0, None))
    ok = (N >= nmin) & (varf > 1e-6 * N) & (varg > 1e-6 * N)
    ncc = np.where(ok & np.isfinite(ncc), ncc, -np.inf)
    # contiguous shift order: index i <-> u = i - (nf - 1), so neighbouring shifts are neighbouring indices
    off = [n - 1 for n in f.shape]
    ncc = np.roll(ncc, off, axis=(0, 1, 2))
    out = []
    for _ in range(top):
        i = int(np.argmax(ncc))
        v = float(ncc.flat[i])
        if not np.isfinite(v):
            break
        idx = np.unravel_index(i, ncc.shape)
        out.append((v, np.array([ix - o for ix, o in zip(idx, off)])))
        ncc[tuple(slice(max(0, ix - sep), ix + sep + 1) for ix in idx)] = -np.inf
    return out


def _registration(fixed: sitk.Image, moving: sitk.Image, fmask, mmask, shrink, smooth_mm, seed=SEED, pct=0.25):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(pct, seed)
    R.SetMetricFixedMask(fmask)
    R.SetMetricMovingMask(mmask)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel(shrink)
    R.SetSmoothingSigmasPerLevel(smooth_mm)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R


def _linear(fixed, moving, fmask, mmask, init, kind, shrink, smooth, iters=200):
    """Rigid or affine refinement of init (a sitk transform mapping fixed -> moving points)."""
    if kind == "rigid":
        t = sitk.Euler3DTransform()
        t.SetCenter(init.GetCenter() if hasattr(init, "GetCenter") else (0, 0, 0))
        if isinstance(init, sitk.Euler3DTransform):
            t.SetParameters(init.GetParameters())
            t.SetFixedParameters(init.GetFixedParameters())
    else:
        t = sitk.AffineTransform(3)
        e = sitk.Euler3DTransform(init) if init.GetName() == "Euler3DTransform" else None
        src = e if e is not None else sitk.AffineTransform(init)
        t.SetCenter(src.GetCenter())
        t.SetMatrix(src.GetMatrix())
        t.SetTranslation(src.GetTranslation())
    R = _registration(fixed, moving, fmask, mmask, shrink, smooth)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=4.0, minStep=0.01, numberOfIterations=iters,
                                               relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(t, inPlace=True)
    R.Execute(fixed, moving)
    return t, R.GetMetricValue()


def _bspline(fixed, moving, fmask, mmask, affine, seed=SEED):
    """Cubic B-spline on top of the affine: mean squared HU difference (HU clipped to [-1000, 1000]; same modality),
    LBFGS-B with bounded coefficients, 8 then 4 mm. (ITK's Correlation metric was tried first: with sparse random
    sampling its per-thread helper sometimes collects zero points and the optimizer stops at the start.)"""
    ext = np.array(fixed.GetSize()) * np.array(fixed.GetSpacing())
    mesh = [int(max(1, round(e / MESH_MM))) for e in ext]
    tx = sitk.BSplineTransformInitializer(fixed, mesh, 3)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.1, seed)
    R.SetMetricFixedMask(fmask)
    R.SetMetricMovingMask(mmask)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel([2, 1])
    R.SetSmoothingSigmasPerLevel([4.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100, maximumNumberOfCorrections=5,
                           maximumNumberOfFunctionEvaluations=200, costFunctionConvergenceFactor=1e7,
                           lowerBound=-BOUND_MM, upperBound=BOUND_MM)
    R.SetMovingInitialTransform(affine)
    R.SetInitialTransform(tx, inPlace=True)
    R.Execute(sitk.Clamp(fixed, sitk.sitkFloat32, -1000.0, 1000.0),
              sitk.Clamp(moving, sitk.sitkFloat32, -1000.0, 1000.0))
    return tx, R.GetMetricValue()


def _score(fc: Canvas, mc: Canvas, T) -> tuple[float, float]:
    """(NCC over the FU body voxels that the BL field of view reaches, fraction of FU body reached) at CANVAS_MM."""
    w = sitk.GetArrayFromImage(sitk.Resample(mc.img, fc.img, T, sitk.sitkLinear, CLIP_HU[0], sitk.sitkFloat32))
    v = sitk.GetArrayFromImage(sitk.Resample(mc.valid, fc.img, T, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
    m = fc.body_arr & v
    cover = float(m.sum()) / max(1, int(fc.body_arr.sum()))
    if m.sum() < 100:
        return -1.0, cover
    a = np.clip(fc.arr[m], -1000, 1000).astype(np.float64)
    b = np.clip(w[m], -1000, 1000).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9)), cover


def _centroid_phys(c: Canvas) -> tuple:
    zyx = np.argwhere(c.body_arr).mean(0) if c.body_arr.any() else (np.array(c.arr.shape) - 1) / 2
    return tuple(_zyx_to_phys(zyx, c.img))


def register(fc: Canvas, mc: Canvas, extra_inits=(), workers: int = 4) -> dict:
    """BL canvas mc -> FU canvas fc. Returns the final transform (FU point -> BL point) and the stage record."""
    t0 = time.time()
    f8, fm8 = _shrink2(fc)
    m8, mm8 = _shrink2(mc)
    fa, ma = sitk.GetArrayFromImage(f8), sitk.GetArrayFromImage(m8)
    hits = _mncc(np.clip(fa, -1000, 1000), sitk.GetArrayFromImage(fm8) > 0,
                 np.clip(ma, -1000, 1000), sitk.GetArrayFromImage(mm8) > 0, workers=workers)
    center = _centroid_phys(fc)
    cands = []
    for v, u in hits:
        # moving index = fixed index + u  =>  moving point = fixed point + (o_m - o_f) + D u h
        d = np.array(m8.GetOrigin()) - np.array(f8.GetOrigin()) + np.array(f8.GetSpacing()) * u[::-1]
        e = sitk.Euler3DTransform()
        e.SetCenter(center)
        e.SetTranslation(tuple(float(x) for x in d))
        cands.append(("mncc", v, e))
    for t in extra_inits:
        a = sitk.AffineTransform(t)
        e = sitk.Euler3DTransform()
        e.SetCenter(center)
        # rigid part of the affine about the new centre: keep the translation of the centre point
        e.SetTranslation(tuple(np.array(a.TransformPoint(center)) - np.array(center)))
        cands.append(("frame", None, e))
    if not cands:
        raise RuntimeError("no overlap candidate")
    best = None
    tried = []
    for name, v, e in cands:
        try:
            r, _ = _linear(f8, m8, fm8, mm8, e, "rigid", [1], [0.0], iters=150)
        except RuntimeError as err:          # e.g. every sample maps outside the moving field of view
            tried.append({"init": name, "mncc": v, "error": str(err).splitlines()[-1][:200]})
            continue
        s, cov = _score(fc, mc, r)
        tried.append({"init": name, "mncc": v, "ncc": s, "cover": cov,
                      "t": [float(x) for x in r.GetTranslation()]})
        if best is None or s > best[0]:
            best = (s, r, name, e)
    if best is None:
        raise RuntimeError(f"no usable initialisation: {tried}")
    # the same-session frame candidate comes from registering the largest pieces, which is more reliable than a global
    # search on a small or self-similar piece (a legs-only piece matched left leg onto right with equal NCC): prefer it
    # unless it is clearly worse
    frame = [t for t in tried if t["init"] == "frame" and "ncc" in t]
    if frame and best[2] != "frame" and frame[0]["ncc"] >= best[0] - 0.05:
        e = next(c[2] for c in cands if c[0] == "frame")
        r, _ = _linear(f8, m8, fm8, mm8, e, "rigid", [1], [0.0], iters=150)
        best = (frame[0]["ncc"], r, "frame", e)
    rigid, _ = _linear(fc.img, mc.img, fc.mask, mc.mask, best[1], "rigid", [2, 1], [4.0, 0.0])
    affine, _ = _linear(fc.img, mc.img, fc.mask, mc.mask, rigid, "affine", [2, 1], [4.0, 0.0])
    s_rig = _score(fc, mc, rigid)
    s_aff = _score(fc, mc, affine)
    linear = "affine"
    if s_aff[0] < s_rig[0] - 0.02:   # the affine drifted (seen with very different fields of view): keep the rigid
        affine = sitk.AffineTransform(3)
        affine.SetCenter(rigid.GetCenter())
        affine.SetMatrix(rigid.GetMatrix())
        affine.SetTranslation(rigid.GetTranslation())
        s_aff, linear = s_rig, "rigid"
    bs, mi = _bspline(fc.img, mc.img, fc.mask, mc.mask, affine)
    jac = _regularity(bs, fc)
    retried = False
    if jac.get("disp_max_mm", 0.0) < 0.5:   # optimizer stalled at the start (rare line-search failure): reseed once
        bs, mi = _bspline(fc.img, mc.img, fc.mask, mc.mask, affine, seed=SEED + 1)
        jac, retried = _regularity(bs, fc), True
    final = sitk.CompositeTransform([sitk.AffineTransform(affine), bs])
    s_fin = _score(fc, mc, final)
    if s_fin[0] < s_aff[0] - 0.02 or jac["fold"] > 0:   # worse, or folded: keep the linear transform
        final = sitk.CompositeTransform([sitk.AffineTransform(affine)])
        s_fin, kept = s_aff, linear
    else:
        kept = "bspline"
    return {"transform": final, "rigid": rigid, "affine": affine, "init": best[3], "rigid8": best[1],
            "init_name": best[2], "tried": tried, "ncc_rigid": s_rig[0], "ncc_affine": s_aff[0], "ncc": s_fin[0],
            "cover": s_fin[1], "kept": kept, "linear": linear, "bspline_regularity": jac, "bspline_retried": retried,
            "mi": mi,
            "seconds": round(time.time() - t0, 1)}


def _regularity(bs, fc: Canvas) -> dict:
    """Label-free sanity of a B-spline on the FU canvas body: displacement size and Jacobian determinant."""
    df = sitk.TransformToDisplacementField(bs, sitk.sitkVectorFloat64, fc.img.GetSize(), fc.img.GetOrigin(),
                                           fc.img.GetSpacing(), fc.img.GetDirection())
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(df))
    d = np.linalg.norm(sitk.GetArrayFromImage(df), axis=-1)
    b = fc.body_arr
    if not b.any():
        return {"fold": 0.0}
    return {"fold": float(np.mean(jac[b] <= 0)), "jac_min": float(jac[b].min()),
            "jac_p1": float(np.percentile(jac[b], 1)),
            "jac_p99": float(np.percentile(jac[b], 99)), "jac_max": float(jac[b].max()),
            "disp_p95_mm": float(np.percentile(d[b], 95)), "disp_max_mm": float(d[b].max())}


def _field(T, ref: sitk.Image, h: float = FIELD_MM) -> sitk.Transform:
    """Displacement-field approximation of T over ref's extent (sub-0.3 mm for these smooth transforms)."""
    sp, size = np.array(ref.GetSpacing()), np.array(ref.GetSize())
    D = np.array(ref.GetDirection()).reshape(3, 3)
    n = (np.ceil((size - 1) * sp / h) + 3).astype(int)
    origin = np.array(ref.GetOrigin()) - D @ np.full(3, h)
    df = sitk.TransformToDisplacementField(T, sitk.sitkVectorFloat64, [int(v) for v in n], tuple(origin),
                                           [h] * 3, ref.GetDirection())
    return sitk.DisplacementFieldTransform(df)


def _frame_shift(a: Canvas, b: Canvas):
    """Do two pieces of one session share a world frame? Best small shift (voxels, [z, y, x]) of b against a and the
    NCC there, over their overlap; None when they do not overlap."""
    w = sitk.GetArrayFromImage(sitk.Resample(b.img, a.img, sitk.Transform(), sitk.sitkLinear, CLIP_HU[0],
                                             sitk.sitkFloat32))
    v = sitk.GetArrayFromImage(sitk.Resample(b.valid, a.img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
    if (a.body_arr & v).sum() < 500:
        return None
    A = np.clip(a.arr, -1000, 1000)
    B = np.clip(w, -1000, 1000)
    best = None

    def sl(d, n):   # a[i] is compared with b[i - d]
        return (slice(d, n), slice(0, n - d)) if d >= 0 else (slice(0, n + d), slice(-d, n))
    for dz in range(-3, 4):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                (az, bz), (ay, by), (ax, bx) = (sl(d, n) for d, n in zip((dz, dy, dx), A.shape))
                m = a.body_arr[az, ay, ax] & v[bz, by, bx]
                if m.sum() < 500:
                    continue
                Am, Bm = A[az, ay, ax][m], B[bz, by, bx][m]
                p, q = Am - Am.mean(), Bm - Bm.mean()
                s = float((p * q).sum() / np.sqrt((p * p).sum() * (q * q).sum() + 1e-9))
                if best is None or s > best[0]:
                    best = (s, (dz, dy, dx))
    return best


# --------------------------------------------------------------------------------------------- build

def _save(path: str, arr) -> None:
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _is_done(pid: str) -> bool:
    return all(os.path.exists(f"{CACHE}/{k}.meta.npy") for k in pair_ids(pid))


def _ratio(fu_mm3, bl_mm3) -> float:
    if not np.isfinite(bl_mm3) or bl_mm3 <= 0:
        return float("inf") if np.isfinite(fu_mm3) and fu_mm3 > 0 else float("nan")
    return float(fu_mm3 / bl_mm3) if np.isfinite(fu_mm3) else 0.0


def _diam(mm3) -> float:
    return float((6.0 * mm3 / np.pi) ** (1.0 / 3.0)) if np.isfinite(mm3) and mm3 > 0 else float("nan")


def _agreement(fc: Canvas, mc: Canvas, Tq, Ta, n: int = 3000) -> float:
    """Median distance (mm) between where two BL-piece transforms send the FU body points that piece q covers. Pieces
    of one session share a world frame, so a correctly registered secondary piece agrees with the anchor piece."""
    v = sitk.GetArrayFromImage(sitk.Resample(mc.valid, fc.img, Tq, sitk.sitkNearestNeighbor, 0)) > 0
    idx = np.argwhere(fc.body_arr & v)
    if len(idx) < 20:
        return float("inf")
    idx = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
    d = []
    for z, y, x in idx:
        pt = fc.img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        d.append(np.linalg.norm(np.array(Tq.TransformPoint(pt)) - np.array(Ta.TransformPoint(pt))))
    return float(np.median(d))


def _choose_bl(regs: dict) -> list[int]:
    """BL pieces for one FU piece, in compositing order: the piece most similar to the FU (canvas NCC after
    registration) wins wherever it has data, the others fill what is left. A piece is used when it covers >= 2 % of
    the FU body with NCC >= 0.3 and either agrees with the anchor piece's mapping (median <= 15 mm; same session
    frame) or is itself very similar (NCC >= 0.7, e.g. a brain series acquired in its own frame). The anchor is the
    piece with the largest coverage x NCC."""
    cand = [q for q, r in regs.items() if r["ncc"] >= 0.3 and r["cover"] >= 0.02]
    if not cand:
        return [max(regs, key=lambda q: regs[q]["cover"] * max(regs[q]["ncc"], 0.0))]
    anchor = max(cand, key=lambda q: regs[q]["cover"] * regs[q]["ncc"])
    keep = [q for q in cand if q == anchor or regs[q].get("agree_mm", float("inf")) <= 15.0 or regs[q]["ncc"] >= 0.7]
    return sorted(keep, key=lambda q: -regs[q]["ncc"])


def build_patient(pid: str, threads: int = 4, force: bool = False) -> list[dict]:
    """Register every BL piece to every FU piece of a patient and cache all its pairs. Returns the pair metas."""
    if not force and _is_done(pid):
        return [np.load(f"{CACHE}/{k}.meta.npy", allow_pickle=True).item() for k in pair_ids(pid)]
    os.makedirs(CACHE, exist_ok=True)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(threads)
    t0 = time.time()
    D = csv(pid)
    fu_n, bl_n = pieces(pid, "FU"), pieces(pid, "BL")
    fu_raw, fu_img, bl_raw, bl_img = {}, {}, {}, {}
    for n in fu_n:
        fu_raw[n], fu_img[n] = _read(_paths(pid, "FU", n)[0])
    for n in bl_n:
        bl_raw[n], bl_img[n] = _read(_paths(pid, "BL", n)[0])
    fc = {n: _canvas(fu_img[n]) for n in fu_n}
    bc = {n: _canvas(bl_img[n]) for n in bl_n}
    P0 = max(fu_n, key=lambda n: fc[n].body_arr.sum())
    Q0 = max(bl_n, key=lambda n: bc[n].body_arr.sum())
    regs = {(P0, Q0): register(fc[P0], bc[Q0], workers=threads)}
    main_aff = regs[(P0, Q0)]["affine"]
    failed = {}
    for p in fu_n:
        for q in bl_n:
            if (p, q) not in regs:
                try:
                    regs[(p, q)] = register(fc[p], bc[q], extra_inits=[main_aff], workers=threads)
                except RuntimeError as err:   # pieces without common anatomy: leave this BL piece out
                    failed[(p, q)] = str(err).splitlines()[-1][:300]
    # agreement of each BL piece's mapping with the anchor piece's, per FU piece (used to accept secondary pieces)
    for p in fu_n:
        mine = {q: regs[(p, q)] for q in bl_n if (p, q) in regs}
        anchor = max(mine, key=lambda q: mine[q]["cover"] * max(mine[q]["ncc"], 0.0)) if mine else None
        for q, r in mine.items():
            r["agree_mm"] = 0.0 if q == anchor else _agreement(fc[p], bc[q], r["transform"], mine[anchor]["transform"])
    # FU pieces sharing a world frame with each other (for projecting lesions owned by a sibling piece)
    frame = {}
    for p in fu_n:
        for p2 in fu_n:
            if p2 != p:
                frame[(p, p2)] = _frame_shift(fc[p], fc[p2])
    metas = []
    for p in fu_n:
        fails = {q: e for (pp, q), e in failed.items() if pp == p}
        metas.append(_build_pair(pid, p, D, fu_raw, fu_img, bl_raw, bl_img, regs, frame, t0, fails))
    return metas


def _build_pair(pid, p, D, fu_raw, fu_img, bl_raw, bl_img, regs, frame, t0, fails=None) -> dict:
    key = f"{pid}_{p:02d}"
    stem = f"{CACHE}/{key}"
    ref = fu_img[p]
    spacing = _spacing_zyx(ref)
    vox_ml = float(np.prod(spacing)) / 1000.0
    fu = np.clip(sitk.GetArrayFromImage(ref), -1024, 3071).astype(np.int16)
    shape = fu.shape

    # BL pieces registered onto this grid, composited
    mine = {q: r for (pp, q), r in regs.items() if pp == p}
    order = _choose_bl(mine) if mine else []
    bl = np.full(shape, -1024, np.int16)
    blv = np.zeros(shape, bool)
    bllab = np.zeros(shape, np.uint8)
    used = []
    for q in order:
        T = _field(mine[q]["transform"], ref)
        w = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(bl_img[q], sitk.sitkFloat32), ref, T, sitk.sitkLinear,
                                                 -4000.0, sitk.sitkFloat32))
        ok = w > -3000
        fill = ok & ~blv
        bl[fill] = np.clip(np.rint(w[fill]), -1024, 3071).astype(np.int16)
        blv |= ok
        del w
        _, lab_lps = _read(_paths(pid, "BL", q)[1])
        lab = sitk.GetArrayFromImage(sitk.Resample(lab_lps, ref, T, sitk.sitkNearestNeighbor, 0, sitk.sitkInt16))
        put = (lab > 0) & (bllab == 0)
        bllab[put] = lab[put].astype(np.uint8)
        used.append({"piece": q, "cover": mine[q]["cover"], "ncc": mine[q]["ncc"], "fill_frac": float(fill.mean())})
        sitk.WriteTransform(mine[q]["transform"], f"{stem}.bl{q:02d}.tfm")

    for q in pieces(pid, "BL"):          # drop transforms of pieces a previous build used but this one does not
        if q not in order and os.path.exists(f"{stem}.bl{q:02d}.tfm"):
            os.remove(f"{stem}.bl{q:02d}.tfm")

    # FU labels: own piece, then siblings that share the frame
    _, own = _read(_paths(pid, "FU", p)[1])
    fulab = sitk.GetArrayFromImage(own).astype(np.int32)
    assert fulab.max() < 256
    fulab = fulab.astype(np.uint8)
    projected, skipped = [], []
    for p2 in pieces(pid, "FU"):
        if p2 == p:
            continue
        fs = frame.get((p, p2))
        if fs is None:
            continue
        if max(abs(v) for v in fs[1]) > 1 or fs[0] < 0.2:
            skipped.append({"piece": p2, "ncc": fs[0], "shift_vox4mm": list(fs[1])})
            continue
        _, sib = _read(_paths(pid, "FU", p2)[1])
        lab = sitk.GetArrayFromImage(sitk.Resample(sib, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
                                                   sitk.sitkInt16))
        put = (lab > 0) & (fulab == 0)
        fulab[put] = lab[put].astype(np.uint8)
        projected.append({"piece": p2, "ncc": fs[0], "labels": sorted(int(v) for v in np.unique(lab[put]))})

    body = body_mask(fu, spacing)

    # lesion table
    fu_cnt = np.bincount(fulab.ravel(), minlength=256)
    bl_cnt = np.bincount(bllab.ravel(), minlength=256)
    zz, yy, xx = np.nonzero(bllab)
    labs = bllab[zz, yy, xx]
    bl_cz = np.bincount(labs, weights=zz, minlength=256) / np.maximum(bl_cnt, 1)
    bl_cy = np.bincount(labs, weights=yy, minlength=256) / np.maximum(bl_cnt, 1)
    bl_cx = np.bincount(labs, weights=xx, minlength=256) / np.maximum(bl_cnt, 1)
    zz, yy, xx = np.nonzero(fulab)
    labs = fulab[zz, yy, xx]
    fu_cz = np.bincount(labs, weights=zz, minlength=256) / np.maximum(fu_cnt, 1)
    fu_cy = np.bincount(labs, weights=yy, minlength=256) / np.maximum(fu_cnt, 1)
    fu_cx = np.bincount(labs, weights=xx, minlength=256) / np.maximum(fu_cnt, 1)
    cover_cnt = np.bincount(labs, weights=blv[zz, yy, xx], minlength=256)
    in_body = np.bincount(labs, weights=body[zz, yy, xx], minlength=256)
    del zz, yy, xx, labs
    touch = np.zeros(256, bool)    # warped BL labels cut by the FU grid boundary (centroid biased; not QA)
    for face in (bllab[0], bllab[-1], bllab[:, 0], bllab[:, -1], bllab[:, :, 0], bllab[:, :, -1]):
        touch[np.unique(face)] = True
    touch[0] = False
    grp_bl = D[D.topology_class == "MERGING"].groupby("merged_into").volume_bl.sum().to_dict()
    sp = np.array(spacing)
    rows = []
    for r in D.itertuples():
        has_bl, has_fu = isinstance(r.cog_bl, str), isinstance(r.cog_fu, str)
        merging = r.topology_class == "MERGING"
        fl = (int(r.merged_into) if merging else int(r.lesion_id)) if has_fu else 0
        bll = int(r.lesion_id) if has_bl else 0
        fu_piece = int(r.img_id_fu) if pd.notna(r.img_id_fu) else -1
        owned = has_fu and fu_piece == p
        unclear = bool(r.linking_unclear) if isinstance(r.linking_unclear, (bool, np.bool_)) else (
            str(r.linking_unclear).strip().lower() == "true")
        bl_mm3 = float(grp_bl[r.merged_into]) if merging else (float(r.volume_bl) if has_bl else float("nan"))
        fu_mm3 = float(r.volume_fu) if has_fu else float("nan")
        row = {"lesion_id": int(r.lesion_id), "fu_label": fl, "bl_label": bll, "topology_class": r.topology_class,
               "lesion_type": r.lesion_type, "merged_into": int(r.merged_into) if merging else 0,
               "fu_piece": fu_piece, "bl_piece": int(r.img_id_bl) if has_bl else -1, "owned": bool(owned),
               "linking_unclear": unclear, "linking_unclear_raw": None if pd.isna(r.linking_unclear) else unclear,
               "use_for_challenge": bool(r.use_for_challenge), "csv_target_lesion": bool(r.target_lesion),
               "fu_ml": fu_mm3 / 1000.0, "bl_ml": (float(r.volume_bl) / 1000.0) if has_bl else float("nan"),
               "bl_ml_matched": bl_mm3 / 1000.0, "ratio": _ratio(fu_mm3 if has_fu else 0.0, bl_mm3),
               "diam_mm": _diam(fu_mm3), "fu_z": np.nan, "fu_y": np.nan, "fu_x": np.nan,
               "fu_nvox": int(fu_cnt[fl]) if fl else 0, "fu_in_grid": bool(fl and fu_cnt[fl] > 0),
               "bl_cover": float(cover_cnt[fl] / fu_cnt[fl]) if fl and fu_cnt[fl] else float("nan"),
               "in_body": float(in_body[fl] / fu_cnt[fl]) if fl and fu_cnt[fl] else float("nan"),
               "blw_z": np.nan, "blw_y": np.nan, "blw_x": np.nan, "blw_nvox": int(bl_cnt[bll]) if bll else 0,
               "blw_frac": float("nan"), "blw_cut": bool(bll and touch[bll]), "qa_mm": float("nan"),
               "prop_mm": float("nan")}
        if owned:
            z, y, x = _phys_to_zyx(_cog_phys(r.cog_fu, fu_raw[p]), ref)
            row.update(fu_z=float(z), fu_y=float(y), fu_x=float(x))
            row["fu_cog_check_vox"] = float(np.abs(np.array([z, y, x]) - [fu_cz[fl], fu_cy[fl], fu_cx[fl]]).max())
            if isinstance(r.cog_propagated, str):
                pz, py, px = _phys_to_zyx(_cog_phys(r.cog_propagated, fu_raw[p]), ref)
                row["prop_mm"] = float(np.linalg.norm((np.array([pz, py, px]) - [z, y, x]) * sp))
        elif row["fu_in_grid"]:
            row.update(fu_z=float(fu_cz[fl]), fu_y=float(fu_cy[fl]), fu_x=float(fu_cx[fl]))
        if bll and bl_cnt[bll] > 0:
            row.update(blw_z=float(bl_cz[bll]), blw_y=float(bl_cy[bll]), blw_x=float(bl_cx[bll]))
            nat = float(r.volume_bl) / 1000.0
            row["blw_frac"] = float(bl_cnt[bll] * vox_ml / nat) if nat > 0 else float("nan")
        if owned and r.topology_class == "UNCHANGED" and bll and bl_cnt[bll] > 0 and not touch[bll]:
            d = (np.array([row["blw_z"], row["blw_y"], row["blw_x"]]) - [row["fu_z"], row["fu_y"], row["fu_x"]]) * sp
            row["qa_mm"] = float(np.linalg.norm(d))
        # point-based error per registration stage (FU centroid mapped into the BL piece vs BL centroid), QA only
        if owned and r.topology_class == "UNCHANGED" and has_bl and (p, int(r.img_id_bl)) in regs:
            rg = regs[(p, int(r.img_id_bl))]
            xf = _cog_phys(r.cog_fu, fu_raw[p])
            xb = _cog_phys(r.cog_bl, bl_raw[int(r.img_id_bl)])
            for st, nm in (("init", "init"), ("rigid", "rigid"), ("affine", "affine"), ("transform", "final")):
                row[f"pt_{nm}_mm"] = float(np.linalg.norm(np.array(rg[st].TransformPoint(tuple(xf))) - xb))
        is_target = owned and not unclear and (r.topology_class == "NEWLYAPPEARING" or (
            r.topology_class in ("UNCHANGED", "MERGING") and row["ratio"] >= GROWTH))
        if merging and fl != int(r.lesion_id):
            is_target = False   # one row per FU lesion: the group's representative carries the target
        row["target"] = bool(is_target)
        rows.append(row)

    qa = [x["qa_mm"] for x in rows if np.isfinite(x["qa_mm"]) and not x["linking_unclear"]]
    targets = [x for x in rows if x["target"]]
    meta = {
        "version": VERSION, "pair": key, "pid": pid, "piece": p, "split": split_of(pid),
        "shape": shape, "spacing": spacing, "origin": ref.GetOrigin(), "direction": ref.GetDirection(),
        "fu_file": _paths(pid, "FU", p)[0], "bl_used": used, "bl_order": order,
        "registrations": {q: {k: v for k, v in r.items()
                              if k not in ("transform", "rigid", "affine", "init", "rigid8")}
                          for q, r in mine.items()},
        "registration_failed": fails or {}, "projected": projected, "projection_skipped": skipped,
        "body_ml": float(body.sum() * vox_ml), "bl_body_cover": float((blv & body).sum() / max(1, body.sum())),
        "lesions": rows, "targets": targets,
        "qa": {"n": len(qa), "median_mm": float(np.median(qa)) if qa else float("nan"),
               "p90_mm": float(np.percentile(qa, 90)) if qa else float("nan"),
               "flag": bool(qa and np.median(qa) > QA_FLAG_MM)},
        "seconds": round(time.time() - t0, 1),
    }
    _save(stem + ".fu.npy", fu)
    _save(stem + ".bl.npy", bl)
    _save(stem + ".bl_valid.npy", blv.astype(np.uint8))
    _save(stem + ".body.npy", body.astype(np.uint8))
    _save(stem + ".fu_lab.npy", fulab)
    _save(stem + ".bl_lab.npy", bllab)
    _save(stem + ".meta.npy", meta)
    return meta


# --------------------------------------------------------------------------------------------- load

@dataclass
class Pair:
    pair_id: str
    pid: str
    piece: int
    fu: np.ndarray        # int16 HU [z, y, x], LPS (x -> left, y -> posterior, z -> superior), native FU grid
    bl: np.ndarray        # int16 HU, baseline registered onto the FU grid (-1024 where bl_valid == 0)
    bl_valid: np.ndarray  # uint8, 1 where the baseline has data
    body: np.ndarray      # uint8 body mask on the FU grid
    fu_lab: np.ndarray    # uint8 FU lesion labels (value = FU label = lesion_id, merged_into for MERGING groups)
    bl_lab: np.ndarray    # uint8 BL lesion labels warped to the FU grid (nearest neighbour; value = lesion_id)
    spacing: tuple        # (sz, sy, sx) mm
    meta: dict = field(repr=False)

    @property
    def shape(self):
        return self.fu.shape

    @property
    def lesions(self) -> pd.DataFrame:
        return pd.DataFrame(self.meta["lesions"])

    @property
    def targets(self) -> pd.DataFrame:
        return _targets_df(self.meta)

    def zyx_to_mm(self, zyx) -> np.ndarray:
        """Physical LPS point (mm) of a [z, y, x] voxel index on the FU grid."""
        z, y, x = (float(v) for v in zyx)
        D = np.array(self.meta["direction"]).reshape(3, 3)
        sz, sy, sx = self.spacing
        return np.array(self.meta["origin"]) + D @ np.array([x * sx, y * sy, z * sz])


def _meta(key: str) -> dict:
    pid, p = _resolve(key)
    path = f"{CACHE}/{pid}_{p:02d}.meta.npy"
    if not os.path.exists(path):
        build_patient(pid)
    return np.load(path, allow_pickle=True).item()


def load_pair(key: str) -> Pair:
    """Pair id "<pid>_<NN>" (or a bare pid with a single FU piece); builds the patient's cache if missing."""
    meta = _meta(key)
    stem = f"{CACHE}/{meta['pair']}"
    a = {k: np.load(f"{stem}.{k}.npy", mmap_mode="r") for k in _ARRAYS}
    return Pair(meta["pair"], meta["pid"], meta["piece"], a["fu"], a["bl"], a["bl_valid"], a["body"], a["fu_lab"],
                a["bl_lab"], tuple(meta["spacing"]), meta)


def lesions(key: str) -> pd.DataFrame:
    """Per-lesion table of a pair: every CSV lesion of the patient; owned rows belong to this FU piece."""
    return pd.DataFrame(_meta(key)["lesions"])


def _targets_df(meta: dict) -> pd.DataFrame:
    cols = ["lesion_id", "fu_label", "topology_class", "lesion_type", "fu_z", "fu_y", "fu_x", "fu_ml", "bl_ml_matched",
            "ratio", "diam_mm", "fu_nvox", "bl_cover", "use_for_challenge", "csv_target_lesion"]
    return pd.DataFrame(meta["targets"], columns=cols).reset_index(drop=True)


def targets(key: str) -> pd.DataFrame:
    """Target lesions of a pair: NEWLYAPPEARING, or matched with volume_fu / volume_bl >= 1.2, not linking_unclear.
    fu_z/fu_y/fu_x is the FU centroid (voxel index on the FU grid), diam_mm the equivalent diameter, bl_cover the
    fraction of the lesion that the registered baseline covers."""
    return _targets_df(_meta(key))
