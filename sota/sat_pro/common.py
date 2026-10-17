"""Shared I/O and the fixed mask -> answer rule for the 3D segmentation baselines (SAT-Pro, VISTA3D).

The same file is copied verbatim into sota/sat_pro/ and sota/vista3d/ so that each method directory is self-contained.

Cached arrays (cache/<ds>/<case>.vol.npy) are int16 HU indexed [z, y, x] in LPS order: x -> patient left,
y -> posterior, z -> superior. `write_nifti` stores such an array as NIfTI with data axes (i, j, k) = (x, y, z) and
the explicit RAS+ affine diag(-sx, -sy, sz): i grows toward the patient's left (RAS x decreases), j toward
posterior (RAS y decreases), k toward superior (RAS z increases). Origin is 0; it does not enter any of the
models' preprocessing.

Mask -> answer rule (fixed before any evaluation run):
  * split the binary mask into 3D connected components, 26-connectivity;
  * point = component centroid (mean voxel index), mapped into the cached-array index space [z, y, x];
  * size = equivalent-sphere diameter (6 V / pi)^(1/3), V = voxel count x voxel volume of the mask grid (mm^3);
  * score = mean predicted probability over the component's voxels;
  * answer = the component with the highest score (ties: larger component first);
  * candidates = the top 10 components by score (candidates[0] is the answer);
  * empty mask -> present false, zyx null.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

CACHE = "/raid/home/CAMCA/hj880/perr/cache"
TOPK = 10


def _numpy2_pickle_compat():
    """meta.npy files were pickled under numpy 2 (module numpy._core); let numpy 1.x unpickle them."""
    import sys

    if not hasattr(np, "_core"):
        import numpy.core as core
        import numpy.core.multiarray as multiarray

        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.multiarray", multiarray)


def load_spacing(ds: str, case: str):
    """Spacing (sz, sy, sx) only. The meta file also holds annotations; they are never used here."""
    _numpy2_pickle_compat()
    meta = np.load(f"{CACHE}/{ds}/{case}.meta.npy", allow_pickle=True).item()
    return tuple(float(s) for s in meta["spacing"])


def load_cached(ds: str, case: str):
    """Volume and spacing only."""
    return np.load(f"{CACHE}/{ds}/{case}.vol.npy"), load_spacing(ds, case)


def nifti_affine(spacing_zyx) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    return np.array([[-sx, 0, 0, 0], [0, -sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]], dtype=np.float64)


def write_nifti(vol_zyx: np.ndarray, spacing_zyx, path: str) -> np.ndarray:
    import nibabel as nib

    aff = nifti_affine(spacing_zyx)
    img = nib.Nifti1Image(np.ascontiguousarray(np.asarray(vol_zyx).transpose(2, 1, 0)), aff)
    img.set_qform(aff, code=1)
    img.set_sform(aff, code=1)
    img.header.set_xyzt_units("mm")
    nib.save(img, path)
    return aff


def nifti_ijk_to_cache_zyx(ijk: np.ndarray) -> np.ndarray:
    """(N, 3) NIfTI voxel (i, j, k) = (x, y, z) -> cached index (z, y, x)."""
    return np.asarray(ijk, dtype=np.float64)[:, ::-1]


def affine_mapper(grid_affine: np.ndarray, spacing_zyx):
    """Map (N, 3) voxel indices of a grid with RAS affine `grid_affine` to cached [z, y, x] indices."""
    M = np.linalg.inv(nifti_affine(spacing_zyx)) @ np.asarray(grid_affine, dtype=np.float64)

    def f(idx):
        idx = np.asarray(idx, dtype=np.float64)
        ijk = idx @ M[:3, :3].T + M[:3, 3]
        return nifti_ijk_to_cache_zyx(ijk)

    return f


def mask_to_answer(mask: np.ndarray, prob: np.ndarray, voxel_mm3: float, to_cache, cache_shape, topk: int = TOPK) -> dict:
    """Apply the fixed rule. `mask`/`prob` live on the same grid; `to_cache` maps (N, 3) grid indices to [z, y, x]."""
    lab, n = ndi.label(np.asarray(mask, bool), structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return {"present": False, "zyx": None, "diameter_mm": None, "score": None, "candidates": [], "n_components": 0}
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        pts = np.argwhere(m)
        cnt = len(pts)
        cen = pts.mean(0) + np.array([s.start for s in sl], dtype=np.float64)
        score = float(np.asarray(prob[sl], dtype=np.float64)[m].mean())
        comps.append((score, cnt, cen))
    comps.sort(key=lambda t: (-t[0], -t[1]))
    top = comps[:topk]
    zyx = to_cache(np.stack([c[2] for c in top]))
    hi = np.array(cache_shape, dtype=np.float64) - 1
    clipped = bool(np.any(zyx < 0) or np.any(zyx > hi))
    zyx = np.clip(zyx, 0, hi)
    cands = [{"zyx": [round(float(v), 2) for v in p], "diameter_mm": round(float((6 * c[1] * voxel_mm3 / np.pi) ** (1 / 3)), 2),
              "score": round(c[0], 4), "n_vox": int(c[1])} for c, p in zip(top, zyx)]
    best = cands[0]
    return {"present": True, "zyx": best["zyx"], "diameter_mm": best["diameter_mm"], "score": best["score"],
            "candidates": cands, "n_components": len(comps), "clipped": clipped}
