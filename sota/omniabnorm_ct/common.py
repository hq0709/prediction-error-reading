"""Cached-array I/O and the fixed mask -> answer rule for OmniAbnorm-CT on our benchmark.

Cached arrays (cache/<ds>/<case>.vol.npy) are int16 HU indexed [z, y, x] in LPS order: x -> patient left,
y -> posterior, z -> superior. An axial slice vol[z] shown as an image (rows = y, columns = x) is in radiological
display convention (anterior at the top, patient's right on the image's left), the convention of the Radiopaedia
images OmniAbnorm-CT was trained on. Every 512 x 512 slice mask the model returns is mapped back to that slice with
the official `recover_single_slice` and written into a 3D mask on the cached grid, so a mask voxel index *is* the
cached [z, y, x] index (no resampling, no affine).

Mask -> answer rule (fixed before any evaluation run; the same rule as sota/sat_pro and sota/vista3d):
  * split the stacked binary 3D mask into connected components, 26-connectivity;
  * point = component centroid (mean voxel index) = cached [z, y, x] index;
  * size = equivalent-sphere diameter (6 V / pi)^(1/3), V = voxel count x sz * sy * sx (mm^3);
  * score = mean model confidence over the component = mean sigmoid probability of the segmentation head;
  * answer = the component with the highest score (ties: larger component first);
  * candidates = the top 10 components by score (candidates[0] is the answer);
  * empty mask -> present false, zyx null.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage as ndi

CACHE = "/raid/home/CAMCA/hj880/perr/cache"
TOPK = 10


def _numpy2_pickle_compat():
    """meta.npy files were pickled under numpy 2 (module numpy._core); let numpy 1.24 unpickle them."""
    if "numpy._core" not in sys.modules:
        import numpy.core as core
        import numpy.core.multiarray as multiarray

        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.multiarray", multiarray)


def load_spacing(ds: str, case: str):
    """Spacing (sz, sy, sx) only. The meta file also holds annotations; they are never read here."""
    _numpy2_pickle_compat()
    meta = np.load(f"{CACHE}/{ds}/{case}.meta.npy", allow_pickle=True).item()
    return tuple(float(s) for s in meta["spacing"])


def load_cached(ds: str, case: str):
    """Volume (int16 HU, [z, y, x]) and spacing only."""
    return np.load(f"{CACHE}/{ds}/{case}.vol.npy"), load_spacing(ds, case)


def add_sphere(vol: np.ndarray, spacing_zyx, center_zyx, diameter_mm: float = 10.0, hu: float = 300.0) -> np.ndarray:
    """README rule 3: a sphere of `diameter_mm` set to +`hu` HU at a known (continuous) cached index."""
    zz, yy, xx = np.ogrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    sz, sy, sx = spacing_zyx
    cz, cy, cx = center_zyx
    d2 = ((zz - cz) * sz) ** 2 + ((yy - cy) * sy) ** 2 + ((xx - cx) * sx) ** 2
    out = vol.copy()
    out[d2 <= (diameter_mm / 2.0) ** 2] = hu
    return out


def mask_to_answer(mask: np.ndarray, prob: np.ndarray, spacing_zyx, topk: int = TOPK) -> dict:
    """Apply the fixed rule to a 3D mask / probability map that live on the cached grid."""
    voxel_mm3 = float(np.prod(spacing_zyx))
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
        zs = pts[:, 0] + sl[0].start
        key_z = int(np.bincount(zs - zs.min()).argmax() + zs.min())  # slice with the largest in-plane area
        comps.append((score, cnt, cen, key_z, (int(zs.min()), int(zs.max()))))
    comps.sort(key=lambda t: (-t[0], -t[1]))
    top = comps[:topk]
    cands = [{"zyx": [round(float(v), 2) for v in c[2]],
              "diameter_mm": round(float((6 * c[1] * voxel_mm3 / np.pi) ** (1 / 3)), 2),
              "score": round(c[0], 4), "n_vox": int(c[1]), "key_z": c[3], "z_range": list(c[4])} for c in top]
    best = cands[0]
    return {"present": True, "zyx": best["zyx"], "diameter_mm": best["diameter_mm"], "score": best["score"],
            "candidates": cands, "n_components": len(comps)}
