"""Mechanical world models: they predict what the study should show, with no training.

continuity3d: lung anatomy is made of structures that continue (vessels, airways, fissures). A voxel is
predicted from pairs of neighbours at distance d on opposite sides, along 13 directions and several
scales; the prediction is the best continuation (both neighbours must agree, so a structure has to
continue on both sides). Vessels and walls are predicted; a compact structure that continues in no
direction is not, and shows up as a positive prediction error. Intensities are capped before the
prediction so that very dense matter (calcium, contrast, bone) does not dominate the error.

The world model caches the full expected volume E and the error volume R = max(V - E, 0) inside the
lung domain; predicted views are planes of E, so they are addressed exactly like observed views.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

from .render import plane_image

CACHE = "/raid/home/CAMCA/hj880/perr/cache/luna"
DIRS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]


def lung_domain(study) -> np.ndarray:
    """Lung field with internal holes filled and small pleural concavities closed (nodules included)."""
    path = f"{CACHE}/{study.case_id}.domain.npy"
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    m = ndimage.binary_closing(np.asarray(study.lung) > 0, structure=np.ones((3, 7, 7)))
    m = np.stack([ndimage.binary_fill_holes(sl) for sl in m]).astype(np.uint8)
    np.save(path, m)
    return m


def lung_mask_dilated(study) -> np.ndarray:
    return lung_domain(study)


class Continuity3D:
    name = "continuity"

    def __init__(self, scales=(5.0, 10.0, 15.0), cap_hu: float = 100.0, mode: str = "min"):
        self.scales, self.cap, self.mode = tuple(scales), cap_hu, mode
        self.tag = f"c3d_{mode}_{'-'.join(str(int(s)) for s in self.scales)}_cap{int(cap_hu)}"

    def _paths(self, study):
        stem = f"{CACHE}/{study.case_id}.{self.tag}"
        return stem + ".E.npy", stem + ".R.npy", stem + ".peaks.npy"

    def volumes(self, study):
        pe, pr, _ = self._paths(study)
        if os.path.exists(pe) and os.path.exists(pr):
            return np.load(pe, mmap_mode="r"), np.load(pr, mmap_mode="r")
        dom = np.asarray(lung_domain(study)) > 0
        zz = np.nonzero(dom.any(axis=(1, 2)))[0]
        yy = np.nonzero(dom.any(axis=(0, 2)))[0]
        xx = np.nonzero(dom.any(axis=(0, 1)))[0]
        sl = (slice(max(0, zz.min() - 2), zz.max() + 3), slice(max(0, yy.min() - 2), yy.max() + 3),
              slice(max(0, xx.min() - 2), xx.max() + 3))
        V = np.clip(np.asarray(study.vol)[sl].astype(np.float32), -1000, self.cap)
        sp = np.array(study.spacing, float)
        pad = int(np.ceil(max(self.scales) / sp.min())) + 1
        P = np.pad(V, pad, mode="edge")
        E = np.full(V.shape, -1e9, np.float32)
        n0, n1, n2 = V.shape
        for d in self.scales:
            for e in DIRS:
                ph = np.array(e, float) * sp
                ph /= np.linalg.norm(ph)
                o = np.round(d * ph / sp).astype(int)
                a = P[pad - o[0]:pad - o[0] + n0, pad - o[1]:pad - o[1] + n1, pad - o[2]:pad - o[2] + n2]
                b = P[pad + o[0]:pad + o[0] + n0, pad + o[1]:pad + o[1] + n1, pad + o[2]:pad + o[2] + n2]
                np.maximum(E, np.minimum(a, b) if self.mode == "min" else (a + b) / 2, out=E)
        R = np.where(dom[sl], np.clip(V - E, 0, None), 0).astype(np.float16)
        Efull = np.clip(np.asarray(study.vol).astype(np.float32), -1000, None)
        Efull[sl] = np.where(dom[sl], np.minimum(E, V), Efull[sl])     # outside the lungs: expected = observed
        Rfull = np.zeros(study.vol.shape, np.float16)
        Rfull[sl] = R
        np.save(pe, Efull.astype(np.int16))
        np.save(pr, Rfull)
        return np.load(pe, mmap_mode="r"), np.load(pr, mmap_mode="r")

    def peaks(self, study, k: int = 10, nms_mm: float = 10.0):
        _, _, pp = self._paths(study)
        if os.path.exists(pp):
            return [tuple(p) for p in np.load(pp).tolist()]
        _, R = self.volumes(study)
        W = np.asarray(R).astype(np.float32)
        r = [max(1, int(np.ceil(nms_mm / s))) for s in study.spacing]
        out = []
        for _ in range(k):
            i = int(np.argmax(W))
            z, y, x = np.unravel_index(i, W.shape)
            v = float(W[z, y, x])
            if v <= 0:
                break
            out.append((float(z), float(y), float(x), v))
            W[max(0, z - r[0]):z + r[0] + 1, max(0, y - r[1]):y + r[1] + 1, max(0, x - r[2]):x + r[2] + 1] = 0
        np.save(pp, np.array(out, np.float32).reshape(-1, 4))
        return out

    def expected(self, study, plane: str, index: int) -> np.ndarray:
        E, _ = self.volumes(study)
        return plane_image(E, plane, index).astype(np.float32)

    def slab_errors(self, study, slices: list[int]) -> list[np.ndarray]:
        """Largest error inside the slab each montage slice stands for (half-way to its neighbours)."""
        _, R = self.volumes(study)
        Z = R.shape[0]
        bounds = [0] + [int((a + b) / 2) for a, b in zip(slices, slices[1:])] + [Z - 1]
        out = []
        for i in range(len(slices)):
            lo, hi = bounds[i], max(bounds[i] + 1, bounds[i + 1])
            out.append(np.asarray(R[lo:hi]).astype(np.float32).max(axis=0))
        return out


class ShapePrior:
    """Lung anatomy is locally a tube (vessels, airways) or a sheet (fissures, pleura). The prediction
    error at a voxel is how far its local shape departs from both: the Hessian blob measure of Li et al.
    (2003), the largest over scales. The expected view is the CT with structures the prior cannot explain
    replaced by aerated lung, feathered at the edges."""

    name = "shape"

    def __init__(self, sigmas_mm=(1.5, 2.5, 4.0, 6.0), cap_hu: float = 200.0, lung_hu: float = -850.0):
        self.sigmas, self.cap, self.lung_hu = tuple(sigmas_mm), cap_hu, lung_hu
        self.tag = "shape_" + "-".join(f"{s:g}" for s in self.sigmas) + f"_cap{int(cap_hu)}"

    def _paths(self, study):
        stem = f"{CACHE}/{study.case_id}.{self.tag}"
        return stem + ".R.npy", stem + ".peaks.npy"

    def error_volume(self, study):
        pr, _ = self._paths(study)
        if os.path.exists(pr):
            return np.load(pr, mmap_mode="r")
        dom = np.asarray(lung_domain(study)) > 0
        m = int(np.ceil(3 * max(self.sigmas) / min(study.spacing))) + 1
        zz = np.nonzero(dom.any(axis=(1, 2)))[0]
        yy = np.nonzero(dom.any(axis=(0, 2)))[0]
        xx = np.nonzero(dom.any(axis=(0, 1)))[0]
        sl = (slice(max(0, zz.min() - m), zz.max() + m + 1), slice(max(0, yy.min() - m), yy.max() + m + 1),
              slice(max(0, xx.min() - m), xx.max() + m + 1))
        V = np.clip(np.asarray(study.vol)[sl].astype(np.float32), -1000, self.cap)
        D = dom[sl]
        idx = np.nonzero(D)
        sp = np.array(study.spacing, float)
        best = np.zeros(len(idx[0]), np.float32)
        for s in self.sigmas:
            H = {}
            for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
                order = [0, 0, 0]
                order[i] += 1
                order[j] += 1
                H[(i, j)] = ndimage.gaussian_filter(V, s / sp, order=order)[idx] * (s * s) / (sp[i] * sp[j])
            M = np.empty((len(idx[0]), 3, 3), np.float32)
            for (i, j), v in H.items():
                M[:, i, j] = v
                M[:, j, i] = v
            lam = np.linalg.eigvalsh(M)
            l1, l3 = lam[:, 0], lam[:, 2]
            dot = np.where(lam[:, 2] < 0, (l3 ** 2) / np.maximum(-l1, 1e-3), 0)
            np.maximum(best, dot.astype(np.float32), out=best)
        R = np.zeros(study.vol.shape, np.float16)
        tmp = np.zeros(D.shape, np.float32)
        tmp[idx] = best
        R[sl] = tmp
        np.save(pr, R)
        return np.load(pr, mmap_mode="r")

    def peaks(self, study, k: int = 10, nms_mm: float = 10.0):
        _, pp = self._paths(study)
        if os.path.exists(pp):
            return [tuple(p) for p in np.load(pp).tolist()]
        W = np.asarray(self.error_volume(study)).astype(np.float32)
        r = [max(1, int(np.ceil(nms_mm / s))) for s in study.spacing]
        out = []
        for _ in range(k):
            i = int(np.argmax(W))
            z, y, x = np.unravel_index(i, W.shape)
            v = float(W[z, y, x])
            if v <= 0:
                break
            out.append((float(z), float(y), float(x), v))
            W[max(0, z - r[0]):z + r[0] + 1, max(0, y - r[1]):y + r[1] + 1, max(0, x - r[2]):x + r[2] + 1] = 0
        np.save(pp, np.array(out, np.float32).reshape(-1, 4))
        return out

    def scale(self, study) -> float:
        """Error value shown as full intensity: the largest peak of this study."""
        pk = self.peaks(study)
        return pk[0][3] if pk else 1.0

    def error_plane(self, study, plane: str, index: int) -> np.ndarray:
        return plane_image(np.asarray(self.error_volume(study)), plane, index).astype(np.float32)

    def expected(self, study, plane: str, index: int) -> np.ndarray:
        obs = plane_image(np.asarray(study.vol), plane, index).astype(np.float32)
        t = np.clip(self.error_plane(study, plane, index) / (0.25 * self.scale(study)), 0, 1)
        w = np.clip(ndimage.gaussian_filter(t, 1.5) * 1.5, 0, 1)
        return obs * (1 - w) + self.lung_hu * w

    def slab_errors(self, study, slices: list[int]) -> list[np.ndarray]:
        R = self.error_volume(study)
        Z = R.shape[0]
        bounds = [0] + [int((a + b) / 2) for a, b in zip(slices, slices[1:])] + [Z - 1]
        return [np.asarray(R[bounds[i]:max(bounds[i] + 1, bounds[i + 1])]).astype(np.float32).max(axis=0)
                for i in range(len(slices))]
