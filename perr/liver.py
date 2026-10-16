"""Second anatomy world: liver tumours in portal-venous CT (MSD Task03 / LiTS, CC-BY-SA 4.0).

The world model says liver parenchyma is homogeneous at the scale of a lesion: the expected value of a liver
voxel is the masked local mean of the liver around it (normalised convolution, sigma 15 mm). The prediction error
is the hypodense deficit, expected minus observed (observed smoothed at 1.5 mm), inside the liver eroded by 3 mm.
Its peaks are the candidate addresses. Targets are tumour components of at least 10 mm equivalent diameter; a
candidate hits one when it lies inside that component's mask grown by 3 mm.
"""

from __future__ import annotations

import io
import os

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .render import plane_image, plane_spacing, png, sample, window
from .study import Nodule, Study

ROOT = "/raid/home/CAMCA/hj880/msd/Task03_Liver"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/liver"
MIN_DIAM, TOL_MM = 10.0, 3.0
TILE, GAP, FOV = 200, 4, 60
QUESTION = ("One candidate from a contrast-enhanced abdominal CT: axial, coronal and sagittal views, 60 mm each, "
            "soft-tissue window; magenta ticks mark the exact address. Is there a focal liver lesion (a tumour at "
            "least 10 mm across, distinct from the surrounding parenchyma) at the marked address, rather than normal "
            "liver, a vessel, the liver edge or a structure outside the liver? Answer with one word: yes or no.")


def cases() -> list[str]:
    return sorted((f[:-7] for f in os.listdir(f"{ROOT}/labelsTr") if f.endswith(".nii.gz") and not f.startswith(".")),
                  key=lambda c: int(c.split("_")[1]))


def load_liver(case: str) -> Study:
    """Study with lung := liver mask (the domain) and nodules := tumour components; also caches label masks."""
    os.makedirs(CACHE, exist_ok=True)
    stem = f"{CACHE}/{case}"
    if not os.path.exists(stem + ".meta.npy"):
        img = sitk.DICOMOrient(sitk.ReadImage(f"{ROOT}/imagesTr/{case}.nii.gz"), "LPS")
        lab = sitk.DICOMOrient(sitk.ReadImage(f"{ROOT}/labelsTr/{case}.nii.gz"), "LPS")
        vol = np.clip(sitk.GetArrayFromImage(img), -1024, 3071).astype(np.int16)
        lab = sitk.GetArrayFromImage(lab).astype(np.uint8)
        sx, sy, sz = img.GetSpacing()
        spacing = (sz, sy, sx)
        tum = lab == 2
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
        np.save(stem + ".vol.npy", vol)
        np.save(stem + ".liver.npy", (lab > 0).astype(np.uint8))
        np.save(stem + ".big.npy", ndi.binary_dilation(big, ball).astype(np.uint8))
        np.save(stem + ".small.npy", ndi.binary_dilation(small, ball).astype(np.uint8))
        np.save(stem + ".meta.npy", {"spacing": spacing, "nodules": nods})
    meta = np.load(stem + ".meta.npy", allow_pickle=True).item()
    s = Study(case, np.load(stem + ".vol.npy", mmap_mode="r"), tuple(meta["spacing"]),
              np.load(stem + ".liver.npy", mmap_mode="r"), "", [Nodule(tuple(z), d) for z, d in meta["nodules"]])
    return s


def label_at(case: str, zyx) -> str:
    z, y, x = (int(round(c)) for c in zyx)
    big = np.load(f"{CACHE}/{case}.big.npy", mmap_mode="r")
    if big[z, y, x]:
        return "big"
    return "small" if np.load(f"{CACHE}/{case}.small.npy", mmap_mode="r")[z, y, x] else "none"


def has_target(s: Study) -> bool:
    return any(n.diameter_mm >= MIN_DIAM for n in s.nodules)


class LiverPrior:
    name = "liver"
    tag = "liverprior_s15_e3"

    def error_volume(self, s: Study):
        path = f"{CACHE}/{s.case_id}.{self.tag}.R.npy"
        if os.path.exists(path):
            return np.load(path, mmap_mode="r")
        M = np.asarray(s.lung) > 0
        zz, yy, xx = (np.nonzero(M.any(axis=ax))[0] for ax in ((1, 2), (0, 2), (0, 1)))
        pad = [int(np.ceil(30 / sp)) for sp in s.spacing]
        sl = tuple(slice(max(0, a.min() - p), b.max() + p + 1) for a, b, p in zip((zz, yy, xx), (zz, yy, xx), pad))
        V = np.clip(np.asarray(s.vol)[sl].astype(np.float32), -200, 300)
        m = M[sl].astype(np.float32)
        sp = np.array(s.spacing, float)
        E = ndi.gaussian_filter(V * m, 15 / sp) / np.maximum(ndi.gaussian_filter(m, 15 / sp), 1e-3)
        Vs = ndi.gaussian_filter(V, 1.5 / sp)
        r = [max(1, int(round(3 / x))) for x in sp]
        core = ndi.binary_erosion(M[sl], structure=np.ones([2 * k + 1 for k in r]))
        R = np.zeros(s.vol.shape, np.float16)
        R[sl] = np.where(core, np.clip(E - Vs, 0, None), 0).astype(np.float16)
        np.save(path, R)
        return np.load(path, mmap_mode="r")

    def top(self, s: Study, k: int = 10, nms_mm: float = 15.0):
        W = np.asarray(self.error_volume(s)).astype(np.float32)
        r = [max(1, int(np.ceil(nms_mm / x))) for x in s.spacing]
        out = []
        for _ in range(k):
            i = int(np.argmax(W))
            z, y, x = np.unravel_index(i, W.shape)
            v = float(W[z, y, x])
            if v <= 0:
                break
            out.append((float(z), float(y), float(x), v))
            W[max(0, z - r[0]):z + r[0] + 1, max(0, y - r[1]):y + r[1] + 1, max(0, x - r[2]):x + r[2] + 1] = 0
        return out


def _tile(s: Study, plane: str, zyx, label: str) -> Image.Image:
    z, y, x = zyx
    Z = s.shape[0]
    idx = {"axial": z, "coronal": y, "sagittal": x}[plane]
    c, r = {"axial": (x, y), "coronal": (x, Z - 1 - z), "sagittal": (y, Z - 1 - z)}[plane]
    dr, dc, _ = plane_spacing(plane, s.spacing)
    h = FOV / 2
    img = plane_image(np.asarray(s.vol), plane, int(round(idx)))
    im = Image.fromarray(window(sample(img, (c - h / dc, r - h / dr, c + h / dc, r + h / dr), TILE), "soft")).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(label) + 6, 15], fill=(0, 0, 0))
    d.text((3, 1), label, fill=(255, 255, 0))
    m = TILE // 2
    d.line([(m - 14, m), (m - 6, m)], fill=(255, 0, 255))
    d.line([(m + 6, m), (m + 14, m)], fill=(255, 0, 255))
    return im


def liver_strips(s: Study, points) -> list[dict]:
    out = []
    for n, (z, y, x, v) in enumerate(points, 1):
        strip = Image.new("RGB", (3 * TILE + 2 * GAP, TILE), (40, 40, 40))
        for i, plane in enumerate(("axial", "coronal", "sagittal")):
            strip.paste(_tile(s, plane, (z, y, x), plane), (i * (TILE + GAP), 0))
        out.append({"peak": n, "zyx": [z, y, x], "wm": v, "image": strip, "label": label_at(s.case_id, (z, y, x))})
    return out
