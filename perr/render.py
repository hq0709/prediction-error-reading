"""Deterministic rendering of views from the raw volume.

A view is a square window onto one plane of the study: plane, slice index (or a full projection),
slab, CT window and a region in plane coordinates. Every rendered image keeps its geometry, so any
point in it maps back to voxel (and millimetre) coordinates. Enlargement is nearest-neighbour, as in
VISTA's inspect: no smoothing, no invented detail.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

WINDOWS = {  # (width, level) in HU
    "lung": (1500.0, -600.0),
    "mediastinum": (400.0, 40.0),
    "bone": (1800.0, 400.0),
    "soft": (350.0, 50.0),
}
BACKGROUND_HU = -1024


@dataclass
class Geom:
    plane: str                       # axial | coronal | sagittal
    index: int | None                # slice along the normal; None = projection through the whole volume
    slab: int                        # half-thickness in voxels for a slab MIP (0 = single slice)
    window: str
    box: tuple[float, float, float, float]   # square region (c0, r0, c1, r1) in plane coordinates
    size: int                        # output is size x size pixels
    masked: bool = False             # projection restricted to the lung mask
    extra: dict = field(default_factory=dict)


def plane_spacing(plane: str, spacing) -> tuple[float, float, float]:
    """(row spacing, column spacing, normal spacing) in mm."""
    sz, sy, sx = spacing
    return {"axial": (sy, sx, sz), "coronal": (sz, sx, sy), "sagittal": (sz, sy, sx)}[plane]


def plane_shape(plane: str, shape) -> tuple[int, int, int]:
    """(rows, cols, depth along the normal)."""
    Z, Y, X = shape
    return {"axial": (Y, X, Z), "coronal": (Z, X, Y), "sagittal": (Z, Y, X)}[plane]


def plane_image(vol: np.ndarray, plane: str, index: int | None, slab: int = 0,
                mask: np.ndarray | None = None) -> np.ndarray:
    """2-D plane image P[row, col]; coronal/sagittal rows run superior -> inferior."""
    v = vol if mask is None else np.where(mask, vol, BACKGROUND_HU)
    axis = {"axial": 0, "coronal": 1, "sagittal": 2}[plane]
    if index is None:
        img = v.max(axis=axis)
    else:
        depth = v.shape[axis]
        lo, hi = max(0, index - slab), min(depth, index + slab + 1)
        sl = [slice(None)] * 3
        sl[axis] = slice(lo, hi)
        img = v[tuple(sl)].max(axis=axis) if hi - lo > 1 else np.take(v, min(max(index, 0), depth - 1), axis=axis)
    if plane != "axial":
        img = img[::-1]
    return np.asarray(img)


def square_box(box, plane: str, spacing) -> tuple[float, float, float, float]:
    """Expand a (c0, r0, c1, r1) box to a square in millimetres around its centre."""
    dr, dc, _ = plane_spacing(plane, spacing)
    c0, r0, c1, r1 = box
    cc, rc = (c0 + c1) / 2, (r0 + r1) / 2
    half = max((c1 - c0) * dc, (r1 - r0) * dr, 4.0) / 2
    return (cc - half / dc, rc - half / dr, cc + half / dc, rc + half / dr)


def sample(img: np.ndarray, box, size: int, fill=BACKGROUND_HU) -> np.ndarray:
    """Nearest-neighbour sampling of img over box onto a size x size grid."""
    c0, r0, c1, r1 = box
    u = (np.arange(size) + 0.5) / size
    cols = np.floor(c0 + u * (c1 - c0)).astype(int)
    rows = np.floor(r0 + u * (r1 - r0)).astype(int)
    R, C = img.shape
    ok_c = (cols >= 0) & (cols < C)
    ok_r = (rows >= 0) & (rows < R)
    out = np.full((size, size), fill, dtype=np.float32)
    sub = img[np.clip(rows, 0, R - 1)][:, np.clip(cols, 0, C - 1)].astype(np.float32)
    keep = ok_r[:, None] & ok_c[None, :]
    out[keep] = sub[keep]
    return out


def window(hu: np.ndarray, name: str) -> np.ndarray:
    w, l = WINDOWS[name]
    return (np.clip((hu - (l - w / 2)) / w, 0, 1) * 255).astype(np.uint8)


def hot(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return (np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], -1) * 255).astype(np.uint8)


def overlay(gray: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Hot-coloured t over a grayscale image; t <= 0.05 leaves the CT visible."""
    g = np.repeat(gray[..., None], 3, -1).astype(np.float32)
    a = np.clip((t - 0.05) * 1.5, 0, 0.85)[..., None]
    return (g * (1 - a) + hot(t).astype(np.float32) * a).astype(np.uint8)


def png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def montage_tile(g: Geom, u: float, v: float) -> tuple[int, float, float]:
    """For a montage view: (slice index, u, v) in the coordinates of a full-size view of that tile."""
    grid = g.extra["grid"]
    tile = g.size / grid
    j = int(np.clip(u // tile, 0, grid - 1))
    i = int(np.clip(v // tile, 0, grid - 1))
    k = g.extra["slices"][i * grid + j]
    return k, (u - j * tile) / tile * g.size, (v - i * tile) / tile * g.size


def to_voxel(g: Geom, u: float, v: float, shape) -> tuple[float | None, float | None, float | None]:
    """Display pixel (u, v) -> continuous voxel (z, y, x); the projected axis is None."""
    Z = shape[0]
    if g.extra.get("montage"):
        k, u, v = montage_tile(g, u, v)
        g = Geom(g.plane, k, 0, g.window, g.box, g.size)
    c0, r0, c1, r1 = g.box
    c = c0 + (u / g.size) * (c1 - c0)
    r = r0 + (v / g.size) * (r1 - r0)
    n = None if g.index is None else float(g.index)
    if g.plane == "axial":
        return (n, r, c)
    if g.plane == "coronal":
        return (Z - 1 - r, n, c)
    return (Z - 1 - r, c, n)


def from_voxel(g: Geom, zyx, shape) -> tuple[float, float]:
    """Voxel (z, y, x) -> display pixel (u, v) in view g (the normal coordinate is ignored)."""
    Z = shape[0]
    z, y, x = zyx
    c, r = {"axial": (x, y), "coronal": (x, Z - 1 - z), "sagittal": (y, Z - 1 - z)}[g.plane]
    c0, r0, c1, r1 = g.box
    return ((c - c0) / (c1 - c0) * g.size, (r - r0) / (r1 - r0) * g.size)
