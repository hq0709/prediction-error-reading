"""Reader strips for the disease-course world: at each candidate address, the registered baseline (top row) and the
follow-up (bottom row), axial / coronal / sagittal, 60 mm each, soft-tissue window; magenta ticks mark the address.
Both rows are observed views (the baseline resampled onto the follow-up grid); no predicted view is shown.

Labels: "target" inside a selected target (new, or grown to x1.728 volume, >= MIN_DIAM; grown by 3 mm); "lesion" inside
any other lesion (smaller or less-grown targets, stable or shrinking lesions); "none" elsewhere.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .render import plane_image, plane_spacing, sample, window

TILE, GAP, FOV = 200, 4, 60
QUESTION_OUTLINE = ("Two CT scans of the same patient at one marked address: the top row is the baseline (registered "
                    "onto the follow-up), the bottom row the follow-up; axial, coronal and sagittal views, 60 mm each, "
                    "soft-tissue window; magenta ticks mark the address; cyan lines outline the lesions annotated on the "
                    "baseline, drawn on both rows. Is there a lesion at the marked address in the follow-up that is new "
                    "(no cyan outline) or has grown beyond its baseline outline (rather than unchanged anatomy, a stable "
                    "or shrinking lesion, or a registration mismatch)? Answer with one word: yes or no.")
QUESTION = ("Two CT scans of the same patient at one marked address: the top row is the baseline (registered onto "
            "the follow-up), the bottom row the follow-up; axial, coronal and sagittal views, 60 mm each, soft-tissue "
            "window; magenta ticks mark the address. Is there a lesion at the marked address in the follow-up that is "
            "new or has grown since the baseline (rather than unchanged anatomy, a stable or shrinking lesion, or a "
            "registration mismatch)? Answer with one word: yes or no.")


def _tile(vol, spacing, plane, zyx, label, outline=None):
    """One 60 mm view; with outline (a label volume on the same grid), its boundary drawn in cyan."""
    z, y, x = zyx
    Z = vol.shape[0]
    idx = {"axial": z, "coronal": y, "sagittal": x}[plane]
    c, r = {"axial": (x, y), "coronal": (x, Z - 1 - z), "sagittal": (y, Z - 1 - z)}[plane]
    dr, dc, _ = plane_spacing(plane, spacing)
    h = FOV / 2
    i = int(round(min(max(idx, 0), vol.shape[{"axial": 0, "coronal": 1, "sagittal": 2}[plane]] - 1)))
    box = (c - h / dc, r - h / dr, c + h / dc, r + h / dr)
    img = plane_image(vol, plane, i)
    rgb = np.stack([window(sample(img, box, TILE), "soft")] * 3, -1)
    if outline is not None:
        m = sample(plane_image(outline, plane, i), box, TILE, fill=0) > 0
        rgb[m & ~ndi.binary_erosion(m)] = (0, 255, 255)
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(label) + 6, 15], fill=(0, 0, 0))
    d.text((3, 1), label, fill=(255, 255, 0))
    m = TILE // 2
    d.line([(m - 14, m), (m - 6, m)], fill=(255, 0, 255))
    d.line([(m + 6, m), (m + 14, m)], fill=(255, 0, 255))
    return im


GROWTH = 1.728       # a matched lesion counts as grown at x1.728 volume (+20% diameter, as in RECIST)


def selected_targets(t, min_diam: float = 10.0, growth: float = GROWTH):
    """The task's targets among a pair's new-or-growing lesions: >= min_diam, and new or grown by >= growth in volume."""
    if not len(t):
        return t
    grown = (t.topology_class == "NEWLYAPPEARING") | (t.ratio.fillna(np.inf) >= growth)
    return t[(t.diam_mm >= min_diam) & grown]


def label_maps(P, min_diam: float, tol_mm: float = 3.0):
    """(target map, any-lesion map) on the follow-up grid, both grown by tol_mm."""
    t = selected_targets(P.targets, min_diam)
    labs = sorted(int(v) for v in t.fu_label) if len(t) else []
    lab = np.asarray(P.fu_lab)
    grow = lambda m: ndi.distance_transform_edt(~m, sampling=P.spacing) <= tol_mm if m.any() else m
    return grow(np.isin(lab, labs) if labs else np.zeros(lab.shape, bool)), grow(lab > 0)


def course_strips(P, points, min_diam: float = 10.0, prior_outline: bool = False) -> list[dict]:
    """With prior_outline, the baseline's annotated lesions (registered onto the follow-up) are outlined in cyan on both
    rows: an observed prior finding, so the reader sees whether the follow-up extends beyond it or is new."""
    T, A = label_maps(P, min_diam)
    fu, bl = np.asarray(P.fu), np.asarray(P.bl)
    ol = np.asarray(P.bl_lab) if prior_outline else None
    out = []
    for n, (z, y, x, v) in enumerate(points, 1):
        strip = Image.new("RGB", (3 * TILE + 2 * GAP, 2 * TILE + GAP), (40, 40, 40))
        for i, plane in enumerate(("axial", "coronal", "sagittal")):
            strip.paste(_tile(bl, P.spacing, plane, (z, y, x), f"baseline {plane}", ol), (i * (TILE + GAP), 0))
            strip.paste(_tile(fu, P.spacing, plane, (z, y, x), f"follow-up {plane}", ol), (i * (TILE + GAP), TILE + GAP))
        i = tuple(int(round(c)) for c in (z, y, x))
        inside = all(0 <= i[d] < T.shape[d] for d in range(3))
        label = "target" if inside and T[i] else ("lesion" if inside and A[i] else "none")
        out.append({"peak": n, "zyx": [float(z), float(y), float(x)], "wm": float(v), "image": strip, "label": label})
    return out
