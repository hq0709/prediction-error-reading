"""Candidate strips for the reader: one row of three 40 mm views per world-model peak, rendered by the harness's
own measure tool, so that training, probing and the agent all show the reader the same picture.

The rank label the contact sheet prints ("peak 3 axial") is replaced by the plane name: the reader judges the
picture, the world model supplies the rank, and the two are combined only in the fusion rule.
"""

from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image, ImageDraw

from .harness import Session
from .score import MIN_DIAM, TOL_MM, _mm
from .worldmodels import ShapePrior

TILE, GAP = 200, 4
MEAS_KEYS = ("equivalent_diameter_mm", "long_axis_mm_on_slice", "mean_HU", "warning")


def unlabel(strip: Image.Image) -> Image.Image:
    strip = strip.copy()
    d = ImageDraw.Draw(strip)
    for i, name in enumerate(("axial", "coronal", "sagittal")):
        x0 = i * (TILE + GAP)
        d.rectangle([x0, 0, x0 + 142, 15], fill=(0, 0, 0))
        d.text((x0 + 3, 1), name, fill=(255, 255, 0))
    return strip


def nodule_label(study, zyx) -> str:
    """'big': inside a nodule >= 6 mm (radius + 3 mm, the scoring rule); 'small': only inside smaller ones; 'none'."""
    p = _mm(zyx, study.spacing)
    near = [n for n in study.nodules if np.linalg.norm(p - _mm(n.zyx, study.spacing)) <= n.diameter_mm / 2 + TOL_MM]
    if any(n.diameter_mm >= MIN_DIAM for n in near):
        return "big"
    return "small" if near else "none"


def strips(study, points, mode: str = "norm1000", size: int = 512, mask_rank: bool = True) -> list[dict]:
    """Render candidate strips for arbitrary (z, y, x, value) points through the harness measure tool."""
    S = Session(study, "harness_wm", mode, size)
    S.peaks = [tuple(map(float, p)) for p in points]
    out = []
    for lo in range(1, len(S.peaks) + 1, 5):
        res, recs = S.call("measure", {"peaks": list(range(lo, min(lo + 5, len(S.peaks) + 1)))})
        table = json.loads(res)["candidates"]
        sheet = Image.open(io.BytesIO(recs[0].png)).convert("RGB")
        for i, c in enumerate(table):
            crop = sheet.crop((0, i * (TILE + GAP), sheet.width, i * (TILE + GAP) + TILE))
            z, y, x, v = S.peaks[c["peak"] - 1]
            out.append({"peak": c["peak"], "zyx": [z, y, x], "wm": v, "image": unlabel(crop) if mask_rank else crop,
                        "measurement": {k: c[k] for k in MEAS_KEYS if c.get(k) is not None},
                        "label": nodule_label(study, (z, y, x))})
    return out


def candidate_rows(study, k: int = 10, proposal=None, **kw) -> list[dict]:
    """The top-k peaks of a proposal (default: the shape world model) of a study, as reader strips."""
    prop = proposal or ShapePrior()
    return strips(study, prop.top(study, k) if hasattr(prop, "top") else prop.peaks(study)[:k], **kw)
