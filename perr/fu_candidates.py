"""Candidate strips for follow-up reading: at each error peak of the no-change world model, the prior and the
current study side by side on T1 after contrast and on FLAIR (axial, 60 mm), so the reader judges change at one
address. The DeepBraTumIA labels (1 enhancing core, 2 necrotic / non-enhancing, 3 oedema) are kept only for
analysis; the reader never sees them.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from scipy import ndimage as ndi

from .followup import FollowSession
from .render import Geom

TILE, GAP, FOV = 200, 4, 60
NAMES = {"CT1": "T1c", "FLAIR": "FLAIR"}
QUESTION = ("Brain MRI of a treated glioblastoma at one address, two time points aligned to the same space, axial, "
            "60 mm each, left to right: prior T1 after contrast, current T1 after contrast, prior FLAIR, current "
            "FLAIR. Magenta ticks mark an address where the current study may differ from the prior. Is "
            "there tumour progression at the marked address (a new or clearly enlarged contrast-enhancing lesion, or "
            "a clearly worse FLAIR abnormality), rather than unchanged disease, shrinking tumour or a mere intensity "
            "difference? Answer with one word: yes or no.")


def tile(S, seq, time, z, y, x):
    g = Geom("axial", int(round(z)), 0, "mri", S.fov_box("axial", x, y, FOV), TILE, extra={"time": time, "seq": seq})
    im = Image.open(io.BytesIO(S.render(g))).convert("RGB")
    d = ImageDraw.Draw(im)
    lab = f"{time} {NAMES[seq]}"
    d.rectangle([0, 0, 8 * len(lab) + 6, 15], fill=(0, 0, 0))
    d.text((3, 1), lab, fill=(255, 255, 0))
    m = TILE // 2
    d.line([(m - 14, m), (m - 6, m)], fill=(255, 0, 255))
    d.line([(m + 6, m), (m + 14, m)], fill=(255, 0, 255))
    return im


def local_tumour(seg, z, y, x, r=8, label=1) -> int:
    z, y, x = (int(round(c)) for c in (z, y, x))
    return int((seg[max(0, z - r):z + r + 1, max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] == label).sum())


def tumour_peaks(fu, k: int = 4):
    """Disease-course world model in tumour space: the expected current tumour is the prior tumour, so the
    prediction errors are the current enhancing (then oedema) components that the prior, grown by 2 mm, does not
    explain, ranked by volume. Addresses are the component voxels nearest to each centroid. When fewer than k
    errors exist, the largest current enhancing components are added (value 0) so the reader can check them."""
    out = []
    for lab, seq in ((1, "CT1"), (3, "FLAIR")):
        cur, pri = fu.seg_current == lab, ndi.binary_dilation(fu.seg_prior == lab, iterations=2)
        comp, n = ndi.label(cur & ~pri, structure=np.ones((3, 3, 3)))
        if n:
            sizes = ndi.sum(np.ones_like(comp), comp, index=np.arange(1, n + 1))
            for i in np.argsort(-sizes)[:k]:
                pts = np.argwhere(comp == i + 1)
                c = pts[np.argmin(((pts - pts.mean(0)) ** 2).sum(1))]
                out.append((seq, float(c[0]), float(c[1]), float(c[2]), float(sizes[i])))
        if len(out) >= k:
            break
    if len(out) < k:
        comp, n = ndi.label(fu.seg_current == 1, structure=np.ones((3, 3, 3)))
        if n:
            sizes = ndi.sum(np.ones_like(comp), comp, index=np.arange(1, n + 1))
            for i in np.argsort(-sizes)[:k - len(out)]:
                pts = np.argwhere(comp == i + 1)
                c = pts[np.argmin(((pts - pts.mean(0)) ** 2).sum(1))]
                out.append(("CT1", float(c[0]), float(c[1]), float(c[2]), 0.0))
    return out[:k]


def fu_strips(fu, k: int = 6, proposal: str = "intensity") -> list[dict]:
    S = FollowSession(fu, "harness_wm")
    out = []
    peaks = S.change_peaks(k) if proposal == "intensity" else tumour_peaks(fu, k)
    for n, (seq, z, y, x, v) in enumerate(peaks, 1):
        strip = Image.new("RGB", (4 * TILE + 3 * GAP, TILE), (40, 40, 40))
        for i, (s, t) in enumerate((("CT1", "prior"), ("CT1", "current"), ("FLAIR", "prior"), ("FLAIR", "current"))):
            strip.paste(tile(S, s, t, z, y, x), (i * (TILE + GAP), 0))
        out.append({"peak": n, "seq": seq, "zyx": [z, y, x], "wm": v, "image": strip,
                    "enh_prior": local_tumour(fu.seg_prior, z, y, x), "enh_current": local_tumour(fu.seg_current, z, y, x)})
    return out
