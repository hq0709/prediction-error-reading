"""What the reader sees for one chest radiograph, and the question it answers per finding.

Three inputs, the same reader recipe:
  global   the radiograph on the canonical 512 x 512 grid;
  wm       the radiograph with the world model's top-K addresses circled and numbered (left), plus each address
           enlarged from the original full-resolution image (right, 2 x 3 grid): the places where the image departs
           most from the normal anatomy the world model expects, shown losslessly. Enlarged views are observed pixels;
           no predicted image is shown.
  grid     control: the same layout at six fixed places (zones of the lung fields), i.e. enlargement without the
           world model. The question never says where the places come from.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .cxr import CACHE

K = 6
CROP_FRAC = 0.2                         # side of an enlarged view, as a fraction of the original image's longer side
TILE = 168
NAMES = {"Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly (an enlarged heart)",
         "Consolidation": "consolidation", "Edema": "pulmonary edema",
         "Enlarged Cardiomediastinum": "an enlarged cardiomediastinal silhouette", "Fracture": "a bone fracture",
         "Lung Lesion": "a lung lesion (nodule or mass)", "Lung Opacity": "a lung opacity",
         "Pleural Effusion": "a pleural effusion", "Pleural Other": "another pleural abnormality (thickening, plaque)",
         "Pneumonia": "pneumonia", "Pneumothorax": "a pneumothorax",
         "Support Devices": "a support device (tube, line, catheter or pacemaker)"}
Q_GLOBAL = "Frontal chest radiograph. Does it show {f}? Answer with one word: yes or no."
Q_WM = ("Frontal chest radiograph (left; numbered circles mark six places of interest) and those places enlarged "
        "(right). Does the radiograph show {f}? Answer with one word: yes or no.")      # same wording for every choice of places


def grid_points(img):
    """Control: six fixed places (upper, middle and lower zones of each lung field) instead of the world model's."""
    rows, cols = np.nonzero(np.asarray(img).any(1))[0], np.nonzero(np.asarray(img).any(0))[0]
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    return [(int(r0 + fr * (r1 - r0)), int(c0 + fc * (c1 - c0)), 0.0) for fr in (0.3, 0.5, 0.7) for fc in (0.3, 0.7)]


def render_global(key: str) -> Image.Image:
    return Image.fromarray(np.load(f"{CACHE}/{key}.npy")).convert("RGB")


def render_wm(key: str, points, original_path: str, geo: dict) -> Image.Image:
    """points: (row, col, value) on the 512 grid, best first; geo: scale and padding of the letterbox."""
    left = render_global(key)
    d = ImageDraw.Draw(left)
    for n, (r, c, _) in enumerate(points[:K], 1):
        d.ellipse([c - 14, r - 14, c + 14, r + 14], outline=(255, 0, 255), width=2)
        d.text((c + 15, r - 16), str(n), fill=(255, 255, 0))
    out = Image.new("RGB", (512 + 2 * TILE, 512), (0, 0, 0))
    out.paste(left, (0, 0))
    orig = Image.open(original_path).convert("L")
    side = int(CROP_FRAC * max(orig.size))
    for n, (r, c, _) in enumerate(points[:K]):
        x = (c - geo["pad_x"]) / geo["scale"]
        y = (r - geo["pad_y"]) / geo["scale"]
        crop = orig.crop((int(x - side / 2), int(y - side / 2), int(x + side / 2), int(y + side / 2)))
        tile = crop.resize((TILE, TILE), Image.LANCZOS).convert("RGB")
        dt = ImageDraw.Draw(tile)
        dt.rectangle([0, 0, 14, 13], fill=(0, 0, 0))
        dt.text((3, 1), str(n + 1), fill=(255, 255, 0))
        out.paste(tile, (512 + (n % 2) * TILE, (n // 2) * TILE + 4))
    return out
