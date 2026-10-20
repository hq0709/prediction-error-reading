"""Training augmentation for the radiograph detector baselines (--aug in train_cxr_detector.py and train_cxr_detr.py):
a random scale (0.9-1.1) and translation (up to 5 % of the canvas) about the canvas centre, and brightness and contrast
jitter (up to 10 %). No flips, since left and right anatomy differ; the YOLO11m run (fliplr 0) and the DenseNet-121
recipe do the same. Boxes follow the image; a box left with a side under 2 px after clipping is dropped.
"""
import numpy as np
from scipy import ndimage as ndi

SIDE = 512


def augment(img, bb, rng):
    """img: uint8 (512, 512) canvas; bb: list of dicts with finding, x, y, w, h on the canvas. Returns both augmented."""
    s = rng.uniform(0.9, 1.1)
    dx, dy = rng.uniform(-0.05, 0.05, 2) * SIDE
    c = (SIDE - 1) / 2
    # output pixel o comes from input i = (o - c - d) / s + c
    out = ndi.affine_transform(img.astype(np.float32), np.array([1 / s, 1 / s]),
                               offset=[c - (c + dy) / s, c - (c + dx) / s], order=1, mode="constant", cval=0.0)
    out = (out - 127.5) * rng.uniform(0.9, 1.1) + 127.5 + rng.uniform(-0.1, 0.1) * 255
    boxes = []
    for b in bb:
        x0, y0 = (b["x"] - c) * s + c + dx, (b["y"] - c) * s + c + dy
        x1, y1 = x0 + b["w"] * s, y0 + b["h"] * s
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(SIDE - 1.0, x1), min(SIDE - 1.0, y1)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            boxes.append({**b, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return np.clip(out, 0, 255).astype(np.uint8), boxes
