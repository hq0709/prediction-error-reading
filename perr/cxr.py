"""Chest radiographs on one canonical grid: every frontal image is letterboxed to 512 x 512 (longer side 512, the
shorter one padded with zeros), uint8, with the scale and offset kept so that boxes and points map back to the
original pixels.

Sources:
  MIMIC-CXR-JPG v2.0.0 (PhysioNet, credentialed): frontal (PA / AP) images, CheXpert-labeler labels, official split.
  NIH ChestX-ray14: 1024 x 1024 PNGs, BBox_List_2017.csv (984 boxes, 8 findings) for localization.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image

MIMIC_JPG = "/raid/home/CAMCA/hj880/wt/dataset/original_mimic/mimic_cxr_jpg"
MIMIC_CSV = "/home/local/PARTNERS/hj880/dMLLM_in_Medicine/data/mimic"
NIH = "/raid/home/CAMCA/hj880/research/assets/data/nih-cxr14/data"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/cxr"
SIZE = 512
FINDINGS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture",
            "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
            "Pneumothorax", "Support Devices")


def mimic_frontal() -> pd.DataFrame:
    """One row per frontal MIMIC image: dicom_id, subject_id, study_id, view, split, study date, labels, path."""
    meta = pd.read_csv(f"{MIMIC_CSV}/mimic-cxr-2.0.0-metadata.csv")
    split = pd.read_csv(f"{MIMIC_CSV}/mimic-cxr-2.0.0-split.csv")
    lab = pd.read_csv(f"{MIMIC_CSV}/mimic-cxr-2.0.0-chexpert (1).csv")
    m = meta.merge(split[["dicom_id", "split"]], on="dicom_id").merge(lab, on=["subject_id", "study_id"], how="left")
    m = m[m.ViewPosition.isin(["PA", "AP"])].copy()
    m["path"] = [f"{MIMIC_JPG}/p{str(s)[:2]}/p{s}/s{t}/{d}.jpg" for s, t, d in zip(m.subject_id, m.study_id, m.dicom_id)]
    return m.reset_index(drop=True)


def letterbox(im: Image.Image, size: int = SIZE):
    """(uint8 array size x size, scale, (pad_x, pad_y)): original pixel (x, y) -> (x * scale + pad_x, y * scale + pad_y)."""
    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = np.asarray(im.convert("L").resize((nw, nh), Image.LANCZOS), np.uint8)
    out = np.zeros((size, size), np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = small
    return out, s, (px, py)


def cache_path(key: str) -> str:
    return f"{CACHE}/{key}.npy"


def build(key: str, path: str) -> dict:
    """Cache one image on the canonical grid; returns its geometry."""
    out = cache_path(key)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im = Image.open(path)
    arr, s, (px, py) = letterbox(im)
    if not os.path.exists(out):
        np.save(out, arr)
    return {"key": key, "w": im.size[0], "h": im.size[1], "scale": s, "pad_x": px, "pad_y": py}


def load(key: str) -> np.ndarray:
    return np.load(cache_path(key))
