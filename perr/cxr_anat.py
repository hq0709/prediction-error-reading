"""Anatomy of a cached radiograph (cxr_anatomy.py writes cache/cxr_anat/<key>.npy, a 128 x 128 uint16 bit field) and the
search domain of the radiograph world model: the thorax. Its CT counterpart is the grown lung domain inside which the
lung world model's addresses are read; on radiographs, addresses outside it fall on shoulders, neck, bowel gas, ECG
leads and markers.

No torch import.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

ANAT = "/raid/home/CAMCA/hj880/perr/cache/cxr_anat"
STRUCTURES = ("Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula", "Left Lung", "Right Lung",
              "Left Hilus Pulmonis", "Right Hilus Pulmonis", "Heart", "Aorta", "Facies Diaphragmatica", "Mediastinum",
              "Weasand", "Spine")
UP = 4                                                     # 128 grid -> 512 canvas


def bits(key: str) -> np.ndarray:
    return np.load(f"{ANAT}/{key}.npy")


def structure(key: str, names, b=None) -> np.ndarray:
    """Union of the named structures on the 128 grid."""
    b = bits(key) if b is None else b
    m = np.zeros(b.shape, bool)
    for n in names:
        m |= ((b >> STRUCTURES.index(n)) & 1).astype(bool)
    return m


def domain(key: str, kind: str = "hull", grow: int = 0) -> np.ndarray:
    """Search domain on the 512 canvas.
    hull:   convex hull of lungs, hila, heart, aorta, diaphragm and mediastinum (bit 15)
    organs: lungs, hila, heart, aorta and mediastinum, holes filled (no sub-diaphragmatic abdomen)
    grow:   dilation in canvas pixels."""
    b = bits(key)
    if kind == "hull":
        m = ((b >> 15) & 1).astype(bool)
    else:
        m = ndi.binary_fill_holes(structure(key, ("Left Lung", "Right Lung", "Left Hilus Pulmonis", "Right Hilus Pulmonis",
                                                   "Heart", "Aorta", "Mediastinum"), b))
    m = np.kron(m, np.ones((UP, UP), bool))
    if grow:
        m = ndi.binary_dilation(m, structure=np.ones((3, 3), bool), iterations=grow)
    return m


def search_domain(key: str, valid: np.ndarray, kind: str = "organs", grow: int = 8) -> np.ndarray:
    """domain() inside the radiograph; the whole radiograph when the segmentation finds (almost) no thorax."""
    D = domain(key, kind, grow) & valid
    return D if D.sum() >= 0.05 * valid.sum() else valid
