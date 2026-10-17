"""Shared conventions for the 13-finding chest X-ray baselines on the official MIMIC-CXR-JPG test split.

Evaluation set: the test rows of results/cxr_mimic.csv = every frontal (PA / AP) image of the official test split
(3,403 images).  Findings: the 13 CheXpert labels other than "No Finding".  Per finding, y = 1 if the label is
1.0 and y = 0 if it is 0.0 or missing; uncertain (-1.0) images are excluded for that finding only.
Methods write results/sota_cxr/<method>.csv with columns: key, then one probability column per finding name.
"""
import numpy as np
import pandas as pd

ROOT = "/raid/home/CAMCA/hj880/perr"
TABLE = f"{ROOT}/results/cxr_mimic.csv"
OUT = f"{ROOT}/results/sota_cxr"
CACHE = f"{ROOT}/cache/cxr"                     # 512 x 512 letterboxed uint8 arrays, cache/cxr/<key>.npy
FINDINGS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture",
            "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
            "Support Devices")
COMPETITION = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")


def split_table(split: str = "test") -> pd.DataFrame:
    """Rows of results/cxr_mimic.csv for one split ("test" or "validate"; train holds only "No Finding" images)."""
    d = pd.read_csv(TABLE)
    return d[d.split == split].reset_index(drop=True)


def targets(df: pd.DataFrame):
    """(y, keep), both (n, 13): y = 1 if the label is 1.0, else 0 (0.0 or missing); keep = label is not -1.0."""
    lab = df[list(FINDINGS)].to_numpy(float)
    return (lab == 1.0).astype(np.float32), lab != -1.0
