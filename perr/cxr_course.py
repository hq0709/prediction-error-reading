"""Chest-radiograph disease course on MIMIC-CXR: each frontal image paired with the same patient's previous frontal
study, the prior registered onto the current image (2-D affine, images only), and per-finding change labels.

Labels per finding f (CheXpert-labeler, report-derived; not mentioned counts as negative, as for classification):
"new" = current f positive and prior f negative or not mentioned; "persistent" = positive in both (a negative for
"new"); "absent" = current f negative or not mentioned; uncertain on either side is left out for that finding.
strict_new_<f> marks the subset whose prior report explicitly negates f.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .cxr import CACHE, FINDINGS, mimic_frontal

REG = "/raid/home/CAMCA/hj880/perr/cache/cxr_reg"
LABELS = [f for f in FINDINGS if f != "No Finding"]


def pairs() -> pd.DataFrame:
    """One row per study with an earlier frontal study of the same patient (one frontal image per study)."""
    m = mimic_frontal()
    m["t"] = m.StudyDate.astype(int) * 1_000_000 + m.StudyTime.astype(float).astype(int)
    m["pa"] = (m.ViewPosition == "PA").astype(int)
    m = m.sort_values(["subject_id", "t", "pa"], ascending=[True, True, False])
    s = m.groupby("study_id").head(1).sort_values(["subject_id", "t"]).copy()
    g = s.groupby("subject_id")
    for c in ["dicom_id", "study_id", "ViewPosition", "path"] + LABELS:
        s[f"prior_{c}"] = g[c].shift(1)
    s = s[s.prior_dicom_id.notna()].copy()
    s["key"] = "mimic/" + s.dicom_id
    s["prior_key"] = "mimic/" + s.prior_dicom_id
    for f in LABELS:
        cur, pri = s[f], s[f"prior_{f}"]
        lab = np.full(len(s), np.nan)
        lab[(cur == 1) & ((pri == 0) | pri.isna())] = 1   # new (not mentioned counts as negative, as for classification)
        lab[(cur == 1) & (pri == 1)] = 0                  # persistent
        lab[(cur == 0) | cur.isna()] = 0                  # absent now
        lab[(cur == -1) | (pri == -1)] = np.nan           # uncertain on either side: left out
        s[f"new_{f}"] = lab
        s[f"strict_new_{f}"] = ((cur == 1) & (pri == 0)).astype(int)
    return s.reset_index(drop=True)


def register(key: str, prior_key: str) -> dict:
    """Affine registration of the prior onto the current image (Mattes MI on the 512 grid); caches the warped prior."""
    import SimpleITK as sitk                      # only the harness environment has it
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)      # one thread per process: the caller runs many
    out = f"{REG}/{key}.npy"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):                       # already registered: only recompute the agreement score
        a, b = np.load(f"{CACHE}/{key}.npy").astype(np.float32), np.load(out).astype(np.float32)
        v = (a > 0) & (b > 0)
        return {"key": key, "reg_metric": float("nan"),
                "reg_ncc": float(np.corrcoef(a[v], b[v])[0, 1]) if v.sum() > 100 else float("nan")}
    fixed = sitk.GetImageFromArray(np.load(f"{CACHE}/{key}.npy").astype(np.float32))
    moving = sitk.GetImageFromArray(np.load(f"{CACHE}/{prior_key}.npy").astype(np.float32))
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(2), sitk.CenteredTransformInitializerFilter.MOMENTS)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2, 0)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200, relaxationFactor=0.7)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(init, inPlace=False)
    try:
        T = R.Execute(fixed, moving)
        metric = R.GetMetricValue()
    except Exception:
        T, metric = init, float("nan")
    warped = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, T, sitk.sitkLinear, 0.0))
    np.save(out, np.clip(warped, 0, 255).astype(np.uint8))
    a = np.load(f"{CACHE}/{key}.npy").astype(np.float32)
    b = warped.astype(np.float32)
    v = (a > 0) & (b > 0)
    ncc = float(np.corrcoef(a[v], b[v])[0, 1]) if v.sum() > 100 else float("nan")
    return {"key": key, "reg_metric": metric, "reg_ncc": ncc}


def load_pair(key: str):
    return np.load(f"{CACHE}/{key}.npy"), np.load(f"{REG}/{key}.npy")
