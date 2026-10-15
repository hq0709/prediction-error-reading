"""LUMIERE follow-up pairs: the current study and the one before it, all four sequences in MNI atlas space.

Arrays are [z, y, x] with x -> patient left, y -> posterior, z -> superior (the CT convention of study.py),
1 mm isotropic. Intensities are z-scored inside the brain mask per sequence and time point, so that a
"no change" world model can subtract time points.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = "/raid/home/CAMCA/hj880/data_ext/lumiere"
IMG = f"{ROOT}/extracted/Imaging"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/lumiere"
SEQS = ("CT1", "T1", "T2", "FLAIR")
FILES = {"CT1": "ct1_skull_strip.nii.gz", "T1": "t1_skull_strip.nii.gz", "T2": "t2_skull_strip.nii.gz",
         "FLAIR": "flair_skull_strip.nii.gz"}
RESPONSE = ("PD", "SD", "PR", "CR")


def week_of(date: str) -> float:
    m = re.match(r"week-(\d+)(?:-(\d+))?", date)
    return float(m.group(1)) + (0.1 * float(m.group(2)) if m.group(2) else 0.0)


def _load(path) -> np.ndarray:
    a = np.asarray(nib.load(path).dataobj).astype(np.float32)   # [x, y, z], LAS
    return np.ascontiguousarray(a[:, ::-1, :].transpose(2, 1, 0))  # -> [z, y, x], LPS


def _complete(patient, date) -> bool:
    d = f"{IMG}/{patient}/{date}/DeepBraTumIA-segmentation/atlas"
    return all(os.path.exists(f"{d}/skull_strip/{f}") for f in FILES.values()) and os.path.exists(
        f"{d}/segmentation/seg_mask.nii.gz")


def ratings() -> pd.DataFrame:
    r = pd.read_csv(f"{ROOT}/LUMIERE-ExpertRating-v202211.csv")
    rc = [c for c in r.columns if c.startswith("Rating (")][0]
    ra = [c for c in r.columns if c.startswith("Rating rationale")][0]
    return r.rename(columns={rc: "rating", ra: "rationale"})


def pairs() -> list[dict]:
    """(prior, current) pairs whose current study has a RANO response rating and both studies are complete."""
    r = ratings()
    out = []
    for patient, g in r.groupby("Patient"):
        if not os.path.isdir(f"{IMG}/{patient}"):
            continue
        dates = sorted([d for d in os.listdir(f"{IMG}/{patient}") if _complete(patient, d)], key=week_of)
        for row in g.itertuples():
            rating = str(row.rating).strip()
            if rating not in RESPONSE or row.Date not in dates:
                continue
            i = dates.index(row.Date)
            if i == 0:
                continue
            out.append({"patient": patient, "prior": dates[i - 1], "current": row.Date, "rating": rating,
                        "rationale": "" if pd.isna(row.rationale) else str(row.rationale),
                        "weeks": round(week_of(row.Date) - week_of(dates[i - 1]), 1)})
    return out


def pilot_pairs(n: int = 60, seed: int = 0, offset: int = 0) -> list[dict]:
    """Balanced PD / non-PD sample, one pair per patient where possible."""
    ps = pairs()
    rng = np.random.default_rng(seed)
    pd_, nonpd = [p for p in ps if p["rating"] == "PD"], [p for p in ps if p["rating"] != "PD"]
    order_pd = [pd_[i] for i in rng.permutation(len(pd_))]
    order_np = [nonpd[i] for i in rng.permutation(len(nonpd))]
    mixed = [x for pair in zip(order_pd, order_np) for x in pair]
    return mixed[offset:offset + n]


@dataclass
class FollowUp:
    case_id: str
    series: dict            # "current:CT1" -> array, ..., "prior:FLAIR" -> array (z-scored)
    brain: np.ndarray       # union brain mask (uint8)
    seg_current: np.ndarray
    seg_prior: np.ndarray
    rating: str
    rationale: str
    weeks: float
    spacing: tuple = (1.0, 1.0, 1.0)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.brain.shape


def load_pair(p: dict) -> FollowUp:
    cid = f"{p['patient']}_{p['prior']}_{p['current']}"
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{cid}.npz"
    if not os.path.exists(path):
        arrs, masks, segs = {}, {}, {}
        for t in ("prior", "current"):
            d = f"{IMG}/{p['patient']}/{p[t]}/DeepBraTumIA-segmentation/atlas"
            m = _load(f"{d}/skull_strip/brain_mask.nii.gz") > 0
            masks[t] = m
            segs[t] = _load(f"{d}/segmentation/seg_mask.nii.gz").astype(np.int8)
            for s in SEQS:
                a = _load(f"{d}/skull_strip/{FILES[s]}")
                v = a[m]
                mu, sd = float(np.median(v)), float(np.percentile(v, 75) - np.percentile(v, 25)) / 1.349 + 1e-6
                arrs[f"{t}:{s}"] = np.where(m, (a - mu) / sd, -5.0).astype(np.float16)
        np.savez_compressed(path, brain=(masks["prior"] | masks["current"]).astype(np.uint8),
                            seg_current=segs["current"], seg_prior=segs["prior"], **arrs)
    z = np.load(path)
    series = {k: z[k].astype(np.float32) for k in z.files if ":" in k}
    return FollowUp(cid, series, z["brain"], z["seg_current"], z["seg_prior"], p["rating"], p["rationale"], p["weeks"])
