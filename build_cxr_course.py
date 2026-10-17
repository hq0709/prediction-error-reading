"""MIMIC-CXR prior/current pairs: choose them, cache any missing image on the 512 grid, register each prior onto its
current image, and write results/cxr_pairs.csv (all validation and test pairs, 30,000 random training pairs).

    python build_cxr_course.py
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from perr.cxr import build, cache_path
from perr.cxr_course import pairs, register


def _cache(args):
    key, path = args
    if not os.path.exists(cache_path(key)):
        try:
            build(key, path)
        except Exception:
            return key
    return None


def _reg(args):
    try:
        return register(*args)
    except Exception as e:
        return {"key": args[0], "reg_metric": np.nan, "reg_ncc": np.nan, "error": repr(e)[:200]}


if __name__ == "__main__":
    p = pairs()
    rng = np.random.default_rng(0)
    tr = p[p.split == "train"]
    p = pd.concat([p[p.split.isin(["validate", "test"])], tr.iloc[rng.permutation(len(tr))[:30000]]])
    jobs = list(zip(p.key, p.path)) + list(zip(p.prior_key, p.prior_path))
    with Pool(96) as pool:
        bad = {k for k in pool.map(_cache, jobs, chunksize=32) if k}
        p = p[~p.key.isin(bad) & ~p.prior_key.isin(bad)]
        reg = pd.DataFrame(pool.map(_reg, list(zip(p.key, p.prior_key)), chunksize=8))
    p = p.merge(reg, on="key")
    p.to_csv("results/cxr_pairs.csv", index=False)
    print(p.split.value_counts().to_dict(), "| unreadable images", len(bad), "| registration NCC median",
          round(float(p.reg_ncc.median()), 3), "| NCC < 0.5:", int((p.reg_ncc < 0.5).sum()))
