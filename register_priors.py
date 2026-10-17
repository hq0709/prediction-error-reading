"""Register each image's previous frontal radiograph onto it (results/cxr_prior_pairs_all.csv), 2-D affine as in
perr/cxr_course.py; already registered pairs are skipped."""
from multiprocessing import Pool

import pandas as pd

from perr.cxr_course import register


def _reg(args):
    try:
        return register(*args)
    except Exception as e:
        return {"key": args[0], "error": repr(e)[:200]}


if __name__ == "__main__":
    d = pd.read_csv("results/cxr_prior_pairs_all.csv")
    with Pool(96) as p:
        r = pd.DataFrame(p.map(_reg, list(zip(d.key, d.prior_key)), chunksize=16))
    r.to_csv("results/cxr_prior_reg_all.csv", index=False)
    print(len(r), "pairs; errors", int(r.get("error", pd.Series(dtype=str)).notna().sum()), "; NCC median", round(float(r.reg_ncc.median()), 3))
