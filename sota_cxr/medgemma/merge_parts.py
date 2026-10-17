"""Merge the test-split shards of run_medgemma.py (--part k/n) into results/sota_cxr/medgemma.csv, in test-table order."""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT, split_table  # noqa: E402

parts = sorted(glob.glob(f"{OUT}/medgemma_part*of*.csv"))
d = pd.concat([pd.read_csv(p) for p in parts])
gt = split_table("test")
assert not d.key.duplicated().any() and set(d.key) == set(gt.key), (len(d), d.key.nunique(), len(gt))
d = gt[["key"]].merge(d, on="key", how="left")
d.to_csv(f"{OUT}/medgemma.csv", index=False)
print(f"merged {len(parts)} parts -> {OUT}/medgemma.csv ({len(d)} rows)")
