"""Format check of the three result files before scoring (no labels read): row counts, case ids, zyx in bounds."""
import json
import math
import sys

import numpy as np

ROOT = "/raid/home/CAMCA/hj880/perr"
cases = {"luna": [u for n in ("fresh1", "fresh2") for u in open(f"{ROOT}/sota/eval_luna_{n}.txt").read().split()],
         "liver": open(f"{ROOT}/sota/eval_liver.txt").read().split(),
         "liver_ext": open(f"{ROOT}/sota/eval_liver_ext.txt").read().split()}
ok = True
for m in sys.argv[1:]:
    for ds, cs in cases.items():
        path = f"{ROOT}/results/sota/{m}_{ds}.jsonl"
        if ds == "liver_ext" and not __import__("os").path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        ids = [r["case"] for r in rows]
        bad = []
        for r in rows:
            if r["zyx"] is not None:
                shp = np.load(f"{ROOT}/cache/{ds}/{r['case']}.vol.npy", mmap_mode="r").shape
                if len(r["zyx"]) != 3 or not all(0 <= c <= n - 1 and math.isfinite(c) for c, n in zip(r["zyx"], shp)):
                    bad.append(r["case"])
                if not r["present"]:
                    bad.append(r["case"] + ":zyx_without_present")
        dup = len(ids) - len(set(ids))
        miss = sorted(set(cs) - set(ids))
        extra = sorted(set(ids) - set(cs))
        print(f"{m:12s} {ds:5s} rows {len(rows)} dup {dup} missing {len(miss)} extra {len(extra)} bad {len(bad)} "
              f"present {sum(bool(r['present']) for r in rows)} zyx {sum(r['zyx'] is not None for r in rows)}")
        ok &= not (dup or miss or extra or bad)
print("FORMAT OK" if ok else "FORMAT PROBLEMS")
