"""Format check of the result files before scoring (no labels read): row counts, case ids, zyx in bounds, field
consistency, and run diagnostics from `raw` (empty answers, slices without <SEG>, component counts, sizes)."""
import json
import math
import os
import sys

import numpy as np

ROOT = "/raid/home/CAMCA/hj880/perr"
cases = {"luna": [u for n in ("fresh1", "fresh2") for u in open(f"{ROOT}/sota/eval_luna_{n}.txt").read().split()],
         "liver": open(f"{ROOT}/sota/eval_liver.txt").read().split(),
         "liver_ext": open(f"{ROOT}/sota/eval_liver_ext.txt").read().split()}
m = sys.argv[1] if len(sys.argv) > 1 else "omniabnorm_ct"
ok = True
for ds, cs in cases.items():
    if not os.path.exists(f"{ROOT}/results/sota/{m}_{ds}.jsonl"):
        continue
    rows = [json.loads(line) for line in open(f"{ROOT}/results/sota/{m}_{ds}.jsonl")]
    ids = [r["case"] for r in rows]
    bad = []
    for r in rows:
        shp = np.load(f"{ROOT}/cache/{ds}/{r['case']}.vol.npy", mmap_mode="r").shape
        if r["present"]:
            pts = [r["zyx"]] + [c["zyx"] for c in r["candidates"]]
            if r["zyx"] != r["candidates"][0]["zyx"] or not (r["diameter_mm"] and r["diameter_mm"] > 0):
                bad.append(r["case"] + ":answer_mismatch")
            for p in pts:
                if len(p) != 3 or not all(math.isfinite(c) and 0 <= c <= n - 1 for c, n in zip(p, shp)):
                    bad.append(r["case"] + ":out_of_bounds")
        elif r["zyx"] is not None or r["candidates"]:
            bad.append(r["case"] + ":zyx_without_present")
    dup = len(ids) - len(set(ids))
    miss = sorted(set(cs) - set(ids))
    extra = sorted(set(ids) - set(cs))
    raws = [json.loads(r["raw"]) for r in rows]
    nseg = sum(x["n_seg"] for x in raws)
    nkey = sum(x["n_key_slices"] for x in raws)
    nmask = sum(x["n_mask"] for x in raws)
    comps = np.array([x["n_components"] for x in raws])
    diam = np.array([r["diameter_mm"] for r in rows if r["present"]])
    full = sum(x["n_key_slices"] == x["shape"][0] for x in raws)
    print(f"{m} {ds:9s} rows {len(rows)} dup {dup} missing {len(miss)} extra {len(extra)} bad {len(bad)} "
          f"present {sum(bool(r['present']) for r in rows)} | all slices swept {full}/{len(rows)}, key slices {nkey}, "
          f"<SEG> {nseg} ({nseg / max(nkey, 1):.1%}), non-empty masks {nmask} ({nmask / max(nkey, 1):.1%}) | "
          f"components median {np.median(comps):.0f} [min {comps.min()}, max {comps.max()}] | answer diameter mm "
          f"median {np.median(diam):.1f} [{diam.min():.1f}, {diam.max():.1f}] | reports {sum(x['report'] is not None for x in raws)}")
    if bad:
        print("   bad:", bad[:10])
    ok &= not (dup or miss or extra or bad)
print("FORMAT OK" if ok else "FORMAT PROBLEMS")
