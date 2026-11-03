"""Format check of a method's result files before scoring (no labels read): row counts, case ids, duplicates, zyx in
bounds, answer = candidates[0], present/zyx consistency, plus run diagnostics from `raw` (empty answers, clipping,
component counts, answer sizes, model time).

    python validate_results.py <method>      (e.g. segvol, clip_universal)
"""
import json
import math
import os
import sys

import numpy as np

ROOT = "/raid/home/CAMCA/hj880/perr"
CASES = {"luna": [u for n in ("fresh1", "fresh2") for u in open(f"{ROOT}/sota/eval_luna_{n}.txt").read().split()],
         "liver": open(f"{ROOT}/sota/eval_liver.txt").read().split(),
         "liver_ext": open(f"{ROOT}/sota/eval_liver_ext.txt").read().split()}


def main():
    m = sys.argv[1]
    ok = True
    for ds, cs in CASES.items():
        path = f"{ROOT}/results/sota/{m}_{ds}.jsonl"
        if not os.path.exists(path):
            print(f"{m} {ds:9s} missing file")
            ok = False
            continue
        rows = [json.loads(line) for line in open(path)]
        ids = [r["case"] for r in rows]
        bad = []
        for r in rows:
            shp = np.load(f"{ROOT}/cache/{ds}/{r['case']}.vol.npy", mmap_mode="r").shape
            if r["present"]:
                if r["zyx"] != r["candidates"][0]["zyx"] or not (r["diameter_mm"] and r["diameter_mm"] > 0):
                    bad.append(r["case"] + ":answer_mismatch")
                for p in [r["zyx"]] + [c["zyx"] for c in r["candidates"]]:
                    if len(p) != 3 or not all(math.isfinite(c) and 0 <= c <= n - 1 for c, n in zip(p, shp)):
                        bad.append(r["case"] + ":out_of_bounds")
                sc = [c["score"] for c in r["candidates"]]
                if sc != sorted(sc, reverse=True):
                    bad.append(r["case"] + ":candidates_not_sorted")
            elif r["zyx"] is not None or r["candidates"]:
                bad.append(r["case"] + ":zyx_without_present")
        raws = [json.loads(r["raw"]) for r in rows]
        dup = len(ids) - len(set(ids))
        miss = sorted(set(cs) - set(ids))
        extra = sorted(set(ids) - set(cs))
        comps = np.array([x["n_components"] for x in raws])
        diam = np.array([r["diameter_mm"] for r in rows if r["present"]])
        tmod = np.array([x["t_model_s"] for x in raws])
        empty = [r["case"] for r in rows if not r["present"]]
        clipped = sum(bool(x.get("clipped")) for x in raws)
        print(f"{m} {ds:9s} rows {len(rows)} dup {dup} missing {len(miss)} extra {len(extra)} bad {len(bad)} | "
              f"present {len(rows) - len(empty)} empty {len(empty)} clipped {clipped} | components median "
              f"{np.median(comps):.0f} [{comps.min()}, {comps.max()}] | answer diameter mm median "
              f"{np.median(diam) if len(diam) else float('nan'):.1f} [{diam.min() if len(diam) else 0:.1f}, "
              f"{diam.max() if len(diam) else 0:.1f}] | model s/case mean {tmod.mean():.1f}")
        if empty:
            mp = [round(x["max_prob"], 3) for r, x in zip(rows, raws) if not r["present"]]
            print(f"   empty: {len(empty)} cases, max prob median {np.median(mp):.2f}; "
                  + (", ".join(f"{c} ({p})" for c, p in zip(empty, mp)) if len(empty) <= 12 else ""))
        if bad:
            print("   bad:", bad[:10])
        ok &= not (dup or miss or extra or bad)
    print("FORMAT OK" if ok else "FORMAT PROBLEMS")


if __name__ == "__main__":
    main()
