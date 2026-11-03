"""Scan recall@K of IterMask3D's addresses (a nodule >= 6 mm within radius + 3 mm of one of the top K; ctx_recall's
rule) on a list of scans, per threshold-rule setting (gamma) written by itermask3d_lung.py infer. Used on the
world-model dev split (60 training-pool scans with a nodule >= 6 mm, held out of world-model and IterMask3D training)
to fix gamma; never on the evaluation sets.

    python sota/itermask3d/dev_recall.py dev --gammas 0.005 0.01 0.02 0.05 0.1
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from ctx_recall import first_hit  # noqa: E402
from perr.study import load_luna  # noqa: E402

KS = (1, 5, 10, 20, 50)
RUNS = f"{ROOT}/sota/itermask3d/runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="inference folder under sota/itermask3d/runs (its scans must be training-pool scans)")
    ap.add_argument("--gammas", type=float, nargs="+", required=True)
    a = ap.parse_args()
    dev = set(json.load(open(f"{ROOT}/runs_wm/ctx_free/split.json"))["dev"])
    res = {}
    for g in a.gammas:
        rows = []
        for f in sorted(os.listdir(f"{RUNS}/{a.out}")):
            if f.endswith(f".g{g:g}.json"):
                r = json.load(open(f"{RUNS}/{a.out}/{f}"))
                assert r["uid"] in dev, "dev_recall reads labels: training-pool dev scans only"
                rows.append(r)
        hits = [first_hit(load_luna(r["uid"]), [tuple(p) for p in r["peaks"]]) for r in rows]
        rec = {k: float(np.mean([h is not None and h <= k for h in hits])) for k in KS}
        res[g] = {"n": len(rows), "recall": rec, "stop_k_mean": float(np.mean([r["stop_k"] for r in rows])),
                  "r2_below_0.85": int(sum(not (r["r2"] >= 0.85) for r in rows)),
                  "shrink_iters_mean": float(np.mean([r["shrink_iters"] for r in rows])),
                  "final_mask_frac_mean": float(np.mean([r["final_mask_frac"] for r in rows]))}
        print(f"gamma {g:g} (n={len(rows)}): " + " ".join(f"R@{k} {v:.3f}" for k, v in rec.items()) +
              f" | stop iteration {res[g]['stop_k_mean']:.1f}, fits R2<0.85 {res[g]['r2_below_0.85']}, "
              f"shrink iterations {res[g]['shrink_iters_mean']:.1f}, final mask {res[g]['final_mask_frac_mean']:.3f}",
              flush=True)
    json.dump(res, open(f"{RUNS}/dev_recall_{a.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
