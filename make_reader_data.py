"""Reader data: candidate strips (rendered exactly as the agent sees them) with labels checked against LUNA16.

Label 1: the candidate lies inside an annotated nodule >= 6 mm (radius + 3 mm, the scoring rule); otherwise 0,
including candidates on smaller nodules (the question asks for >= 6 mm).
Training pool (LUNA16 scans outside every evaluation set; 10% of scans held out for validation):
  "wm"  the top-K shape world-model peaks of each scan;
  "gt"  with --gt, four strips per annotated nodule >= 6 mm and one per smaller nodule, centred within 2 mm
        (resp. 1 mm) of the annotated centre, to give the reader more positives and hard negatives.
Evaluation (--eval LO HI): the top-K world-model peaks of pilot_cases()[LO:HI], nothing else.

    python make_reader_data.py --gt
    python make_reader_data.py --eval 60 160
"""
import argparse
import json
import os
import zlib
from multiprocessing import Pool

import numpy as np

from perr.candidates import candidate_rows, strips
from perr.score import MIN_DIAM
from perr.study import load_luna, pilot_cases
from perr.worldmodels import IntensityPeaks

OUT = "/raid/home/CAMCA/hj880/perr/data/reader"
K = 10
GT = False
PROPOSAL = None            # None: the shape world model; "intensity": the no-world-model ablation


def gt_points(s, rng):
    pts = []
    for n in s.nodules:
        big = n.diameter_mm >= MIN_DIAM
        for _ in range(4 if big else 1):
            j = rng.uniform(-1, 1, 3) * (2.0 if big else 1.0) / np.array(s.spacing)
            pts.append((*(np.array(n.zyx, float) + j), 0.0))
    return pts


def work(uid):
    try:
        s = load_luna(uid)
        prop = IntensityPeaks() if PROPOSAL == "intensity" else None
        src = "int" if PROPOSAL == "intensity" else "wm"
        rows = [dict(r, source=src) for r in candidate_rows(s, K, proposal=prop)]
        if GT and s.nodules:
            rows += [dict(r, source="gt") for r in strips(s, gt_points(s, np.random.default_rng(zlib.crc32(uid.encode()))))]
        out = []
        for r in rows:
            path = f"{OUT}/img/{uid}_{r['source']}{r['peak']}.png"
            r.pop("image").save(path)
            out.append({**r, "uid": uid, "image": path, "small": r["label"] == "small", "label": int(r["label"] == "big")})
        return out
    except Exception as e:  # keep going; report at the end
        return [{"uid": uid, "error": repr(e)[:200]}]


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--eval", type=int, nargs=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--proposal", choices=["shape", "intensity"], default="shape")
    ap.add_argument("--k", type=int, default=10, help="candidates per scan")
    a = ap.parse_args()
    K = a.k
    GT = a.gt and not a.eval
    PROPOSAL = None if a.proposal == "shape" else a.proposal
    os.makedirs(f"{OUT}/img", exist_ok=True)
    if a.eval:
        uids = pilot_cases(a.eval[1])[a.eval[0]:a.eval[1]]
    else:
        uids = open("results/train_uids.txt").read().split()
    with Pool(a.workers) as p:
        results = p.map(work, uids, chunksize=1)
    rows = [r for rs in results for r in rs if "error" not in r]
    errs = [r for rs in results for r in rs if "error" in r]
    if a.eval:
        tag = ("" if PROPOSAL is None else "int_") + ("" if K == 10 else f"k{K}_")
        write(f"{OUT}/eval_{tag}{a.eval[0]}_{a.eval[1]}.jsonl", rows)
        splits = {"eval": rows}
    else:
        val = set(np.random.default_rng(0).choice(uids, size=len(uids) // 10, replace=False))
        splits = {"train": [r for r in rows if r["uid"] not in val], "val": [r for r in rows if r["uid"] in val]}
        for name, rs in splits.items():
            write(f"{OUT}/{name}.jsonl", rs)
    for name, rs in splits.items():
        for src in ("wm", "gt", "int"):
            sub = [r for r in rs if r["source"] == src]
            if sub:
                print(f"{name} {src}: {len(sub)} rows, {sum(r['label'] for r in sub)} positive, "
                      f"{sum(r['small'] for r in sub)} on small nodules", flush=True)
    print(f"scans {len(uids)}, errors {len(errs)} {errs[:3]}", flush=True)
