"""Reader data: candidate strips (rendered exactly as the agent sees them) with labels checked against LUNA16.

Label 1: the candidate lies inside an annotated nodule >= 6 mm (radius + 3 mm, the scoring rule); otherwise 0,
including candidates on smaller nodules (the question asks for >= 6 mm).
Training pool (LUNA16 scans outside every evaluation set; 10% of scans held out for validation):
  "wm"  the top-K shape world-model peaks of each scan;
  "gt"  with --gt, four strips per annotated nodule >= 6 mm and one per smaller nodule, centred within 2 mm
        (resp. 1 mm) of the annotated centre, to give the reader more positives and hard negatives.
Evaluation (--eval LO HI): the top-K world-model peaks of pilot_cases()[LO:HI], nothing else.
--proposal picks the world model: shape (default), context (the learned anatomy model; context_raw without the
precision weighting, the ablation), patchcore (PatchCore peaks saved by ct_patchcore.py, a non-predictive model of
normal), mixed (the two
rank-interleaved), fused (the two fused by reciprocal rank), or intensity (no world model, the ablation); --out
keeps each proposal's data apart.

    python make_reader_data.py --gt
    python make_reader_data.py --eval 60 160
    python make_reader_data.py --gt --proposal mixed --out data/reader_mixed
"""
import argparse
import json
import os
import zlib
from multiprocessing import Pool

import numpy as np

from perr.candidates import candidate_rows, strips
from perr.context_wm import ContextPrior, FusedPrior, MixedPrior
from perr.score import MIN_DIAM
from perr.study import load_luna, pilot_cases
from perr.worldmodels import IntensityPeaks, ShapePrior

OUT = "/raid/home/CAMCA/hj880/perr/data/reader"
K = 10
GT = False
PROPOSAL = "shape"
CTX_TAG = "nll"
SIGMA, NMS, TWO_SIDED = 2.0, 10.0, False


class _Saved:
    """Addresses saved by another proposer: a results JSON with each scan's ranked peaks in original voxel coordinates
    (e.g. ct_patchcore.py's PatchCore peaks)."""

    def __init__(self, path):
        self.by = {r["uid"]: r["peaks"] for r in json.load(open(path))}

    def top(self, study, k=10):
        return [tuple(p) for p in self.by[study.case_id][:k]]


class _Top:
    """A proposal with its non-maximum suppression width fixed (the hyperparameter sweep)."""

    def __init__(self, prop, nms_mm):
        self.prop, self.nms_mm = prop, nms_mm

    def top(self, study, k=10):
        return self.prop.top(study, k, nms_mm=self.nms_mm)


PROPOSALS = {"shape": lambda: ShapePrior(), "intensity": lambda: IntensityPeaks(),
             "context": lambda: _Top(ContextPrior(CTX_TAG, sigma=SIGMA, two_sided=TWO_SIDED), NMS),
             "patchcore": lambda: _Saved("results/ct_patchcore_wrn50_recall.json"),
             "context_raw": lambda: ContextPrior(CTX_TAG, weight="none"),
             "mixed": lambda: MixedPrior(CTX_TAG, weight="pool"),
             "fused": lambda: FusedPrior(CTX_TAG, weight="pool")}


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
        src = "int" if PROPOSAL == "intensity" else "wm"
        rows = [dict(r, source=src) for r in candidate_rows(s, K, proposal=PROPOSALS[PROPOSAL]())]
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
    ap.add_argument("--proposal", choices=sorted(PROPOSALS), default="shape")
    ap.add_argument("--ctx-tag", default="nll", help="cached error of the learned world model to use")
    ap.add_argument("--k", type=int, default=10, help="candidates per scan")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sigma", type=float, default=2.0, help="smoothing of the context error (mm; sweep)")
    ap.add_argument("--nms", type=float, default=10.0, help="non-maximum suppression half-width (mm; sweep)")
    ap.add_argument("--two-sided", action="store_true", help="departures of either sign (magnitude of the error; sweep)")
    a = ap.parse_args()
    K, OUT, CTX_TAG = a.k, os.path.abspath(a.out), a.ctx_tag
    SIGMA, NMS, TWO_SIDED = a.sigma, a.nms, a.two_sided
    GT = a.gt and not a.eval
    PROPOSAL = a.proposal
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
        tag = ("int_" if PROPOSAL == "intensity" else "") + ("" if K == 10 else f"k{K}_")
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
