"""Liver reader data: three-plane strips at the liver world model's top-10 addresses, labelled from the masks.

Cases with a target tumour (>= 10 mm) are split in half by a seeded permutation: "eval" (never used for any
choice) and the training pool, which also takes every case without a target. Training rows add four strips per
target tumour (centre jittered by up to 3 mm) and one per smaller tumour. Labels: 1 inside a target (mask grown by
3 mm); 0 outside every tumour; candidates on smaller tumours are 0 at evaluation and skipped in training.

    python make_liver_data.py
    python make_liver_data.py --proposal fused --out data/liver_reader_fused     # with the learned liver world model
"""
# LIVER_TWO_SIDED=1: the learned model's departures of either sign (hyper- as well as hypo-attenuating findings)
import os as _os
TWO_SIDED = _os.environ.get("LIVER_TWO_SIDED") == "1"
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from perr.context_wm import ContextPrior, fuse_rrf
from perr.liver import CONTEXT_FOV, MIN_DIAM, QUESTION, QUESTION_2SCALE, LiverPrior, cases, has_target, liver_strips, load_liver

OUT = "/raid/home/CAMCA/hj880/perr/data/liver_reader"
PROPOSAL = "liver"
TAG = "lnll"
CONTEXT = False


def proposal_top(s, k=10):
    """liver: the local-mean world model; context: the learned one (sigma fixed on the liver dev split); fused: the two
    by reciprocal rank (c = 10, as for the lung; chosen over either alone on the training pool)."""
    learned = lambda n: ContextPrior(tag=TAG, sigma=3.0, organ="liver", two_sided=TWO_SIDED).top(s, n, nms_mm=15.0)
    if PROPOSAL == "context":
        return learned(k)
    if PROPOSAL == "fused":
        return fuse_rrf([LiverPrior().top(s, 50), learned(50)], s.spacing, c=10.0, nms_mm=15.0)[:k]
    return LiverPrior().top(s, k)


def split():
    cs = cases()
    tgt = [c for c in cs if has_target(load_liver(c))]
    perm = np.random.default_rng(0).permutation(len(tgt))
    ev = sorted(tgt[i] for i in perm[: len(tgt) // 2])
    return ev, [c for c in cs if c not in ev]


def work(args):
    case, is_eval = args
    s = load_liver(case)
    ctx = CONTEXT_FOV if CONTEXT else None
    rows = [dict(r, source="wm") for r in liver_strips(s, proposal_top(s, 10), context_fov=ctx)]
    if not is_eval:
        rng = np.random.default_rng(int(case.split("_")[1]))
        pts = []
        for n in s.nodules:
            big = n.diameter_mm >= MIN_DIAM
            for _ in range(4 if big else 1):
                j = rng.uniform(-1, 1, 3) * (3.0 if big else 1.0) / np.array(s.spacing)
                pts.append((*(np.array(n.zyx) + j), 0.0))
        rows += [dict(r, source="gt") for r in liver_strips(s, pts, context_fov=ctx)]
    out = []
    for r in rows:
        if r["label"] == "small" and not is_eval:
            continue
        path = f"{OUT}/img/{case}_{r['source']}{r['peak']}.png"
        r.pop("image").save(path)
        out.append({**r, "uid": case, "image": path, "small": r["label"] == "small", "label": int(r["label"] == "big"),
                    "question": QUESTION_2SCALE if CONTEXT else QUESTION})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=["liver", "context", "fused"], default="liver")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tag", default="lnll", help="cached error of the learned liver world model")
    ap.add_argument("--context", action="store_true", help="add a 150 mm row of the same planes to each strip")
    a = ap.parse_args()
    PROPOSAL, OUT, TAG, CONTEXT = a.proposal, os.path.abspath(a.out), a.tag, a.context
    os.makedirs(f"{OUT}/img", exist_ok=True)
    ev, pool = split()
    with Pool(24) as p:
        res = p.map(work, [(c, True) for c in ev] + [(c, False) for c in pool], chunksize=1)
    rows = [r for rs in res for r in rs]
    held = set(np.random.default_rng(1).choice(pool, size=len(pool) // 10, replace=False))
    parts = {"eval": [r for r in rows if r["uid"] in ev],
             "train": [r for r in rows if r["uid"] not in ev and r["uid"] not in held],
             "val": [r for r in rows if r["uid"] in held]}
    for name, rs in parts.items():
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        for src in ("wm", "gt"):
            sub = [r for r in rs if r["source"] == src]
            if sub:
                print(f"{name} {src}: {len(sub)} rows, {sum(r['label'] for r in sub)} positive, {len({r['uid'] for r in sub})} cases")
