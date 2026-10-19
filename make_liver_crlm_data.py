"""Extra liver reader training data from TCIA CRLM: strips at the fused liver world model's top-10 addresses, plus
four jittered strips per tumour >= 10 mm and one per smaller tumour, labelled from the CRLM masks exactly as the MSD
training pool (perr/make_liver_data.py). The merged training set is the MSD pool plus CRLM; validation stays the MSD
validation cases.

    python make_liver_crlm_data.py <tag> <msd_data_dir> <out_dir> [context|plain] [holdout]
    e.g. python make_liver_crlm_data.py lcrlm data/liver_reader_fused_lcrlm data/liver_reader_all
"holdout" moves 40 CRLM cases into validation (for choosing between reader inputs on more than 7 MSD cases).
"""
# LIVER_TWO_SIDED=1: the learned model's departures of either sign (hyper- as well as hypo-attenuating findings)
import os as _os
TWO_SIDED = _os.environ.get("LIVER_TWO_SIDED") == "1"
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from perr.context_wm import ContextPrior, fuse_rrf
from perr.liver import CONTEXT_FOV, MIN_DIAM, QUESTION, QUESTION_2SCALE, LiverPrior, liver_strips
from perr.liver_crlm import CACHE, label_at, load_case

TAG, OUT, CONTEXT = "lcrlm", "data/liver_reader_all", False
HOLDOUT = 40          # CRLM cases kept for validation (with the MSD validation cases), so that reader choices see more data


def work(case):
    s = load_case(case)
    learned = ContextPrior(tag=TAG, sigma=3.0, organ="liver_crlm", two_sided=TWO_SIDED).top(s, 50, nms_mm=15.0)
    top = fuse_rrf([LiverPrior(cache=CACHE).top(s, 50), learned], s.spacing, c=10.0, nms_mm=15.0)[:10]
    ctx = CONTEXT_FOV if CONTEXT else None
    rows = [dict(r, source="wm") for r in liver_strips(s, top, label_fn=label_at, context_fov=ctx)]
    rng = np.random.default_rng(int(case.split("_")[1]))
    pts = []
    for n in s.nodules:
        big = n.diameter_mm >= MIN_DIAM
        for _ in range(4 if big else 1):
            j = rng.uniform(-1, 1, 3) * (3.0 if big else 1.0) / np.array(s.spacing)
            pts.append((*(np.array(n.zyx) + j), 0.0))
    rows += [dict(r, source="gt") for r in liver_strips(s, pts, label_fn=label_at, context_fov=ctx)]
    out = []
    for r in rows:
        if r["label"] == "small":
            continue                               # smaller tumours are skipped in training, as for MSD
        path = os.path.abspath(f"{OUT}/img/{case}_{r['source']}{r['peak']}.png")
        r.pop("image").save(path)
        out.append({**r, "uid": case, "image": path, "small": False, "label": int(r["label"] == "big"),
                    "question": QUESTION_2SCALE if CONTEXT else QUESTION})
    return out


if __name__ == "__main__":
    TAG, msd, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    CONTEXT = len(sys.argv) > 4 and sys.argv[4] == "context"
    holdout = len(sys.argv) > 5 and sys.argv[5] == "holdout"
    os.makedirs(f"{OUT}/img", exist_ok=True)
    cases = open("results/crlm_cases.txt").read().split()
    with Pool(16) as p:
        rows = [r for rs in p.map(work, cases, chunksize=1) for r in rs]
    held = set(np.random.default_rng(0).choice(cases, size=HOLDOUT, replace=False)) if holdout else set()
    msd_train = [json.loads(l) for l in open(f"{msd}/train.jsonl")]
    msd_val = [json.loads(l) for l in open(f"{msd}/val.jsonl")]
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in msd_train + [r for r in rows if r["uid"] not in held]:
            f.write(json.dumps(r) + "\n")
    with open(f"{OUT}/val.jsonl", "w") as f:
        for r in msd_val + [r for r in rows if r["uid"] in held]:
            f.write(json.dumps(r) + "\n")
    for src in ("wm", "gt"):
        sub = [r for r in rows if r["source"] == src]
        print(f"CRLM {src}: {len(sub)} rows, {sum(r['label'] for r in sub)} positive")
    print(f"train: MSD {len(msd_train)} + CRLM {len(rows)} rows")
