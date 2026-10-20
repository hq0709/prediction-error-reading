"""Extra liver reader training data from MSD Task08 (perr/liver_t08.py), built exactly as make_liver_crlm_data.py:
strips at the fused liver world model's top-10 addresses, plus four jittered strips per tumour >= 10 mm and one per
smaller tumour (smaller ones then skipped, as for MSD and CRLM), appended to the training rows of a merged MSD + CRLM
set; validation is unchanged.

    python make_liver_t08_data.py <tag> <merged MSD + CRLM dir> <out_dir> [context|plain]
    e.g. python make_liver_t08_data.py lt08 data/liver_reader_all3 data/liver_reader_all3_t08 context
"""
# LIVER_TWO_SIDED=1: the learned model's departures of either sign (hyper- as well as hypo-attenuating findings)
import os as _os
TWO_SIDED = _os.environ.get("LIVER_TWO_SIDED") == "1"
TRAIN_K = int(_os.environ.get("LIVER_TRAIN_K", "10"))   # world-model addresses per case in the reader data
import json
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np

from perr.context_wm import ContextPrior, fuse_rrf
from perr.liver import CONTEXT_FOV, MIN_DIAM, QUESTION, QUESTION_2SCALE, LiverPrior, liver_strips
from perr.liver_t08 import CACHE, LIST, label_at, load_case

TAG, OUT, CONTEXT = "lt08", "data/liver_reader_all3_t08", False


def work(case):
    s = load_case(case)
    learned = ContextPrior(tag=TAG, sigma=3.0, organ="liver_t08", two_sided=TWO_SIDED).top(s, max(50, 2 * TRAIN_K), nms_mm=15.0)
    top = fuse_rrf([LiverPrior(cache=CACHE).top(s, max(50, 2 * TRAIN_K)), learned], s.spacing, c=10.0, nms_mm=15.0)[:TRAIN_K]
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
            continue
        path = os.path.abspath(f"{OUT}/img/{case}_{r['source']}{r['peak']}.png")
        r.pop("image").save(path)
        out.append({**r, "uid": case, "image": path, "small": False, "label": int(r["label"] == "big"),
                    "question": QUESTION_2SCALE if CONTEXT else QUESTION})
    return out


if __name__ == "__main__":
    TAG, merged, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    CONTEXT = len(sys.argv) > 4 and sys.argv[4] == "context"
    os.makedirs(f"{OUT}/img", exist_ok=True)
    cases = open(LIST).read().split()
    with Pool(16) as p:
        rows = [r for rs in p.map(work, cases, chunksize=1) for r in rs]
    base = [json.loads(l) for l in open(f"{merged}/train.jsonl")]
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in base + rows:
            f.write(json.dumps(r) + "\n")
    shutil.copy(f"{merged}/val.jsonl", f"{OUT}/val.jsonl")
    for src in ("wm", "gt"):
        sub = [r for r in rows if r["source"] == src]
        print(f"Task08 {src}: {len(sub)} rows, {sum(r['label'] for r in sub)} positive")
    print(f"train: MSD + CRLM {len(base)} + Task08 {len(rows)} rows")
