"""Extra liver reader training data from TCIA CRLM: strips at the fused liver world model's top-10 addresses, plus
four jittered strips per tumour >= 10 mm and one per smaller tumour, labelled from the CRLM masks exactly as the MSD
training pool (perr/make_liver_data.py). The merged training set is the MSD pool plus CRLM; validation stays the MSD
validation cases.

    python make_liver_crlm_data.py <tag> <msd_data_dir> <out_dir>
    e.g. python make_liver_crlm_data.py lcrlm data/liver_reader_fused_lcrlm data/liver_reader_all
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from perr.context_wm import ContextPrior, fuse_rrf
from perr.liver import MIN_DIAM, QUESTION, LiverPrior, liver_strips
from perr.liver_crlm import CACHE, label_at, load_case

TAG, OUT = "lcrlm", "data/liver_reader_all"


def work(case):
    s = load_case(case)
    learned = ContextPrior(tag=TAG, sigma=3.0, organ="liver_crlm").top(s, 50, nms_mm=15.0)
    top = fuse_rrf([LiverPrior(cache=CACHE).top(s, 50), learned], s.spacing, c=10.0, nms_mm=15.0)[:10]
    rows = [dict(r, source="wm") for r in liver_strips(s, top, label_fn=label_at)]
    rng = np.random.default_rng(int(case.split("_")[1]))
    pts = []
    for n in s.nodules:
        big = n.diameter_mm >= MIN_DIAM
        for _ in range(4 if big else 1):
            j = rng.uniform(-1, 1, 3) * (3.0 if big else 1.0) / np.array(s.spacing)
            pts.append((*(np.array(n.zyx) + j), 0.0))
    rows += [dict(r, source="gt") for r in liver_strips(s, pts, label_fn=label_at)]
    out = []
    for r in rows:
        if r["label"] == "small":
            continue                               # smaller tumours are skipped in training, as for MSD
        path = os.path.abspath(f"{OUT}/img/{case}_{r['source']}{r['peak']}.png")
        r.pop("image").save(path)
        out.append({**r, "uid": case, "image": path, "small": False, "label": int(r["label"] == "big"), "question": QUESTION})
    return out


if __name__ == "__main__":
    TAG, msd, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(f"{OUT}/img", exist_ok=True)
    cases = open("results/crlm_cases.txt").read().split()
    with Pool(16) as p:
        rows = [r for rs in p.map(work, cases, chunksize=1) for r in rs]
    msd_train = [json.loads(l) for l in open(f"{msd}/train.jsonl")]
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in msd_train + rows:
            f.write(json.dumps(r) + "\n")
    os.system(f"cp {msd}/val.jsonl {OUT}/val.jsonl")
    for src in ("wm", "gt"):
        sub = [r for r in rows if r["source"] == src]
        print(f"CRLM {src}: {len(sub)} rows, {sum(r['label'] for r in sub)} positive")
    print(f"train: MSD {len(msd_train)} + CRLM {len(rows)} rows")
