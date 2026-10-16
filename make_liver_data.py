"""Liver reader data: three-plane strips at the liver world model's top-10 addresses, labelled from the masks.

Cases with a target tumour (>= 10 mm) are split in half by a seeded permutation: "eval" (never used for any
choice) and the training pool, which also takes every case without a target. Training rows add four strips per
target tumour (centre jittered by up to 3 mm) and one per smaller tumour. Labels: 1 inside a target (mask grown by
3 mm); 0 outside every tumour; candidates on smaller tumours are 0 at evaluation and skipped in training.

    python make_liver_data.py
"""
import json
from multiprocessing import Pool

import numpy as np

from perr.liver import MIN_DIAM, QUESTION, LiverPrior, cases, has_target, liver_strips, load_liver

OUT = "/raid/home/CAMCA/hj880/perr/data/liver_reader"


def split():
    cs = cases()
    tgt = [c for c in cs if has_target(load_liver(c))]
    perm = np.random.default_rng(0).permutation(len(tgt))
    ev = sorted(tgt[i] for i in perm[: len(tgt) // 2])
    return ev, [c for c in cs if c not in ev]


def work(args):
    case, is_eval = args
    s = load_liver(case)
    rows = [dict(r, source="wm") for r in liver_strips(s, LiverPrior().top(s, 10))]
    if not is_eval:
        rng = np.random.default_rng(int(case.split("_")[1]))
        pts = []
        for n in s.nodules:
            big = n.diameter_mm >= MIN_DIAM
            for _ in range(4 if big else 1):
                j = rng.uniform(-1, 1, 3) * (3.0 if big else 1.0) / np.array(s.spacing)
                pts.append((*(np.array(n.zyx) + j), 0.0))
        rows += [dict(r, source="gt") for r in liver_strips(s, pts)]
    out = []
    for r in rows:
        if r["label"] == "small" and not is_eval:
            continue
        path = f"{OUT}/img/{case}_{r['source']}{r['peak']}.png"
        r.pop("image").save(path)
        out.append({**r, "uid": case, "image": path, "small": r["label"] == "small", "label": int(r["label"] == "big"),
                    "question": QUESTION})
    return out


if __name__ == "__main__":
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
