"""Liver reading budget and address list, chosen on a third site: 100 MSD Task08 cases (Memorial Sloan Kettering) are held
out of reader training (results/t08_dev_cases.txt) and read at up to 30 addresses from each address list: the fused
list (local-mean prior and learned world model, reciprocal rank), the local-mean prior alone and the learned world model
alone. The same lists are built for the evaluation sets (MSD evaluation half, HCC-TACE-Seg external), so the choice made
on Task08 is applied unchanged. Two-row strips (60 and 150 mm) as the final liver reader.

    python make_liver_kdev.py split                       # dev cases; reader training rows without them
    python make_liver_kdev.py strips t08|ext|msd <tag>    # data/liver_k30_<set>_<list>/eval.jsonl for each list
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from perr.context_wm import ContextPrior, fuse_rrf
from perr.liver import CONTEXT_FOV, QUESTION_2SCALE, LiverPrior, liver_strips

K = 30
SETS = {"t08": "liver_t08", "ext": "liver_ext", "msd": "liver"}
TAG = "lt08"


def module(which):
    if which == "t08":
        from perr import liver_t08 as m
        return m.CACHE, m.load_case, m.label_at, open("results/t08_dev_cases.txt").read().split()
    if which == "ext":
        from perr import liver_ext as m
        return m.CACHE, m.load_case, m.label_at, open("sota/eval_liver_ext.txt").read().split()
    from perr import liver as m
    return None, m.load_liver, m.label_at, open("sota/eval_liver.txt").read().split()


def lists(s, which, cache):
    prior = (LiverPrior(cache=cache) if cache else LiverPrior()).top(s, 60)
    learned = ContextPrior(tag=TAG, sigma=3.0, organ=SETS[which]).top(s, 60, nms_mm=15.0)
    return {"fused": fuse_rrf([prior, learned], s.spacing, c=10.0, nms_mm=15.0)[:K], "prior": prior[:K], "learned": learned[:K]}


def work(job):
    which, case = job
    cache, load, label_at, _ = module(which)
    s = load(case)
    out = {}
    for name, pts in lists(s, which, cache).items():
        d = f"data/liver_k30_{which}_{name}"
        rows = []
        for r in liver_strips(s, pts, label_fn=label_at, context_fov=CONTEXT_FOV):
            path = os.path.abspath(f"{d}/img/{case}_wm{r['peak']}.png")
            r.pop("image").save(path)
            rows.append({**r, "uid": case, "image": path, "source": "wm", "small": r["label"] == "small",
                         "label": int(r["label"] == "big"), "question": QUESTION_2SCALE})
        out[name] = rows
    return out


def main():
    global TAG
    if sys.argv[1] == "split":
        from perr.liver_t08 import load_case
        cases = open("results/t08_cases.txt").read().split()
        with_target = [c for c in cases if any(n.diameter_mm >= 10 for n in load_case(c).nodules)]
        dev = sorted(np.random.default_rng(0).choice(with_target, size=100, replace=False).tolist())
        open("results/t08_dev_cases.txt", "w").write("\n".join(dev) + "\n")
        src, out = "data/liver_reader_all3_t08", "data/liver_reader_all3_t08h"
        os.makedirs(out, exist_ok=True)
        rows = [json.loads(l) for l in open(f"{src}/train.jsonl")]
        keep = [r for r in rows if r["uid"] not in set(dev)]
        with open(f"{out}/train.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in keep)
        os.system(f"cp {src}/val.jsonl {out}/val.jsonl")
        print(f"{len(with_target)} Task08 cases with a target; {len(dev)} held out; training rows {len(rows)} -> {len(keep)}")
        return
    which, TAG = sys.argv[2], sys.argv[3]
    _, _, _, cases = module(which)
    for name in ("fused", "prior", "learned"):
        os.makedirs(f"data/liver_k30_{which}_{name}/img", exist_ok=True)
    with Pool(24) as p:
        res = p.map(work, [(which, c) for c in cases], chunksize=1)
    for name in ("fused", "prior", "learned"):
        rows = [r for x in res for r in x[name]]
        with open(f"data/liver_k30_{which}_{name}/eval.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        by = {}
        for r in rows:
            by.setdefault(r["uid"], []).append(r)
        rec = {k: np.mean([any(c["label"] for c in sorted(v, key=lambda c: c["peak"])[:k]) for v in by.values()]) for k in (10, 20, 30)}
        print(which, name, len(rows), "rows, recall@10/20/30", {k: round(float(v), 3) for k, v in rec.items()}, flush=True)


if __name__ == "__main__":
    main()
