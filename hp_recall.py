"""Address recall for the hyperparameter sweep (appendix): scan recall@K (a nodule >= 6 mm among the top K addresses)
of the precision-weighted, signed context error, one hyperparameter at a time around the final setting (sigma 2 mm,
suppression half-width 10 mm), on the world-model dev split or on a uid list.

    python hp_recall.py nllfree runs_wm/ctx_free/split.json     # dev split (60 pool scans held out of world-model training)
    python hp_recall.py nllfree results/eval_uids_fresh.txt     # Set 1 + Set 2
    python hp_recall.py hpb0 runs_wm/ctx_free/split.json --only-final   # a retrained world model, final sigma and suppression

Writes results/hp_recall_<tag>_<dev|eval>.json: {variant: {K: recall}}.
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np

from perr.context_wm import ISO, load_iso, peaks_from_error, to_original
from perr.score import score_peaks
from perr.study import load_luna

TAG = "nllfree"
KS = (1, 5, 10, 20, 30)
FINAL = (2.0, 10.0)
VARIANTS = [FINAL] + [(s, 10.0) for s in (1.0, 3.0, 4.0)] + [(2.0, n) for n in (5.0, 15.0, 20.0)] + [(2.0, 10.0, "two")]


def name_of(v):
    return f"s{v[0]:g}_n{v[1]:g}" + ("_two" if len(v) > 2 else "")


def first_hit(s, pk):
    return next((k + 1 for k in range(len(pk)) if score_peaks(s, pk[k:k + 1], 1)), None)


def work(args):
    uid, variants = args
    s = load_luna(uid)
    if not any(n.diameter_mm >= 6 for n in s.nodules):
        return None
    R = np.load(f"{ISO}/{uid}.{TAG}R.npy", mmap_mode="r")
    S = np.load(f"{ISO}/{uid}.{TAG}S.npy", mmap_mode="r")
    _, D, meta = load_iso(uid)
    out = {}
    for v in variants:
        sg, nms, two = v[0], v[1], len(v) > 2
        pk = [(*to_original(p[:3], meta), p[3]) for p in
              peaks_from_error(R, D, k=max(KS), sigma=sg, nms_mm=nms, signed=True, S=S, weight="pool", two_sided=two)]
        out[name_of(v)] = first_hit(s, pk)
    return out


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("uids")
    ap.add_argument("--only-final", action="store_true")
    a = ap.parse_args()
    TAG = a.tag
    dev = a.uids.endswith(".json")
    uids = json.load(open(a.uids))["dev"] if dev else open(a.uids).read().split()
    variants = [FINAL] if a.only_final else VARIANTS
    with Pool(32) as p:
        res = [r for r in p.map(work, [(u, variants) for u in uids], chunksize=1) if r is not None]
    out = {}
    for var in variants:
        v = name_of(var)
        out[v] = {k: float(np.mean([r[v] is not None and r[v] <= k for r in res])) for k in KS}
        print(f"{TAG} {'dev' if dev else 'eval'} {v}: n={len(res)} " + " ".join(f"@{k} {out[v][k]:.3f}" for k in KS), flush=True)
    json.dump(out, open(f"results/hp_recall_{TAG}_{'dev' if dev else 'eval'}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
