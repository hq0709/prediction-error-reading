"""Set 1 / Set 2 hit of IterMask3D's addresses read by the final lung verifier, with the rules of
collect_supp.lung_ablation (K = 10): the map's top address without the verifier, map score x verifier probability
(the selection rule of the final method and of the PatchCore and blob ablations), verifier probability alone, and the
oracle (any of the 10 addresses within radius + 3 mm = recall@10). The final method, PatchCore and the blob filter are
recomputed with the same code for reference.

    python sota/itermask3d/score.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from perr.score import score_case  # noqa: E402
from perr.study import load_luna  # noqa: E402

SOURCES = {   # name: (verifier results, candidate data), as collect_supp.LUNG_ABLATION
    "itermask3d": ("reader_abl_free_imk3d_eval_{}_{}.json", "data/abl_free_imk3d/eval_{}_{}.jsonl"),
    "final (PER world model)": ("reader_q8_ctxfree_nomeas_eval_{}_{}.json", "data/reader_ctxfree/eval_{}_{}.jsonl"),
    "patchcore": ("reader_abl_free_pc_eval_{}_{}.json", "data/abl_free_pc/eval_{}_{}.jsonl"),
    "shape (blob filter)": ("reader_abl_free_shape_eval_{}_{}.json", "data/abl_free_shape/eval_{}_{}.jsonl"),
}


def lung_ablation(k=10):
    out = {}
    for name, (res, data) in SOURCES.items():
        ent = {}
        for lo, hi in ((60, 160), (160, 260)):
            rp, dp = "results/" + res.format(lo, hi), data.format(lo, hi)
            if not (os.path.exists(rp) and os.path.exists(dp)):
                continue
            meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(dp))}
            by = defaultdict(list)
            for r in json.load(open(rp)):
                by[r["uid"]].append({**r, "zyx": meta[(r["uid"], r["peak"])]["zyx"]})
            h = defaultdict(list)
            for u, cs in by.items():
                cs = sorted(cs, key=lambda c: c["peak"])[:k]
                s = load_luna(u)
                hit = [score_case(s, {"present": True, "zyx": c["zyx"]})["hit6"] for c in cs]
                h["no_verifier_top1"].append(hit[0])
                h["map_x_verifier"].append(hit[max(range(len(cs)), key=lambda i: cs[i]["wm"] / cs[0]["wm"] * cs[i]["p"])])
                h["verifier_maxp"].append(hit[max(range(len(cs)), key=lambda i: cs[i]["p"])])
                h["recall_at_10"].append(any(hit))
            ent[f"set{1 if lo == 60 else 2}"] = {r: float(np.mean(v)) for r, v in h.items()} | {"n": len(by)}
        out[name] = ent
    return out


if __name__ == "__main__":
    res = lung_ablation()
    for name, ent in res.items():
        for s, v in ent.items():
            print(f"{name:26s} {s} (n={v['n']}): top-1 without verifier {v['no_verifier_top1']:.2f} | "
                  f"map x verifier {v['map_x_verifier']:.2f} | verifier max-p {v['verifier_maxp']:.2f} | "
                  f"recall@10 {v['recall_at_10']:.2f}")
    json.dump(res, open("results/itermask3d_lung_scores.json", "w"), indent=1)
