"""Test-time augmentation (TTA) of the fixed verifiers: average each candidate's P(yes) (or its logit) over a few
perturbed views, then apply the unchanged selection rule. No retraining; only the scoring changes.

Views
  radiographs  id; z0.9 / z1.1: both crops re-rendered from the original image with their sides divided by 0.9 / 1.1
               (make_cxr_tta_views.py); c0.9 / c1.1: contrast scaled about mid-grey (train_cxr_focal_reader.py
               --tta-view). No flips (laterality), no translation.
  lung CT      id; t0..t3: the strip re-rendered with its centre moved 1 mm towards the vertices of a regular
               tetrahedron (make_reader_tta.py); the mean offset is zero.

Configurations compared on validation only (declared before scoring):
  radiographs  base; scale3 = {id, z0.9, z1.1}; contrast3 = {id, c0.9, c1.1}; all5; each with mean P or mean logit.
               Criterion: held-out tenth of ChestX-Det train (wmu_cxdet_val), image-level mean AUROC over the four
               focal findings with max-p over the 60 union candidates (the criterion that kept cxr_focal_e3);
               hit and LROC reported.
  lung CT      base; tta5 = {id, t0..t3}; tta4 = {t0..t3}; each with mean P or mean logit.
               Criterion: out-of-fold scores of the 162 training-pool scans with a nodule >= 6 mm (each half scored by
               the verifier trained on the other half, q8_ctxfree_cvA/B; choose_rule_cv.py), top-1 hit with world
               model x verifier at K = 10; candidate AUROC reported.
TTA is adopted only if its best configuration beats base on the criterion; test numbers are printed only with --test.

    python tta_eval.py val            # validation table and the decision
    python tta_eval.py test           # evaluation sets, base and the configurations (after the decision)
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

R = "results/cxr_focal"
TYPES = ("Nodule", "Mass", "Calcification", "Fracture")
EPS = 1e-6


def agg(ps, how):
    ps = np.clip(np.asarray(ps, float), EPS, 1 - EPS)
    if how == "prob":
        return ps.mean(0)
    lg = np.log(ps / (1 - ps)).mean(0)
    return 1 / (1 + np.exp(-lg))


# ---------------------------------------------------------------- radiographs
CXR_SETS = {"base": ("id",), "scale3": ("id", "z0.9", "z1.1"), "contrast3": ("id", "c0.9", "c1.1"),
            "all5": ("id", "z0.9", "z1.1", "c0.9", "c1.1")}


def cxr_view_file(view, split):
    s = f"cxdet_{split}"
    return {"id": f"{R}/tta_e3_id/wmu_{s}.jsonl" if split == "val" else f"{R}/cxr_focal_e3/wmu_{s}.jsonl",
            "z0.9": f"{R}/tta_e3/tta_z0.9_wmu_{s}.jsonl", "z1.1": f"{R}/tta_e3/tta_z1.1_wmu_{s}.jsonl",
            "c0.9": f"{R}/tta_e3_c0.9/wmu_{s}.jsonl", "c1.1": f"{R}/tta_e3_c1.1/wmu_{s}.jsonl"}[view]


def cxr(split):
    from cxr_recall import boxes
    from score_cxr_focal import evaluate
    b = boxes("cxdet")
    b = b[b.split == ("train" if split == "val" else "test")]
    views = {}
    for v in CXR_SETS["all5"]:
        f = cxr_view_file(v, split)
        if os.path.exists(f):
            views[v] = {(r["key"], r["i"]): r for r in map(json.loads, open(f))}
    out = {}
    if split == "val" and os.path.exists(f"{R}/cxr_focal_e3/wmu_cxdet_val.jsonl"):   # the stored baseline file
        rows = [json.loads(l) for l in open(f"{R}/cxr_focal_e3/wmu_cxdet_val.jsonl")]
        out["base (stored file)"] = evaluate(rows, b, TYPES, ks=(60,), rules=("max-p",))[["AUROC", "hit", "LROC"]].mean().to_dict()
    for name, vs in CXR_SETS.items():
        if not all(v in views for v in vs):
            continue
        for how in (("prob",) if name == "base" else ("prob", "logit")):
            ref = views["id"]
            rows = []
            for k, r in ref.items():
                q = dict(r)
                for t in TYPES:
                    q[f"p_{t}"] = float(agg([[views[v][k][f"p_{t}"]] for v in vs], how)[0])
                rows.append(q)
            res = evaluate(rows, b, TYPES, ks=(60,), rules=("max-p",))
            out[name if name == "base" else f"{name}/{how}"] = res[["AUROC", "hit", "LROC"]].mean().to_dict()
    return out


# ---------------------------------------------------------------- lung CT
LUNG_SETS = {"base": ("z",), "tta5": ("z", "t0", "t1", "t2", "t3"), "tta4": ("t0", "t1", "t2", "t3")}


def lung_rows(split, view):
    """Scored candidate rows of one view: split "val" = both out-of-fold halves; "eval_60_160" / "eval_160_260"."""
    if split == "val":
        fs = [f"results/reader_tta_cv{h}_{view}_heldout.json" for h in "AB"]
    elif view == "z":
        fs = [f"results/reader_q8_ctxfree_nomeas_{split}.json"]     # the final verifier's stored scores
    else:
        fs = [f"results/reader_tta_final_{view}_{split}.json"]
    if not all(os.path.exists(f) for f in fs):
        return None
    return [r for f in fs for r in json.load(open(f))]


def lung_metrics(rows, k=10, uids=None):
    by = defaultdict(list)
    for r in rows:
        if uids is None or r["uid"] in uids:
            by[r["uid"]].append(r)
    cases = [sorted(v, key=lambda r: r["peak"])[:k] for v in by.values()]
    fuse = np.mean([max(cs, key=lambda c: c["wm"] / cs[0]["wm"] * c["p"])["label"] for cs in cases])
    maxp = np.mean([max(cs, key=lambda c: c["p"])["label"] for cs in cases])
    oracle = np.mean([any(c["label"] for c in cs) for cs in cases])
    pos = np.array([c["p"] for cs in cases for c in cs if c["label"]])
    neg = np.array([c["p"] for cs in cases for c in cs if not c["label"]])
    auc = float(np.mean([(a > neg).mean() + 0.5 * (a == neg).mean() for a in pos]))
    return {"n": len(cases), "hit_wmxv": float(fuse), "hit_maxp": float(maxp), "oracle": float(oracle),
            "verif_miss": float(oracle - fuse), "cand_AUROC": auc}


def lung(split):
    uids = set(json.load(open("results/pool_scans_with_big.json"))) if split == "val" else None
    views = {v: lung_rows(split, v) for v in LUNG_SETS["tta5"]}
    out = {}
    if split == "val":
        stored = [r for h in "AB" for r in json.load(open(f"results/reader_q8_ctxfree_cv{h}_heldout.json"))]
        out["base (stored file)"] = lung_metrics(stored, uids=uids)
    for name, vs in LUNG_SETS.items():
        if any(views[v] is None for v in vs):
            continue
        idx = [{(r["uid"], r["peak"]): r for r in views[v]} for v in vs]
        for how in (("prob",) if name == "base" else ("prob", "logit")):
            rows = [dict(r, p=float(agg([[d[key]["p"]] for d in idx], how)[0])) for key, r in idx[0].items()
                    if all(key in d for d in idx)]
            out[name if name == "base" else f"{name}/{how}"] = lung_metrics(rows, uids=uids)
    return out


def decide(table, crit, second):
    base = table.get("base")
    cands = {k: v for k, v in table.items() if k not in ("base", "base (stored file)")}
    if base is None or not cands:
        return "incomplete"
    best = max(cands, key=lambda k: (cands[k][crit], cands[k][second]))
    adopt = cands[best][crit] > base[crit]
    return (f"best TTA {best}: {crit} {cands[best][crit]:.4f} vs base {base[crit]:.4f} -> "
            f"{'ADOPT TTA' if adopt else 'keep base (TTA does not beat it)'}")


def show(title, table):
    print(title)
    for k, v in table.items():
        print(f"  {k:22s} " + "  ".join(f"{m} {x:.4f}" if isinstance(x, float) else f"{m} {x}" for m, x in v.items()))


if __name__ == "__main__":
    which = sys.argv[1]
    res = {}
    if which == "val":
        res["cxr_val"] = cxr("val")
        show("Radiographs, held-out tenth (wmu_cxdet_val), K=60 max-p, mean over 4 findings", res["cxr_val"])
        print("  decision:", decide(res["cxr_val"], "AUROC", "LROC"))
        res["lung_val"] = lung("val")
        show("Lung CT, out-of-fold pool scans with a nodule >= 6 mm, K=10", res["lung_val"])
        print("  decision:", decide(res["lung_val"], "hit_wmxv", "cand_AUROC"))
    else:
        res["cxr_test"] = cxr("test")
        show("Radiographs, ChestX-Det test, K=60 max-p, mean over 4 findings", res["cxr_test"])
        for s in ("eval_60_160", "eval_160_260"):
            res[f"lung_{s}"] = lung(s)
            show(f"Lung CT, {s} (Set {'1' if s == 'eval_60_160' else '2'}), K=10", res[f"lung_{s}"])
    json.dump(res, open(f"results/tta_{which}.json", "w"), indent=1)
