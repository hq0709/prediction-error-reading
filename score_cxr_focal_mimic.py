"""MIMIC-CXR focal findings from readers trained on ChestX-Det boxes, against the report labels of the official test
split (uncertain labels left out, as sota_cxr/score.py).

  focal reader (train_cxr_focal_reader.py, results/cxr_focal/<run>/wm_mimic_<split>.jsonl): per image, "Lung Lesion" =
      the maximum P(nodule or mass[, calcified granuloma]) over the world model's top-K addresses, "Fracture" = the
      maximum P(fracture)
  baselines trained on the same boxes (train_cxr_detector.py, train_cxr_yolo.py, train_cxr_oneshot.py --score-mimic,
      results/cxr_focal/<run>/mimic_<split>.jsonl): the same maxima over their one row per image and finding (top box
      score, or P(yes) of the whole-image answer)

Each is also combined with a whole-image classifier trained on the MIMIC report labels: the mean of the two
rank-normalised scores with weight w on the classifier. Every choice (which findings make a lesion, K, w) is made on the
validation split.

    python score_cxr_focal_mimic.py densenet121_60k cxr_focal cxr_detector cxr_oneshot
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from score_cxr_focal import auroc

ROOT = "/raid/home/CAMCA/hj880/perr"
FINDINGS = {"Lung Lesion": ("lesion_nm", "lesion_nmc"), "Fracture": ("fracture",)}


def read(path):
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    return d[d.src == "wm"]


def union(a, b, k=30, nms=16):
    """Both world models' addresses taken in turn (512 first), repeats within nms px dropped: the rows of
    cxr_focal_data.py --proposer wmu rebuilt from the scored rows of each world model."""
    cols = ["r", "c"] + [f"p_{t}" for t in ("Nodule", "Mass", "Calcification", "Fracture")]
    ga = {key: g.sort_values("rank")[cols].to_numpy() for key, g in a[a["rank"] < k].groupby("key")}
    gb = {key: g.sort_values("rank")[cols].to_numpy() for key, g in b[b["rank"] < k].groupby("key")}
    out = []
    for key, xa in ga.items():
        xb = gb.get(key, np.zeros((0, len(cols))))
        kept = []
        for i in range(max(len(xa), len(xb))):
            for x in (xa, xb):
                if i < len(x) and all(max(abs(x[i, 0] - q[0]), abs(x[i, 1] - q[1])) > nms for q in kept):
                    kept.append(x[i])
        for j, x in enumerate(kept[:2 * k]):
            out.append({"key": key, "rank": j, "src": "wm", **dict(zip(cols, x))})
    return pd.DataFrame(out)


def image_scores(d, ks=(10, 20, 30)):
    d["nm"] = np.maximum(d.p_Nodule, d.p_Mass)
    d["nmc"] = np.maximum(d.nm, d.p_Calcification)
    out = {}
    for k in ks:
        g = d[d["rank"] < k].groupby("key")
        out[("lesion_nm", k)] = g.nm.max()
        out[("lesion_nmc", k)] = g.nmc.max()
        out[("fracture", k)] = g.p_Fracture.max()
    return pd.DataFrame(out)


def ranks(v):
    return pd.Series(v).rank(pct=True).to_numpy()


def labels(split):
    d = pd.read_csv(f"{ROOT}/results/cxr_mimic.csv")
    return d[d.split == split].set_index("key")


def run_sets(run):
    """(label, validation rows, test rows, K choices) for each address set of the run found on disk."""
    f = lambda pre, s: f"{ROOT}/results/cxr_focal/{run}/{pre}mimic_{s}.jsonl"  # noqa: E731
    have = lambda pre: all(os.path.exists(f(pre, s)) for s in ("validate", "test"))  # noqa: E731
    out = []
    if have(""):                                                   # baselines: one row per image and finding
        out.append((run, read(f("", "validate")), read(f("", "test")), (10,)))
    if have("wm_"):
        va, te = read(f("wm_", "validate")), read(f("wm_", "test"))
        out.append((f"{run} (512 world model)", va, te, (10, 20, 30)))
        if have("wmh_") and not have("wmu_"):
            ha, ht = read(f("wmh_", "validate")), read(f("wmh_", "test"))
            out.append((f"{run} (both world models)", union(va, ha), union(te, ht), (30, 60)))
    if have("wmu_"):                                               # union rows scored directly (make_wmu_mimic.py)
        out.append((f"{run} (both world models)", read(f("wmu_", "validate")), read(f("wmu_", "test")), (30, 60)))
    for pre, label in (("random_", "random thorax points"), ("wmr_", "512 world model, whole radiograph"),
                       ("wmn_", "512 world model trained on NIH normals"),
                       ("wma_", "512 world model trained on MIMIC images of any label")):
        if have(pre):                                              # ablations: other address sources, same reader
            out.append((f"{run} ({label})", read(f(pre, "validate")), read(f(pre, "test")), (10, 20, 30)))
    return out


def wm_alone():
    """The 512 world model without a reader: its error at the best of its addresses, the same score for every finding."""
    out = []
    for s in ("validate", "test"):
        d = pd.DataFrame([json.loads(l) for l in open(f"{ROOT}/data/cxr_focal/wm/mimic_{s}.jsonl")])
        d = d[d.src == "wm"]
        for t in ("Nodule", "Mass", "Calcification", "Fracture"):
            d[f"p_{t}"] = d["wm"]
        out.append(d)
    return ("512 world model alone (no reader)", out[0], out[1], (30,))


def main():
    args = sys.argv[1:]
    glob = next(a for a in args if os.path.exists(f"{ROOT}/results/sota_cxr/{a}_validate.csv"))
    runs = [a for a in args if a != glob]
    gva = pd.read_csv(f"{ROOT}/results/sota_cxr/{glob}_validate.csv").set_index("key")
    gte = pd.read_csv(f"{ROOT}/results/sota_cxr/{glob}.csv").set_index("key")
    lva, lte = labels("validate"), labels("test")
    sets = [s for run in runs if run != "wm_alone" for s in run_sets(run)] + ([wm_alone()] if "wm_alone" in runs else [])
    rows = []
    for finding, kinds in FINDINGS.items():
        def ev(scores, glob_s, lab, kind, k, w):
            keys = [x for x in lab.index if x in scores.index and x in glob_s.index]
            y = lab.loc[keys, finding]
            keys = [x for x, o in zip(keys, y.isin([0, 1]) | y.isna()) if o]
            y = lab.loc[keys, finding].fillna(0).to_numpy() == 1
            s = (1 - w) * ranks(scores.loc[keys, (kind, k)]) + w * ranks(glob_s.loc[keys, finding])
            return auroc(s, y), int(y.sum())
        for n, (label, dva, dte, ks) in enumerate(sets):
            va, te = image_scores(dva, ks), image_scores(dte, ks)
            best = max(((kind, k) for kind in kinds for k in ks), key=lambda c: ev(va, gva, lva, *c, 0.0)[0])
            wbest = max(np.linspace(0, 1, 11), key=lambda w: ev(va, gva, lva, *best, w)[0])
            if n == 0:
                rows.append({"finding": finding, "reader": f"whole-image {glob}", "config": "",
                             "alone val": ev(va, gva, lva, *best, 1.0)[0], "alone test": ev(te, gte, lte, *best, 1.0)[0],
                             "test pos": ev(te, gte, lte, *best, 1.0)[1]})
            rows.append({"finding": finding, "reader": label, "config": f"{best[0]} K={best[1]}",
                         "alone val": ev(va, gva, lva, *best, 0.0)[0], "alone test": ev(te, gte, lte, *best, 0.0)[0],
                         "w": wbest, "combined val": ev(va, gva, lva, *best, wbest)[0],
                         "combined test": ev(te, gte, lte, *best, wbest)[0],
                         "combined test, w=0.5": ev(te, gte, lte, *best, 0.5)[0]})
    pd.set_option("display.width", 240)
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f", na_rep=""))
    print("\nfixed rules (no choice on validation): lesion = nodule or mass, every address read, w = 0.5")
    fixed = []
    for label, dva, dte, ks in [(f"whole-image {glob}", None, None, None)] + sets:
        row = {"reader": label}
        for split, d, gs, lab in (("val", dva, gva, lva), ("test", dte, gte, lte)):
            sc = image_scores(d, (max(ks),)) if d is not None else None
            for finding, kind in (("Lung Lesion", "lesion_nm"), ("Fracture", "fracture")):
                keys = [x for x in lab.index if x in gs.index and (sc is None or x in sc.index)]
                y = lab.loc[keys, finding]
                keys = [x for x, o in zip(keys, y.isin([0, 1]) | y.isna()) if o]
                y = lab.loc[keys, finding].fillna(0).to_numpy() == 1
                g = ranks(gs.loc[keys, finding])
                if sc is None:
                    row[f"{finding} {split}"] = auroc(g, y)
                    continue
                r = ranks(sc.loc[keys, (kind, max(ks))])
                row[f"{finding} {split}"] = auroc(r, y)
                row[f"{finding} {split} +glob"] = auroc(0.5 * r + 0.5 * g, y)
        fixed.append(row)
    cols = ["reader"] + [f"{f} {s}{c}" for s in ("val", "test") for f in ("Lung Lesion", "Fracture") for c in ("", " +glob")]
    print(pd.DataFrame(fixed).reindex(columns=cols).to_string(index=False, float_format="%.3f", na_rep=""))


if __name__ == "__main__":
    main()
