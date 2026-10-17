"""MIMIC-CXR focal findings read from the world model's addresses (train_cxr_focal_reader.py, trained on ChestX-Det
boxes): per image, "Lung Lesion" = the reader's maximum P(nodule or mass[, calcified granuloma]) over the addresses and
"Fracture" = its maximum P(fracture), against the report labels of the official test split (uncertain labels left out,
as sota_cxr/score.py). Also the reader combined with a whole-image classifier: mean of the two rank-normalised scores
with weight w, every choice (which findings make a lesion, K, w) made on the validation split.

    python score_cxr_focal_mimic.py cxr_focal densenet121_60k
"""
import json
import sys

import numpy as np
import pandas as pd

from perr.cxr import FINDINGS  # noqa: F401
from score_cxr_focal import auroc

ROOT = "/raid/home/CAMCA/hj880/perr"


def image_scores(path, ks=(10, 20, 30)):
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    d = d[d.src == "wm"]
    out = {}
    for k in ks:
        e = d[d["rank"] < k]
        g = e.groupby("key")
        out[("lesion_nm", k)] = g.apply(lambda x: np.maximum(x.p_Nodule, x.p_Mass).max())
        out[("lesion_nmc", k)] = g.apply(lambda x: np.maximum(np.maximum(x.p_Nodule, x.p_Mass), x.p_Calcification).max())
        out[("fracture", k)] = g.p_Fracture.max()
    return pd.DataFrame(out)


def ranks(v):
    return pd.Series(v).rank(pct=True).to_numpy()


def labels(split):
    d = pd.read_csv(f"{ROOT}/results/cxr_mimic.csv")
    return d[d.split == split].set_index("key")


def main():
    name, glob = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "densenet121_60k")
    va = image_scores(f"{ROOT}/results/cxr_focal/{name}/wm_mimic_validate.jsonl")
    te = image_scores(f"{ROOT}/results/cxr_focal/{name}/wm_mimic_test.jsonl")
    gva = pd.read_csv(f"{ROOT}/results/sota_cxr/{glob}_validate.csv").set_index("key")
    gte = pd.read_csv(f"{ROOT}/results/sota_cxr/{glob}.csv").set_index("key")
    lva, lte = labels("validate"), labels("test")
    rows = []
    for finding, kinds in (("Lung Lesion", ("lesion_nm", "lesion_nmc")), ("Fracture", ("fracture",))):
        def ev(split_scores, glob_s, lab, kind, k, w):
            keys = [x for x in lab.index if x in split_scores.index and x in glob_s.index]
            y = lab.loc[keys, finding]
            ok = y.isin([0, 1]) | y.isna()
            keys = [x for x, o in zip(keys, ok) if o]
            y = lab.loc[keys, finding].fillna(0).to_numpy() == 1
            s = (1 - w) * ranks(split_scores.loc[keys, (kind, k)]) + w * ranks(glob_s.loc[keys, finding])
            return auroc(s, y), int(y.sum())
        best = max(((kind, k) for kind in kinds for k in (10, 20, 30)), key=lambda c: ev(va, gva, lva, *c, 0.0)[0])
        wbest = max(np.linspace(0, 1, 11), key=lambda w: ev(va, gva, lva, *best, w)[0])
        for label, w in (("focal reader", 0.0), (f"whole-image {glob}", 1.0), (f"combined (w={wbest:.1f})", wbest)):
            a_va, _ = ev(va, gva, lva, *best, w)
            a_te, npos = ev(te, gte, lte, *best, w)
            rows.append({"finding": finding, "reader": label, "config": f"{best[0]} K={best[1]}", "val": a_va, "test": a_te, "test pos": npos})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
