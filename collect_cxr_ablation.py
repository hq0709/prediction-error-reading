"""Ablation of the focal radiograph reader, one component at a time: ChestX-Det test (mean over nodule / mass /
calcification / fracture: AUROC, pointing hit, LROC) and MIMIC-CXR official test (report labels, AUROC of the reader
alone; lesion = nodule or mass; every address read).

    python collect_cxr_ablation.py
"""
import json
import os

import pandas as pd

from cxr_recall import boxes
from score_cxr_focal import evaluate
from score_cxr_focal_mimic import auroc, image_scores, labels, read

R = "results/cxr_focal"
D = "data/cxr_focal"
TYPES = ("Nodule", "Mass", "Calcification", "Fracture")
COMP = ("WM addresses", "thorax domain", "verifier", "1024 WM", "trained on union", "report labels")
# label, components, ChestX-Det rows, K, MIMIC test rows (or None)
ROWS = [
    ("random thorax points + reader", "011000", f"{R}/cxr_focal/random_cxdet_test.jsonl", 30,
     f"{R}/cxr_focal/random_mimic_test.jsonl"),
    ("WM addresses over the whole radiograph + reader", "101000", f"{R}/cxr_focal/wmr_cxdet_test.jsonl", 30,
     f"{R}/cxr_focal/wmr_mimic_test.jsonl"),
    ("WM alone (error at its best address)", "110000", "wm_alone", 30, "wm_alone"),
    ("512 WM + reader", "111000", f"{R}/cxr_focal/wm_cxdet_test.jsonl", 30, f"{R}/cxr_focal/wm_mimic_test.jsonl"),
    ("both WMs + same reader", "111100", f"{R}/cxr_focal/wmu_cxdet_test.jsonl", 60, "union:cxr_focal"),
    ("both WMs + reader trained on their addresses", "111110", f"{R}/cxr_focal_wmu/wmu_cxdet_test.jsonl", 60,
     f"{R}/cxr_focal_wmu/wmu_mimic_test.jsonl"),
    ("  + report labels, silence as negative", "111111", f"{R}/cxr_focal_wmu_mil/wmu_cxdet_test.jsonl", 60,
     f"{R}/cxr_focal_wmu_mil/wmu_mimic_test.jsonl"),
    ("  + report labels, stated findings", "111111", f"{R}/cxr_focal_wmu_milpos/wmu_cxdet_test.jsonl", 60,
     f"{R}/cxr_focal_wmu_milpos/wmu_mimic_test.jsonl"),
    ("  + report labels, stated and seen", "111111", f"{R}/cxr_focal_wmu_milgate/wmu_cxdet_test.jsonl", 60,
     f"{R}/cxr_focal_wmu_milgate/wmu_mimic_test.jsonl"),
]


def wm_rows(path):
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        for t in TYPES:
            r[f"p_{t}"] = r["wm"]
    return rows


def mimic(src, k):
    if src == "wm_alone":
        d = pd.DataFrame(wm_rows(f"{D}/wm/mimic_test.jsonl"))
        d = d[d.src == "wm"]
    elif src.startswith("union:"):
        from score_cxr_focal_mimic import union
        run = src.split(":")[1]
        d = union(read(f"{R}/{run}/wm_mimic_test.jsonl"), read(f"{R}/{run}/wmh_mimic_test.jsonl"))
    elif os.path.exists(src):
        d = read(src)
    else:
        return {}
    sc, lab = image_scores(d, (k,)), labels("test")
    out = {}
    for finding, kind in (("Lung Lesion", "lesion_nm"), ("Fracture", "fracture")):
        keys = [x for x in lab.index if x in sc.index]
        y = lab.loc[keys, finding]
        keys = [x for x, o in zip(keys, y.isin([0, 1]) | y.isna()) if o]
        out[f"MIMIC {finding}"] = auroc(sc.loc[keys, (kind, k)].to_numpy(), lab.loc[keys, finding].fillna(0).to_numpy() == 1)
    return out


def main():
    b = boxes("cxdet")
    out = []
    for label, comp, cx, k, mm in ROWS:
        if cx == "wm_alone":
            rows = wm_rows(f"{D}/wm/cxdet_test.jsonl")
        elif os.path.exists(cx):
            rows = [json.loads(l) for l in open(cx)]
        else:
            continue
        res = evaluate(rows, b, TYPES, ks=(k,), rules=("max-p",))
        out.append({"row": label, **{c: "✓" if f == "1" else "" for c, f in zip(COMP, comp)},
                    **{m: res[m].mean() for m in ("AUROC", "hit", "LROC")}, **mimic(mm, k)})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out).to_string(index=False, float_format="%.3f", na_rep="–"))


if __name__ == "__main__":
    main()
