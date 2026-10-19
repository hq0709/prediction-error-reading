"""Label-efficiency curve on the ChestX-Det focal findings (nodule, mass, calcification, fracture; test split): every
method trained on the same subsets of the boxed training images (results/cxr_cxdet_boxes<b>_keys.txt, all normal
images kept) and scored like collect_cxr_focal.py (mean over the four findings of presence AUROC, pointing hit, LROC).
Our reader reads the 60 addresses of both radiograph world models, which never saw a box at any budget.

    python collect_cxr_curve.py
Writes results/cxr_curve.csv.
"""
import json
import os

import pandas as pd

from cxr_recall import boxes
from score_cxr_focal import TYPES, evaluate

R = "results/cxr_focal"
BUDGETS = (1, 5, 10, 25, 50, 100)
METHODS = [  # label, run name (budget 100 drops the suffix), file, K
    ("Faster R-CNN", "cxr_detector", "cxdet_test.jsonl", 1),
    ("RetinaNet", "cxr_retinanet", "cxdet_test.jsonl", 1),
    ("FCOS", "cxr_fcos", "cxdet_test.jsonl", 1),
    ("YOLO11m", "cxr_yolo11m", "cxdet_test.jsonl", 1),
    ("RT-DETRv2-R50", "cxr_rtdetrv2", "cxdet_test.jsonl", 1),
    ("Deformable DETR", "cxr_deformdetr", "cxdet_test.jsonl", 1),
    ("one-shot Qwen3-VL-8B", "cxr_oneshot", "cxdet_test.jsonl", 1),
    ("PER (ours)", "cxr_focal_wmu", "wmu_cxdet_test.jsonl", 60),
]


def main():
    b = boxes("cxdet")
    b = b[b.split == "test"]
    out = []
    for label, run, fn, k in METHODS:
        for bud in BUDGETS:
            path = f"{R}/{run}{'' if bud == 100 else f'_b{bud}'}/{fn}"
            if not os.path.exists(path):
                continue
            rows = [json.loads(l) for l in open(path)]
            res = evaluate(rows, b, TYPES, ks=(k,), rules=("max-p",))
            out.append({"method": label, "boxes %": bud, **{m: res[m].mean() for m in ("AUROC", "hit", "LROC")}})
    df = pd.DataFrame(out)
    df.to_csv("results/cxr_curve.csv", index=False)
    pd.set_option("display.width", 220)
    for m in ("LROC", "hit", "AUROC"):
        t = df.pivot(index="method", columns="boxes %", values=m).reindex([x[0] for x in METHODS])
        print(f"\n{m}\n" + t.to_string(float_format="%.3f"))


if __name__ == "__main__":
    main()
