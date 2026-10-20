"""One table for the ChestX-Det focal findings (nodule, mass, calcification, fracture; test split): every method's
mean over the four findings of presence AUROC, pointing hit and LROC (score_cxr_focal.evaluate), at K = 30 addresses
for the readers that read addresses (max-p and world model x reader), and K = 172 (every lattice place) for the
lattice ablation.

    python collect_cxr_focal.py
"""
import json
import os

import pandas as pd

from choose_cxr_aug import chosen
from cxr_recall import boxes
from score_cxr_focal import TYPES, evaluate

R = "results/cxr_focal"
ROWS = [  # label, file, K, rule
    ("Faster R-CNN (boxes)", f"{R}/{chosen('Faster R-CNN')}/cxdet_test.jsonl", 1, "max-p"),   # augmentation on/off:
    ("RetinaNet (boxes)", f"{R}/{chosen('RetinaNet')}/cxdet_test.jsonl", 1, "max-p"),         # choose_cxr_aug.py
    ("FCOS (boxes)", f"{R}/{chosen('FCOS')}/cxdet_test.jsonl", 1, "max-p"),
    ("YOLO11m (boxes)", f"{R}/cxr_yolo11m/cxdet_test.jsonl", 1, "max-p"),
    ("RT-DETRv2-R50 (boxes)", f"{R}/{chosen('RT-DETRv2-R50')}/cxdet_test.jsonl", 1, "max-p"),
    ("Deformable DETR (boxes)", f"{R}/{chosen('Deformable DETR')}/cxdet_test.jsonl", 1, "max-p"),
    ("one-shot Qwen3-VL-8B (answer + point)", f"{R}/{chosen('one-shot Qwen3-VL-8B')}/cxdet_test.jsonl", 1, "max-p"),
    ("both world models + reader trained on their addresses", f"{R}/{chosen('PER (ours)')}/wmu_cxdet_test.jsonl", 60, "max-p"),
    ("  ... + hard negatives", f"{R}/cxr_focal_wmu_hard/wmu_cxdet_test.jsonl", 60, "max-p"),
    ("  ... + MIMIC report labels (silence negative)", f"{R}/cxr_focal_wmu_mil/wmu_cxdet_test.jsonl", 60, "max-p"),
    ("  ... + MIMIC report labels (stated findings)", f"{R}/cxr_focal_wmu_milpos/wmu_cxdet_test.jsonl", 60, "max-p"),
    ("  ... + MIMIC report labels (stated and seen)", f"{R}/cxr_focal_wmu_milgate/wmu_cxdet_test.jsonl", 60, "max-p"),
    ("512 world model + reader with MIMIC report labels", f"{R}/cxr_focal_mil/wm_cxdet_test.jsonl", 30, "max-p"),
    ("both world models, 60 addresses + reader", f"{R}/cxr_focal/wmu_cxdet_test.jsonl", 60, "max-p"),
    ("  both world models, 30 addresses", f"{R}/cxr_focal/wmu_cxdet_test.jsonl", 30, "max-p"),
    ("  1024 world model, 30 addresses", f"{R}/cxr_focal/wmh_cxdet_test.jsonl", 30, "max-p"),
    ("world-model addresses + reader (max-p)", f"{R}/cxr_focal/wm_cxdet_test.jsonl", 30, "max-p"),
    ("world-model addresses + reader (WM x reader)", f"{R}/cxr_focal/wm_cxdet_test.jsonl", 30, "wm x reader"),
    ("  top 10 addresses (max-p)", f"{R}/cxr_focal/wm_cxdet_test.jsonl", 10, "max-p"),
    ("PatchCore addresses + same reader", f"{R}/cxr_focal/patchcore_cxdet_test.jsonl", 30, "max-p"),
    ("random thorax addresses + same reader", f"{R}/cxr_focal/random_cxdet_test.jsonl", 30, "max-p"),
    ("every lattice place + same reader (no proposer)", f"{R}/cxr_focal/grid_cxdet_test.jsonl", 1000, "max-p"),
    ("world-model addresses over the whole radiograph + same reader", f"{R}/cxr_focal/wmr_cxdet_test.jsonl", 30, "max-p"),
    ("Faster R-CNN, 25% of boxes", f"{R}/cxr_detector_b25/cxdet_test.jsonl", 1, "max-p"),
    ("reader, 25% of boxes", f"{R}/cxr_focal_b25/wm_cxdet_test.jsonl", 30, "max-p"),
    ("reader, 25% of boxes, both world models (60)", f"{R}/cxr_focal_b25/wmu_cxdet_test.jsonl", 60, "max-p"),
    ("one-shot, 25% of boxes", f"{R}/cxr_oneshot_b25/cxdet_test.jsonl", 1, "max-p"),
    ("YOLO11m, 25% of boxes", f"{R}/cxr_yolo11m_b25/cxdet_test.jsonl", 1, "max-p"),
    ("Faster R-CNN, 10% of boxes", f"{R}/cxr_detector_b10/cxdet_test.jsonl", 1, "max-p"),
    ("reader, 10% of boxes", f"{R}/cxr_focal_b10/wm_cxdet_test.jsonl", 30, "max-p"),
    ("reader, 10% of boxes, both world models (60)", f"{R}/cxr_focal_b10/wmu_cxdet_test.jsonl", 60, "max-p"),
    ("one-shot, 10% of boxes", f"{R}/cxr_oneshot_b10/cxdet_test.jsonl", 1, "max-p"),
    ("YOLO11m, 10% of boxes", f"{R}/cxr_yolo11m_b10/cxdet_test.jsonl", 1, "max-p"),
]


def main():
    b = boxes("cxdet")
    b = b[b.split == "test"]
    out = []
    for label, path, k, rule in ROWS:
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        res = evaluate(rows, b, TYPES, ks=(k,), rules=(rule,))
        per = res.set_index("finding")
        out.append({"method": label, **{m: res[m].mean() for m in ("AUROC", "hit", "LROC")},
                    **{f"{t[:4]} hit": per.loc[t, "hit"] for t in TYPES}})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(out).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
