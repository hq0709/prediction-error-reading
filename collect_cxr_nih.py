"""External pointing test on NIH ChestX-ray14's radiologist boxes (nodule, mass; boxes are not exhaustive, so only the
pointing hit among the boxed images is reported): readers and detectors trained on ChestX-Det, applied unchanged.

    python collect_cxr_nih.py
"""
import json
import os

import pandas as pd

from choose_cxr_aug import chosen
from cxr_recall import boxes
from score_cxr_focal import evaluate

R = "results/cxr_focal"
TYPES = ("Nodule", "Mass")
ROWS = [  # label, file, K
    ("Faster R-CNN", f"{R}/{chosen('Faster R-CNN')}/nih_test.jsonl", 1),       # augmentation: choose_cxr_aug.py
    ("RetinaNet", f"{R}/{chosen('RetinaNet')}/nih_test.jsonl", 1),
    ("FCOS", f"{R}/{chosen('FCOS')}/nih_test.jsonl", 1),
    ("YOLO11m", f"{R}/cxr_yolo11m/nih_test.jsonl", 1),
    ("RT-DETRv2-R50", f"{R}/{chosen('RT-DETRv2-R50')}/nih_test.jsonl", 1),
    ("Deformable DETR", f"{R}/{chosen('Deformable DETR')}/nih_test.jsonl", 1),
    ("one-shot Qwen3-VL-8B (answer + point)", f"{R}/{chosen('one-shot Qwen3-VL-8B')}/nih_test.jsonl", 1),
    ("both world models + reader trained on their addresses", f"{R}/{chosen('PER (ours)')}/wmu_nih_test.jsonl", 60),
    ("  ... + MIMIC report labels (silence negative)", f"{R}/cxr_focal_wmu_mil/wmu_nih_test.jsonl", 60),
    ("  ... + MIMIC report labels (stated findings)", f"{R}/cxr_focal_wmu_milpos/wmu_nih_test.jsonl", 60),
    ("  ... + MIMIC report labels (stated and seen)", f"{R}/cxr_focal_wmu_milgate/wmu_nih_test.jsonl", 60),
    ("world-model addresses + reader", f"{R}/cxr_focal/wm_nih_test.jsonl", 30),
    ("PatchCore addresses + same reader", f"{R}/cxr_focal/patchcore_nih_test.jsonl", 30),
    ("random thorax addresses + same reader", f"{R}/cxr_focal/random_nih_test.jsonl", 30),
]


def main():
    b = boxes("nih")
    out = []
    for label, path, k in ROWS:
        if not os.path.exists(path):
            continue
        res = evaluate([json.loads(l) for l in open(path)], b, TYPES, ks=(k,), rules=("max-p",)).set_index("finding")
        out.append({"method": label, **{f"{t} hit": res.loc[t, "hit"] for t in TYPES},
                    "boxed images": "/".join(str(int((b.finding == t).sum())) for t in TYPES)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(out).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
