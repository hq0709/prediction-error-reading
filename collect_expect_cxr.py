"""Kept radiograph verifier (two-panel views) against the verifier that also sees the world model's expectation (2 x 2
views, render_views_expect.py), on the same wmu candidates: image-level mean over nodule, mass, calcified granuloma and
fracture of presence AUROC, pointing hit and LROC with max-p over every candidate (score_cxr_focal.evaluate). The
validation set (held-out tenth of ChestX-Det train) decides; test is reported after.

    python collect_expect_cxr.py val          # decision split
    python collect_expect_cxr.py test
"""
import json
import sys

from cxr_recall import boxes
from score_cxr_focal import TYPES, evaluate

RUNS = [("kept verifier (two-panel views)", "results/cxr_focal/cxr_focal_e3/wmu_{s}.jsonl"),
        ("verifier with expectation (2 x 2 views)", "results/cxr_focal/cxr_focal_expect/wmu_x_{s}.jsonl")]


def main():
    split = sys.argv[1]
    s = {"val": "cxdet_val", "test": "cxdet_test"}[split]
    b = boxes("cxdet")
    for label, path in RUNS:
        rows = [json.loads(line) for line in open(path.format(s=s))]
        keys = {r["key"] for r in rows}
        res = evaluate(rows, b[b.key.isin(keys)], TYPES, ks=(1000,), rules=("max-p",))
        per = " ".join(f"{t[:4]} {a:.3f}" for t, a in zip(res.finding, res.AUROC))
        print(f"{split} {label}: {len(keys)} images, AUROC {res.AUROC.mean():.3f} hit {res.hit.mean():.3f} "
              f"LROC {res.LROC.mean():.3f}  ({per})")


if __name__ == "__main__":
    main()
