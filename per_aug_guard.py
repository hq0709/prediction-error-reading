"""Print the full-data PER verifier run kept by choose_cxr_aug.py (with or without augmentation, whichever
has the higher image-level mean AUROC on the held-out tenth) and exit 0, or exit 3 when the
run without augmentation is kept, so the jobs that re-score every candidate set run only for a new verifier."""
import sys

from choose_cxr_aug import verifier

RUNS = ("cxr_focal_wmu", "cxr_focal_wmuaug")
val = {r: v for r in RUNS if (v := verifier(r)) is not None}
best = max(val, key=val.get)
print("held-out image-level AUROC: " + ", ".join(f"{r} {v:.4f}" for r, v in val.items()) + f" -> {best}", file=sys.stderr)
print(best)
sys.exit(0 if best != RUNS[0] else 3)
