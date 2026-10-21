"""Print the full-data PER verifier run kept by choose_cxr_aug.py: the highest image-level mean AUROC on the held-out
tenth among the given runs (default: the two-epoch runs without and with augmentation).

    python per_aug_guard.py                       # exit 3 when the run without augmentation is kept (nothing to re-score)
    python per_aug_guard.py RUN [RUN ...]         # always exit 0: every given run is new
"""
import sys

from choose_cxr_aug import verifier

RUNS = tuple(sys.argv[1:]) or ("cxr_focal_wmu", "cxr_focal_wmuaug")
val = {r: v for r in RUNS if (v := verifier(r)) is not None}
best = max(val, key=val.get)
print("held-out image-level AUROC: " + ", ".join(f"{r} {v:.4f}" for r, v in val.items()) + f" -> {best}", file=sys.stderr)
print(best)
sys.exit(0 if len(sys.argv) > 1 or best != RUNS[0] else 3)
