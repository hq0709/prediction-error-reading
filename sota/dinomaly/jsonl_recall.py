"""Focal-box recall@K of the candidates written by cxr_focal_data.py (data/cxr_focal/<name>/cxdet_<split>.jsonl): the
share of Nodule / Mass / Calcification / Fracture boxes (cxr_recall.boxes("cxdet")) of the listed images that contain
one of the image's top-K candidates. Used for the reported held-out-tenth recall@45 of a proposer.

    python -I -B sota/madad/jsonl_recall.py data/cxr_focal/madad_dev/cxdet_train.jsonl
"""
import json
import sys

import numpy as np

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(0, ROOT)
from cxr_recall import boxes          # noqa: E402

TYPES = ("Nodule", "Mass", "Calcification", "Fracture")
KS = (10, 30, 45)


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    cand = {}
    for r in rows:
        if r["src"] == "wm":
            cand.setdefault(r["key"], []).append((r["rank"], r["r"], r["c"]))
    cand = {k: [(rr, c) for _, rr, c in sorted(v)] for k, v in cand.items()}
    b = boxes("cxdet")
    b = b[b.key.isin(set(cand)) & b.finding.isin(TYPES)]
    out = {"images": len(cand), "boxes": int(len(b)), "candidates_per_image": float(np.mean([len(v) for v in cand.values()]))}
    for k in KS:
        hit = [any(r.x <= c <= r.x + r.w and r.y <= rr <= r.y + r.h for rr, c in cand[r.key][:k]) for r in b.itertuples()]
        out[f"recall@{k}"] = float(np.mean(hit))
        if k == 45:
            for t in TYPES:
                sel = (b.finding == t).to_numpy()
                out[f"{t}@45"] = float(np.mean(np.array(hit)[sel])) if sel.any() else float("nan")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
