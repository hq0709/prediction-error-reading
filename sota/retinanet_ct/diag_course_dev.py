"""Development-pair diagnostic of a longitudinal RetinaNet checkpoint (development labels only): where does the top box
land (target / other lesion / nothing, PER's grown maps), and at which rank does the first box on a target appear?

    CUDA_VISIBLE_DEVICES=0 venvs/clipvenv/bin/python diag_course_dev.py runs_det/rc_scratch [--channels 2] [--ckpt model.pt]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import train_ct_retinanet_course as m  # noqa: E402
from perr.autopet import load_pair  # noqa: E402
from perr.course_candidates import label_maps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--channels", type=int, default=2)
ap.add_argument("--ckpt", default="model.pt")
a = ap.parse_args()
m.CHANNELS = a.channels
dev = torch.device("cuda")
det = m.make_detector(dev)
det.network.load_state_dict(torch.load(f"{a.run}/{a.ckpt}", map_location=dev))
keys = m.pair_keys(m.DEV_ROWS)
top_where, first_rank, out = {"target": 0, "lesion": 0, "none": 0}, [], {}
for k in keys:
    s = m.Scan(k)
    rows = m.detect(det, s, dev)
    T, A = label_maps(load_pair(k), m.MIN_DIAM)

    def where(z):
        i = tuple(int(round(c)) for c in z)
        if not all(0 <= i[d] < T.shape[d] for d in range(3)):
            return "none"
        return "target" if T[i] else ("lesion" if A[i] else "none")
    labels = [where(r["zyx"]) for r in rows[:100]]
    has_t = bool(T.any())
    out[k] = {"has_target": has_t, "top": labels[:10], "scores": [round(r["score"], 3) for r in rows[:10]],
              "sizes": [round(r["size_mm"], 1) for r in rows[:10]]}
    if has_t:
        top_where[labels[0] if labels else "none"] += 1
        first_rank.append(labels.index("target") + 1 if "target" in labels else 999)
    print(k, "target" if has_t else "-", labels[:10], out[k]["scores"][:5], flush=True)
fr = np.array(first_rank)
print(f"pairs with a target {len(fr)}: top box on {top_where}; first target box rank {sorted(fr.tolist())}; "
      f"recall@10 {np.mean(fr <= 10):.2f} @100 {np.mean(fr <= 100):.2f}")
print("top box on any lesion, pairs without a target:",
      sum(v["top"][:1] == ["lesion"] for v in out.values() if not v["has_target"]), "of",
      sum(not v["has_target"] for v in out.values()))
json.dump(out, open(f"{a.run}/diag_dev_{a.ckpt.replace('.pt', '')}.json", "w"), indent=1)
