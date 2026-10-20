"""Findings nobody labelled: on ChestX-Det test radiographs, the 9 finding types that no method was trained on (every
reader and detector here learned only nodule, mass, calcification and fracture boxes) — consolidation, effusion,
fibrosis, pleural thickening, cardiomegaly, emphysema, diffuse nodule, atelectasis, pneumothorax. A method points at an
unseen finding when one of its first K addresses lies inside a box of that finding (images with such a box).

  world model: the addresses of both radiograph world models (512 and 1024 grids, trained on normal radiographs only),
      ranked by precision-weighted prediction error; one-sided (brighter than expected, the focal-reader setting) and
      two-sided (departures of either sign, e.g. pneumothorax and emphysema are darker than expected)
  detectors / one-shot reader trained on the 4 focal finding types: their top box per trained type, ranked by score
  random thorax points: chance

    python open_set_cxr.py
Writes results/open_set_cxr.csv.
"""
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from cxr_recall import boxes
from perr.cxr import CACHE
from perr.cxr_anat import search_domain
from perr.cxr_wm import error_map, peaks, valid_region

ROOT = "/raid/home/CAMCA/hj880/perr"
UNSEEN = ("Consolidation", "Effusion", "Fibrosis", "Pleural Thickening", "Cardiomegaly", "Emphysema", "Diffuse Nodule",
          "Atelectasis", "Pneumothorax")
KS = (1, 4, 10, 30)
DETECTORS = {"Faster R-CNN": "cxr_detector", "RetinaNet": "cxr_retinanet", "FCOS": "cxr_fcos", "YOLO11m": "cxr_yolo11m",
             "RT-DETRv2-R50": "cxr_rtdetrv2", "Deformable DETR": "cxr_deformdetr", "one-shot Qwen3-VL-8B": "cxr_oneshot"}


def union_addresses(key, sign, k=30, nms=16):
    img = np.load(f"{CACHE}/{key}.npy")
    D = search_domain(key, valid_region(img))
    lists = []
    for t in ("cxr", "cxrh"):
        R = np.load(f"{ROOT}/cache/cxr_wm/{key}.{t}R.npy").astype(np.float32)
        S = np.load(f"{ROOT}/cache/cxr_wm/{key}.{t}S.npy").astype(np.float32)
        lists.append(peaks(error_map(R, S, 2.0, sign), D, k=k, nms=nms))
    out = []
    for a, b in zip(*lists):
        for p in (a, b):
            if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > nms for q in out):
                out.append(p)
    return key, [(r, c) for r, c, _ in out[:2 * k]]


def random_addresses(key, k=60, nms=16):
    img = np.load(f"{CACHE}/{key}.npy")
    D = search_domain(key, valid_region(img))
    W = np.random.default_rng(abs(hash(key)) % 2 ** 32).random(D.shape).astype(np.float32)
    return key, [(r, c) for r, c, _ in peaks(W, D, k=k, nms=nms)]


def hit_at(addr, bx, k):
    return any(any(b.x <= c <= b.x + b.w and b.y <= r <= b.y + b.h for b in bx) for r, c in addr[:k])


def main():
    b = boxes("cxdet")
    b = b[b.split == "test"]
    keys = sorted(b.key.unique())
    by = {u: {k: list(g.itertuples()) for k, g in b[b.finding == u].groupby("key")} for u in UNSEEN}
    methods = {}
    with Pool(24) as p:
        methods["world model, one-sided (focal setting)"] = dict(p.starmap(union_addresses, [(k, 1) for k in keys]))
        methods["world model, two-sided"] = dict(p.starmap(union_addresses, [(k, 0) for k in keys]))
        methods["random thorax points"] = dict(p.map(random_addresses, keys))
    for label, run in DETECTORS.items():
        dump = f"{ROOT}/results/cxr_boxes/{run}.json"            # every box (dump_cxr_boxes.py), else top box per type
        if os.path.exists(dump):
            methods[label] = {k: [(r, c) for _, r, c in v] for k, v in json.load(open(dump)).items()}
            continue
        rows = [json.loads(l) for l in open(f"{ROOT}/results/cxr_focal/{run}/cxdet_test.jsonl")]
        per = defaultdict(list)
        for r in rows:
            per[r["key"]].append((max(v for k_, v in r.items() if k_.startswith("p_")), r["r"], r["c"]))
        methods[label] = {k: [(r, c) for _, r, c in sorted(v, reverse=True)] for k, v in per.items()}
    out = []
    for size, cap in (("all boxes", 1.0), ("boxes < 5% of the image", 0.05), ("boxes < 2% of the image", 0.02)):
        for label, addr in methods.items():
            for u in UNSEEN:
                bu = b[(b.finding == u) & (b.w * b.h / 512 ** 2 < cap)]
                imgs = {k: list(g.itertuples()) for k, g in bu.groupby("key")}
                if not imgs:
                    continue
                for k in KS:
                    h = [hit_at(addr.get(key, []), bx, k) for key, bx in imgs.items()]
                    out.append({"boxes": size, "method": label, "finding": u, "K": k, "images": len(imgs), "hit": float(np.mean(h))})
    df = pd.DataFrame(out)
    df.to_csv(f"{ROOT}/results/open_set_cxr.csv", index=False)
    pd.set_option("display.width", 250)
    for size in df.boxes.unique():
        d = df[df.boxes == size]
        print(f"\npointing at unseen finding types ({size}), mean over types, hit within the first K addresses")
        print(d.groupby(["method", "K"]).hit.mean().unstack("K").round(3).to_string())
    print("\nper finding type at K = 10, all boxes")
    d = df[(df.boxes == "all boxes") & (df.K == 10)]
    print(d.pivot(index="method", columns="finding", values="hit").round(2).to_string())


if __name__ == "__main__":
    main()
