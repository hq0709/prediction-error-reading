"""Supervised detector baseline for focal radiograph findings: Faster R-CNN (ResNet-50 FPN v2, COCO weights, torchvision)
fine-tuned on the ChestX-Det training boxes of nodule, mass, calcification and fracture, on the 512 canvas (perr/cxr.py)
with flips off (laterality matters little here, but the boxes are left as annotated). Same training images and
held-out tenth as the focal reader; the epoch is chosen on that tenth by mean image-level AUROC.

Test rows are written in the focal reader's format (per image and finding: the highest-scoring box centre and its
score), so score_cxr_focal.py scores it like the readers.

    CUDA_VISIBLE_DEVICES=2 python train_cxr_detector.py --out runs_reader/cxr_detector
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from cxr_recall import boxes
from perr.cxr import CACHE

TYPES = ("Nodule", "Mass", "Calcification", "Fracture")


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sample(key, by):
    x = torch.from_numpy(np.load(f"{CACHE}/{key}.npy")).float().div(255)[None].expand(3, -1, -1).contiguous()
    bb = by.get(key, [])
    t = {"boxes": torch.tensor([[b["x"], b["y"], b["x"] + max(b["w"], 2), b["y"] + max(b["h"], 2)] for b in bb], dtype=torch.float32).view(-1, 4),
         "labels": torch.tensor([TYPES.index(b["finding"]) + 1 for b in bb], dtype=torch.int64)}
    return x, t


@torch.no_grad()
def predict(model, keys, dev, bs=16):
    model.eval()
    out = {}
    for i in range(0, len(keys), bs):
        ks = keys[i:i + bs]
        xs = [torch.from_numpy(np.load(f"{CACHE}/{k}.npy")).float().div(255)[None].expand(3, -1, -1).to(dev) for k in ks]
        for k, d in zip(ks, model(xs)):
            out[k] = {}
            for j, t in enumerate(TYPES):
                m = d["labels"] == j + 1
                if m.any():
                    s, b = d["scores"][m].max(0)
                    bx = d["boxes"][m][b]
                    out[k][t] = (float(s), float((bx[0] + bx[2]) / 2), float((bx[1] + bx[3]) / 2))
                else:
                    out[k][t] = (0.0, 256.0, 256.0)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda", 0)
    g = pd.read_csv("results/cxr_cxdet.csv")
    b = boxes("cxdet")
    b = b[b.finding.isin(TYPES)]
    by = {k: v[["finding", "x", "y", "w", "h"]].to_dict("records") for k, v in b.groupby("key")}
    fkeys = sorted({json.loads(l)["key"] for l in open("data/cxr_focal/wm/cxdet_train.jsonl")})
    random.Random(0).shuffle(fkeys)
    val_keys = fkeys[:len(fkeys) // 10]
    tr_keys = [k for k in g[g.split == "train"].key if k not in set(val_keys)]
    test_keys = sorted(g[g.split == "test"].key)
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, min_size=512, max_size=512,
                                       box_detections_per_img=100, box_score_thresh=0.001)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, len(TYPES) + 1)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    steps = a.epochs * (len(tr_keys) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    best = -1.0
    for ep in range(a.epochs):
        random.shuffle(tr_keys)
        tot = []
        for i in range(0, len(tr_keys) - a.bs + 1, a.bs):
            batch = [sample(k, by) for k in tr_keys[i:i + a.bs]]
            xs = [x.to(dev) for x, _ in batch]
            ts = [{k: v.to(dev) for k, v in t.items()} for _, t in batch]
            loss = sum(model(xs, ts).values())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot.append(loss.item())
        pr = predict(model, val_keys, dev)
        v = np.nanmean([auroc([pr[k][t][0] for k in val_keys], [int(any(x["finding"] == t for x in by.get(k, []))) for k in val_keys]) for t in TYPES])
        print(f"epoch {ep} loss {np.mean(tot):.3f} validation mean AUROC {v:.3f}", flush=True)
        if v > best:
            best = v
            torch.save(model.state_dict(), f"{a.out}/best.pt")
    model.load_state_dict(torch.load(f"{a.out}/best.pt"))
    pr = predict(model, test_keys, dev)
    os.makedirs("results/cxr_focal/cxr_detector", exist_ok=True)
    with open("results/cxr_focal/cxr_detector/cxdet_test.jsonl", "w") as f:
        for k in test_keys:
            for j, t in enumerate(TYPES):
                s, x, y = pr[k][t]
                f.write(json.dumps({"key": k, "i": j, "r": y, "c": x, "wm": 1.0, "src": "wm", "rank": 0,
                                    **{f"p_{u}": (s if u == t else 0.0) for u in TYPES}}) + "\n")
    print(f"best validation {best:.3f}; test rows written", flush=True)


if __name__ == "__main__":
    main()
