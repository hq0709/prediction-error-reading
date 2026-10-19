"""Transformer detector baselines for focal radiograph findings: RT-DETRv2 (ResNet-50-vd) or Deformable DETR (ResNet-50),
COCO weights from Hugging Face, fine-tuned on the same ChestX-Det training boxes, held-out tenth and 512 canvas as
train_cxr_detector.py (no augmentation, epoch chosen on the held-out tenth by mean image-level AUROC), with the
optimiser of their papers (AdamW, lr 1e-4, backbone 1e-5, weight decay 1e-4, gradient clipping 0.1).

Test rows are written in the focal reader's format (per image and finding: the highest-scoring box centre and its
score); also MIMIC validate/test and NIH, as train_cxr_detector.score_sets.

    CUDA_VISIBLE_DEVICES=6 python train_cxr_detr.py --model PekingU/rtdetr_v2_r50vd --out runs_reader/cxr_rtdetrv2
    CUDA_VISIBLE_DEVICES=6 python train_cxr_detr.py --model SenseTime/deformable-detr --out runs_reader/cxr_deformdetr
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from cxr_balance import balanced
from cxr_recall import boxes
from perr.cxr import CACHE
from train_cxr_detector import TYPES, auroc


def images(keys):
    return [np.repeat(np.load(f"{CACHE}/{k}.npy")[..., None], 3, -1) for k in keys]


def targets(keys, by):
    out = []
    for k in keys:
        bb = by.get(k, [])
        xywh = np.array([[b["x"], b["y"], max(b["w"], 2), max(b["h"], 2)] for b in bb], np.float32).reshape(-1, 4)
        cxcywh = np.stack([xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]], 1) / 512
        out.append({"class_labels": torch.tensor([TYPES.index(b["finding"]) for b in bb], dtype=torch.long),
                    "boxes": torch.from_numpy(np.clip(cxcywh, 0, 1)).float()})
    return out


@torch.no_grad()
def predict(model, proc, keys, dev, bs=32):
    model.eval()
    out = {}
    for i in range(0, len(keys), bs):
        ks = keys[i:i + bs]
        enc = proc(images=images(ks), return_tensors="pt").to(dev)
        res = proc.post_process_object_detection(model(**enc), threshold=0.0, target_sizes=[(512, 512)] * len(ks))
        for k, r in zip(ks, res):
            out[k] = {}
            for j, t in enumerate(TYPES):
                m = r["labels"] == j
                if m.any():
                    s, q = r["scores"][m].max(0)
                    bx = r["boxes"][m][q]
                    out[k][t] = (float(s), float((bx[0] + bx[2]) / 2), float((bx[1] + bx[3]) / 2))
                else:
                    out[k][t] = (0.0, 256.0, 256.0)
    model.train()
    return out


def write_rows(pr, keys, path):
    with open(path, "w") as f:
        for k in keys:
            for j, t in enumerate(TYPES):
                s, x, y = pr[k][t]
                f.write(json.dumps({"key": k, "i": j, "r": y, "c": x, "wm": 1.0, "src": "wm", "rank": 0,
                                    **{f"p_{u}": (s if u == t else 0.0) for u in TYPES}}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-keys", default=None, help="only these training images (few-box settings)")
    ap.add_argument("--balance", action="store_true", help="class-balanced epochs (cxr_balance.py)")
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
    if a.train_keys:
        allowed = set(open(a.train_keys).read().split())
        tr_keys = [k for k in tr_keys if k in allowed]
    if a.balance:
        tr_keys = balanced(tr_keys, lambda k: k in by)
    test_keys = sorted(g[g.split == "test"].key)

    proc = AutoImageProcessor.from_pretrained(a.model, size={"height": 512, "width": 512})
    model = AutoModelForObjectDetection.from_pretrained(a.model, num_labels=len(TYPES), ignore_mismatched_sizes=True,
                                                        id2label=dict(enumerate(TYPES)),
                                                        label2id={t: j for j, t in enumerate(TYPES)}).to(dev)
    back = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    opt = torch.optim.AdamW([{"params": back, "lr": a.lr / 10}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    steps = a.epochs * (len(tr_keys) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    name = os.path.basename(a.out.rstrip("/"))
    best = -1.0
    for ep in range(a.epochs):
        random.shuffle(tr_keys)
        tot = []
        for i in range(0, len(tr_keys) - a.bs + 1, a.bs):
            ks = tr_keys[i:i + a.bs]
            enc = proc(images=images(ks), return_tensors="pt").to(dev)
            lab = [{k: v.to(dev) for k, v in t.items()} for t in targets(ks, by)]
            loss = model(**enc, labels=lab).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); sched.step()
            tot.append(loss.item())
        pr = predict(model, proc, val_keys, dev)
        v = np.nanmean([auroc([pr[k][t][0] for k in val_keys], [int(any(x["finding"] == t for x in by.get(k, []))) for k in val_keys])
                        for t in TYPES])
        print(f"epoch {ep} loss {np.mean(tot):.3f} validation mean AUROC {v:.3f}", flush=True)
        if v > best:
            best = v
            model.save_pretrained(f"{a.out}/best")
    model = AutoModelForObjectDetection.from_pretrained(f"{a.out}/best").to(dev)
    os.makedirs(f"results/cxr_focal/{name}", exist_ok=True)
    write_rows(predict(model, proc, test_keys, dev), test_keys, f"results/cxr_focal/{name}/cxdet_test.jsonl")
    print(f"best validation {best:.3f}; test rows written", flush=True)
    for st in ("mimic_validate", "mimic_test", "nih_test"):
        keys = sorted({json.loads(l)["key"] for l in open(f"data/cxr_focal/wm/{st}.jsonl")})
        write_rows(predict(model, proc, keys, dev), keys, f"results/cxr_focal/{name}/{st}.jsonl")
        print(f"scored {st}: {len(keys)} images", flush=True)


if __name__ == "__main__":
    main()
