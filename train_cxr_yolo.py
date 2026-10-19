"""Ultralytics detector baselines for focal radiograph findings (YOLO11, RT-DETR), COCO weights, fine-tuned on the same
ChestX-Det training boxes, held-out tenth and 512 canvas as train_cxr_detector.py, with ultralytics' own recipe
(augmentation, schedule, checkpoint chosen by its validation fitness on the held-out tenth) except horizontal flips,
which are off as for the other detectors and the readers.

Test rows are written in the focal reader's format (per image and finding: the highest-scoring box centre and its
score), so score_cxr_focal.py scores it like the readers.

    CUDA_VISIBLE_DEVICES=6 venvs/sota_cxr_yolo/bin/python train_cxr_yolo.py --model yolo11m.pt --out runs_reader/cxr_yolo11m
    CUDA_VISIBLE_DEVICES=6 venvs/sota_cxr_yolo/bin/python train_cxr_yolo.py --model rtdetr-l.pt --out runs_reader/cxr_rtdetr \
        --optimizer AdamW --lr0 0.0001 --weight-decay 0.0001
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from PIL import Image

ROOT = "/raid/home/CAMCA/hj880/perr"
os.environ.setdefault("YOLO_CONFIG_DIR", f"{ROOT}/runs_reader/ultralytics/config")

from cxr_balance import balanced  # noqa: E402
from cxr_recall import boxes  # noqa: E402
from perr.cxr import CACHE  # noqa: E402
from train_cxr_detector import TYPES, auroc  # noqa: E402

PNG = f"{ROOT}/data/cxr_yolo/png"
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]   # ultralytics resets CUDA_VISIBLE_DEVICES to its device


def prepare(name, train_keys, balance=False):
    """YOLO-format copy of the split: PNGs shared across runs (symlinked), one label file per image (a training image
    repeated by class balancing gets one symlink and label file per repeat)."""
    g = pd.read_csv(f"{ROOT}/results/cxr_cxdet.csv")
    b = boxes("cxdet")
    b = b[b.finding.isin(TYPES)]
    by = {k: v[["finding", "x", "y", "w", "h"]].to_dict("records") for k, v in b.groupby("key")}
    fkeys = sorted({json.loads(l)["key"] for l in open(f"{ROOT}/data/cxr_focal/wm/cxdet_train.jsonl")})
    random.Random(0).shuffle(fkeys)
    val_keys = fkeys[:len(fkeys) // 10]
    tr_keys = [k for k in g[g.split == "train"].key if k not in set(val_keys)]
    if train_keys:
        allowed = set(open(train_keys).read().split())
        tr_keys = [k for k in tr_keys if k in allowed]
    if balance:
        tr_keys = balanced(tr_keys, lambda k: k in by)
    test_keys = sorted(g[g.split == "test"].key)
    root = f"{ROOT}/data/cxr_yolo/{name}"
    os.makedirs(PNG, exist_ok=True)
    for split, keys in (("train", tr_keys), ("val", val_keys), ("test", test_keys)):
        os.makedirs(f"{root}/images/{split}", exist_ok=True)
        os.makedirs(f"{root}/labels/{split}", exist_ok=True)
        seen = {}
        for k in keys:
            stem = k.replace("/", "_")
            src = f"{PNG}/{stem}.png"
            if not os.path.exists(src):
                Image.fromarray(np.load(f"{CACHE}/{k}.npy")).save(src)
            seen[k] = seen.get(k, -1) + 1
            if seen[k]:
                stem = f"{stem}_r{seen[k]}"
            dst = f"{root}/images/{split}/{stem}.png"
            if not os.path.lexists(dst):
                os.symlink(src, dst)
            with open(f"{root}/labels/{split}/{stem}.txt", "w") as f:
                for x in by.get(k, []):
                    w, h = max(x["w"], 2), max(x["h"], 2)
                    f.write(f"{TYPES.index(x['finding'])} {(x['x'] + w / 2) / 512:.6f} {(x['y'] + h / 2) / 512:.6f} "
                            f"{w / 512:.6f} {h / 512:.6f}\n")
    with open(f"{root}/data.yaml", "w") as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for j, t in enumerate(TYPES):
            f.write(f"  {j}: {t}\n")
    return root, val_keys, test_keys, by


def predict(model, keys, bs=64):
    out = {}
    for i in range(0, len(keys), bs):
        ks = keys[i:i + bs]
        ims = [np.repeat(np.load(f"{CACHE}/{k}.npy")[..., None], 3, -1) for k in ks]
        for k, r in zip(ks, model.predict(ims, imgsz=512, conf=1e-6, max_det=100, device=DEVICE, verbose=False, batch=32)):
            cls, conf, xyxy = r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()
            out[k] = {}
            for j, t in enumerate(TYPES):
                m = np.flatnonzero(cls == j)
                if len(m):
                    q = m[np.argmax(conf[m])]
                    out[k][t] = (float(conf[q]), float((xyxy[q, 0] + xyxy[q, 2]) / 2), float((xyxy[q, 1] + xyxy[q, 3]) / 2))
                else:
                    out[k][t] = (0.0, 256.0, 256.0)
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
    ap.add_argument("--model", required=True, help="yolo11m.pt, rtdetr-l.pt, ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--optimizer", default="auto", help="RT-DETR: AdamW (its paper's recipe; ultralytics' auto lr diverges)")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--train-keys", default=None, help="only these training images (few-box settings)")
    ap.add_argument("--balance", action="store_true", help="class-balanced epochs (cxr_balance.py)")
    ap.add_argument("--score-sets", action="store_true", help="only score ChestX-Det test, MIMIC validate/test and NIH with the trained model")
    ap.add_argument("--resume", action="store_true", help="finish an interrupted training from its last checkpoint")
    a = ap.parse_args()
    name = os.path.basename(a.out.rstrip("/"))
    out = os.path.abspath(a.out)
    root, val_keys, test_keys, by = prepare(name, a.train_keys, a.balance)
    work = f"{ROOT}/runs_reader/ultralytics"
    os.makedirs(work, exist_ok=True)
    os.chdir(work)   # ultralytics drops downloaded weights (and its AMP check model) into the working directory
    from ultralytics import RTDETR, YOLO
    M = RTDETR if a.model.startswith("rtdetr") else YOLO
    res = f"{ROOT}/results/cxr_focal/{name}"
    os.makedirs(res, exist_ok=True)
    if a.resume:
        M(f"{out}/train/weights/last.pt").train(resume=True, device=DEVICE)
    elif not a.score_sets:
        model = M(a.model)
        model.train(data=f"{root}/data.yaml", imgsz=512, epochs=a.epochs, batch=a.batch, device=DEVICE, seed=0, workers=8,
                    project=out, name="train", exist_ok=True, plots=False, fliplr=0.0, optimizer=a.optimizer, lr0=a.lr0,
                    weight_decay=a.weight_decay)
    best = M(f"{out}/train/weights/best.pt")
    if not a.score_sets:
        pv = predict(best, val_keys)
        v = np.nanmean([auroc([pv[k][t][0] for k in val_keys],
                              [int(any(x["finding"] == t for x in by.get(k, []))) for k in val_keys]) for t in TYPES])
        print(f"best validation {v:.3f}", flush=True)
    write_rows(predict(best, test_keys), test_keys, f"{res}/cxdet_test.jsonl")
    print("test rows written", flush=True)
    for st in ("mimic_validate", "mimic_test", "nih_test"):
        keys = sorted({json.loads(l)["key"] for l in open(f"{ROOT}/data/cxr_focal/wm/{st}.jsonl")})
        write_rows(predict(best, keys), keys, f"{res}/{st}.jsonl")
        print(f"scored {st}: {len(keys)} images", flush=True)


if __name__ == "__main__":
    main()
