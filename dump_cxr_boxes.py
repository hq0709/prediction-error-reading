"""Every box a trained radiograph detector proposes (not only its top box per trained finding type): the first N boxes
per ChestX-Det test image by score, any class, as centres. For the unseen-finding test (open_set_cxr.py), so detectors
are compared with the world model at the same number of addresses.

    python dump_cxr_boxes.py frcnn runs_reader/cxr_detector            # torchvision (train_cxr_detector.py)
    python dump_cxr_boxes.py detr runs_reader/cxr_rtdetrv2              # transformers (train_cxr_detr.py)
    python dump_cxr_boxes.py yolo runs_reader/cxr_yolo11m               # ultralytics (train_cxr_yolo.py)
Writes results/cxr_boxes/<run name>.json: {key: [[score, row, col], ...]}.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from perr.cxr import CACHE

N = 60


def nms_centres(rows, d=16):
    """Boxes whose centres repeat within d px are one address (the world model's addresses use the same suppression)."""
    out = []
    for s, r, c in sorted(rows, reverse=True):
        if all(max(abs(r - q[1]), abs(c - q[2])) > d for q in out):
            out.append((s, r, c))
        if len(out) == N:
            break
    return out


def main():
    kind, run = sys.argv[1], sys.argv[2].rstrip("/")
    g = pd.read_csv("results/cxr_cxdet.csv")
    keys = sorted(g[g.split == "test"].key)
    dev = torch.device("cuda")
    out = {}
    if kind in ("frcnn", "retinanet", "fcos"):
        from train_cxr_detector import build
        model = build(kind).to(dev)
        model.load_state_dict(torch.load(f"{run}/best.pt", map_location=dev))
        model.eval()
        with torch.no_grad():
            for i in range(0, len(keys), 16):
                ks = keys[i:i + 16]
                xs = [torch.from_numpy(np.load(f"{CACHE}/{k}.npy")).float().div(255)[None].expand(3, -1, -1).to(dev) for k in ks]
                for k, d in zip(ks, model(xs)):
                    b, s = d["boxes"].cpu().numpy(), d["scores"].cpu().numpy()
                    out[k] = nms_centres([(float(si), float((bi[1] + bi[3]) / 2), float((bi[0] + bi[2]) / 2)) for bi, si in zip(b, s)])
    elif kind == "detr":
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        model = AutoModelForObjectDetection.from_pretrained(f"{run}/best").to(dev).eval()
        name = json.load(open(f"{run}/best/config.json")).get("_name_or_path", "")
        proc = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd" if "rtdetr" in run else "SenseTime/deformable-detr",
                                                  size={"height": 512, "width": 512})
        with torch.no_grad():
            for i in range(0, len(keys), 32):
                ks = keys[i:i + 32]
                ims = [np.repeat(np.load(f"{CACHE}/{k}.npy")[..., None], 3, -1) for k in ks]
                enc = proc(images=ims, return_tensors="pt").to(dev)
                res = proc.post_process_object_detection(model(**enc), threshold=0.0, target_sizes=[(512, 512)] * len(ks))
                for k, r in zip(ks, res):
                    b, s = r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()
                    out[k] = nms_centres([(float(si), float((bi[1] + bi[3]) / 2), float((bi[0] + bi[2]) / 2)) for bi, si in zip(b, s)])
    else:
        os.environ.setdefault("YOLO_CONFIG_DIR", "runs_reader/ultralytics/config")
        from ultralytics import YOLO
        model = YOLO(f"{run}/train/weights/best.pt")
        for i in range(0, len(keys), 64):
            ks = keys[i:i + 64]
            ims = [np.repeat(np.load(f"{CACHE}/{k}.npy")[..., None], 3, -1) for k in ks]
            for k, r in zip(ks, model.predict(ims, imgsz=512, conf=1e-3, max_det=300, verbose=False, batch=32)):
                b, s = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
                out[k] = nms_centres([(float(si), float((bi[1] + bi[3]) / 2), float((bi[0] + bi[2]) / 2)) for bi, si in zip(b, s)])
    os.makedirs("results/cxr_boxes", exist_ok=True)
    json.dump(out, open(f"results/cxr_boxes/{os.path.basename(run)}.json", "w"))
    print(os.path.basename(run), len(out), "images; median boxes per image", int(np.median([len(v) for v in out.values()])))


if __name__ == "__main__":
    main()
