"""DenseNet-121 reader of a radiograph together with its world-model expectation (same recipe as the supervised
baseline in sota_cxr/densenet121, which is the "image" mode here; only the input channels change).

Modes (three input channels, the same random affine applied to all):
  image      [X, X, X]                 the radiograph alone (the baseline)
  prior      [X, P, X]                 with the patient's previous radiograph registered onto it (zeros if none):
                                       the history as the world model's expectation of this image
  wm         [X, Z, X]                 with the world model's precision-weighted error map Z (observed minus expected,
                                       divided by the model's own s.d.; signed, clipped to +-4)
  wm_prior   [X, Z, P]                 both

Trains on the image keys in --keys (the same 60k training images for every mode), picks the epoch on the official
validate split, writes test probabilities to results/sota_cxr/<name>.csv (scored by sota_cxr/score.py).

    CUDA_VISIBLE_DEVICES=3 python train_cxr_cls.py --mode prior --keys results/cxr_train60k_keys.txt --name dn60k_prior
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr/sota_cxr")
sys.path.insert(0, "/raid/home/CAMCA/hj880/perr/sota_cxr/densenet121")
from common import CACHE, FINDINGS, targets  # noqa: E402
from train import auroc, build_model, letterbox_jpg  # noqa: E402  (the baseline's own model and helpers)
from perr.cxr import mimic_frontal  # noqa: E402

REG = "/raid/home/CAMCA/hj880/perr/cache/cxr_reg"
ZMAP = "/raid/home/CAMCA/hj880/perr/cache/cxr_z"
IMAGE_CHANNELS = {"image": [1.0, 1.0, 1.0], "prior": [1.0, 1.0, 1.0], "wm": [1.0, 0.0, 1.0], "wm_prior": [1.0, 0.0, 1.0]}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def channel(path, fill=0):
    return np.load(path) if os.path.exists(path) else np.full((512, 512), fill, np.uint8)


class CXR(Dataset):
    def __init__(self, df, mode):
        self.keys, self.paths, self.mode = df.key.tolist(), df.path.tolist(), mode
        y, keep = targets(df)
        self.y, self.keep = torch.from_numpy(y), torch.from_numpy(keep.astype(np.float32))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        p = f"{CACHE}/{k}.npy"
        x = np.load(p) if os.path.exists(p) else letterbox_jpg(self.paths[i])
        pri = channel(f"{REG}/{k}.npy")
        z = channel(f"{ZMAP}/{k}.npy", 128)
        ch = {"image": (x, x, x), "prior": (x, pri, x), "wm": (x, z, x), "wm_prior": (x, z, pri)}[self.mode]
        return torch.from_numpy(np.stack(ch)), self.y[i], self.keep[i]


def prep(x, res, train, mode):
    """uint8 (B, 3, 512, 512) -> normalised (B, 3, res, res); train: intensity jitter on image channels, one random
    affine for all channels."""
    x = F.interpolate(x.float().div_(255), size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    if train:
        b, dev = x.shape[0], x.device
        u = lambda *s: torch.rand(*s, device=dev) * 2 - 1  # noqa: E731
        img = torch.tensor(IMAGE_CHANNELS[mode], device=dev).view(1, 3, 1, 1)       # error maps get no jitter
        jit = ((x - 0.5) * (1 + 0.1 * u(b, 1, 1, 1)) + 0.5 + 0.05 * u(b, 1, 1, 1)).clamp(0, 1)
        x = img * jit + (1 - img) * x
        ang, sc = math.radians(10) * u(b), 1 + 0.1 * u(b)
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, 0.1 * u(b)], 1), torch.stack([sin, cos, 0.1 * u(b)], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, loader, res, dev, mode):
    model.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(prep(x.to(dev, non_blocking=True), res, False, mode)).float()).cpu())
    model.train()
    return torch.cat(out).numpy()


def mean_auc(prob, ds):
    y, keep = ds.y.numpy(), ds.keep.numpy() > 0
    return float(np.nanmean([auroc(y[keep[:, j], j], prob[keep[:, j], j]) for j in range(len(FINDINGS))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["image", "prior", "wm", "wm_prior"], required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda", 0)
    out = f"/raid/home/CAMCA/hj880/perr/runs_cls/{a.name}"
    os.makedirs(out, exist_ok=True)
    m = mimic_frontal()
    m["key"] = "mimic/" + m.dicom_id
    tr = m[(m.split == "train") & m.key.isin(set(open(a.keys).read().split()))].reset_index(drop=True)
    va, te = m[m.split == "validate"].reset_index(drop=True), m[m.split == "test"].reset_index(drop=True)
    dtr, dva, dte = CXR(tr, a.mode), CXR(va, a.mode), CXR(te, a.mode)
    print(f"{a.mode}: train {len(dtr)}, validate {len(dva)}, test {len(dte)}", flush=True)
    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, num_workers=16, pin_memory=True, drop_last=True,
                     persistent_workers=True, prefetch_factor=4)
    lva = DataLoader(dva, batch_size=128, num_workers=8, pin_memory=True)
    lte = DataLoader(dte, batch_size=128, num_workers=8, pin_memory=True)
    model = build_model().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = a.epochs * len(ltr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    best, hist = -1.0, []
    for ep in range(a.epochs):
        t0 = time.time()
        for x, y, k in ltr:
            x, y, k = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), k.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(prep(x, a.res, True, a.mode))
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * k).sum() / k.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
        v = mean_auc(predict(model, lva, a.res, dev, a.mode), dva)
        hist.append({"epoch": ep, "val_mean_auroc": v})
        print(f"epoch {ep} validate mean AUROC {v:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if v > best:
            best = v
            torch.save(model.state_dict(), f"{out}/best.pt")
    json.dump(hist, open(f"{out}/history.json", "w"), indent=1)
    model.load_state_dict(torch.load(f"{out}/best.pt"))
    prob = predict(model, lte, a.res, dev, a.mode)
    pd.DataFrame(prob, columns=list(FINDINGS)).assign(key=dte.keys)[["key"] + list(FINDINGS)].to_csv(
        f"/raid/home/CAMCA/hj880/perr/results/sota_cxr/{a.name}.csv", index=False)
    print(f"best validate mean AUROC {best:.4f}; test probabilities written", flush=True)


if __name__ == "__main__":
    main()
