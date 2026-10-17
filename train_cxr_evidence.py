"""Two-stream radiograph reader: the whole radiograph (the baseline DenseNet-121 recipe at 320 px) plus K full-resolution
evidence views (make_cxr_crops.py), read by a second DenseNet-121 shared across views; per finding, attention over the
views pools their features, and the logit is global + pooled-evidence.

  --crops wm     views at the world model's top-K addresses
  --crops grid   control: views at six fixed lung-field zones (enlargement without the world model)
  --crops prior  one view: the patient's previous frontal radiograph (the 512 px image of the previous study, read at
                 the global resolution); images without a previous study get a learned "no prior" term instead

Everything else (training images, labels, schedule, epoch chosen on the official validate split) is the image-only
baseline's (train_cxr_cls.py). Test probabilities go to results/sota_cxr/<name>.csv (scored by sota_cxr/score.py).

    CUDA_VISIBLE_DEVICES=5 python train_cxr_evidence.py --crops wm --keys results/cxr_train60k_keys.txt --name ev60k_wm
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
import torchvision
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr/sota_cxr")
sys.path.insert(0, "/raid/home/CAMCA/hj880/perr/sota_cxr/densenet121")
from common import CACHE, FINDINGS, targets  # noqa: E402
from train import auroc  # noqa: E402
from perr.cxr import mimic_frontal  # noqa: E402

CROPS = "/raid/home/CAMCA/hj880/perr/cache/cxr_crops_{}"
K, SIDE = 6, 224
PRIORS = "/raid/home/CAMCA/hj880/perr/results/cxr_prior_keys.csv"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class CXR(Dataset):
    def __init__(self, df, crops):
        self.keys, self.crops = df.key.tolist(), crops
        y, keep = targets(df)
        self.y, self.keep = torch.from_numpy(y), torch.from_numpy(keep.astype(np.float32))
        if crops == "prior":
            pk = pd.read_csv(PRIORS).set_index("key").prior_key
            self.prior = [pk.get(k) if isinstance(pk.get(k), str) else None for k in self.keys]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        x = np.load(f"{CACHE}/{k}.npy")
        if self.crops == "prior":
            q = self.prior[i]
            v = (np.load(f"{CACHE}/{q}.npy") if q else np.zeros((512, 512), np.uint8))[None]
            return torch.from_numpy(x), torch.from_numpy(v), self.y[i], self.keep[i], float(q is not None)
        p = f"{CROPS.format(self.crops)}/{k}.npy"
        v = np.load(p) if os.path.exists(p) else np.zeros((K, SIDE, SIDE), np.uint8)
        return torch.from_numpy(x), torch.from_numpy(v), self.y[i], self.keep[i], 1.0


def augment(x, train):
    """(B, 1, H, W) in [0, 1] -> same; train: brightness / contrast jitter and a random affine, as the baseline."""
    if not train:
        return x
    b, dev = x.shape[0], x.device
    u = lambda *s: torch.rand(*s, device=dev) * 2 - 1  # noqa: E731
    x = ((x - 0.5) * (1 + 0.1 * u(b, 1, 1, 1)) + 0.5 + 0.05 * u(b, 1, 1, 1)).clamp(0, 1)
    ang, sc = math.radians(10) * u(b), 1 + 0.1 * u(b)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, 0.1 * u(b)], 1), torch.stack([sin, cos, 0.1 * u(b)], 1)], 1)
    return F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def norm(x):
    x = (x.expand(-1, 3, -1, -1) - MEAN.to(x.device)) / STD.to(x.device)
    return x.contiguous(memory_format=torch.channels_last)


def prep(x, v, res, train):
    x = F.interpolate(x.unsqueeze(1).float().div_(255), size=(res, res), mode="bilinear", antialias=True,
                      align_corners=False)
    b, k, s = v.shape[:3]
    v = v.reshape(b * k, 1, s, s).float().div_(255)
    if s != SIDE:                                                   # the previous radiograph: read like the current one
        v = F.interpolate(v, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    return norm(augment(x, train)), norm(augment(v, train))


def trunk():
    m = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, memory_efficient=True)
    return m.features


class TwoStream(nn.Module):
    def __init__(self, n=len(FINDINGS)):
        super().__init__()
        self.g, self.l = trunk(), trunk()
        self.head_g = nn.Linear(1024, n)
        self.att = nn.Linear(1024, n)
        self.head_l = nn.Linear(1024, n)
        self.absent = nn.Parameter(torch.zeros(n))                                  # term used when there is no view

    @staticmethod
    def pool(f):
        return F.adaptive_avg_pool2d(F.relu(f), 1).flatten(1)

    def forward(self, x, v, have):
        b = x.shape[0]
        fg = self.pool(self.g(x))
        fl = self.pool(self.l(v)).view(b, v.shape[0] // b, -1)                     # (B, K, 1024)
        a = self.att(fl).softmax(1)                                                 # (B, K, n) attention per finding
        pooled = torch.einsum("bkn,bkd->bnd", a, fl)                               # (B, n, 1024)
        local = (pooled * self.head_l.weight).sum(-1) + self.head_l.bias
        h = have.view(b, 1).to(local.dtype)
        return self.head_g(fg) + h * local + (1 - h) * self.absent


@torch.no_grad()
def predict(model, loader, res, dev):
    model.eval()
    out = []
    for x, v, _, _, h in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xs, vs = prep(x.to(dev, non_blocking=True), v.to(dev, non_blocking=True), res, False)
            out.append(torch.sigmoid(model(xs, vs, h.to(dev)).float()).cpu())
    model.train()
    return torch.cat(out).numpy()


def mean_auc(prob, ds):
    y, keep = ds.y.numpy(), ds.keep.numpy() > 0
    return float(np.nanmean([auroc(y[keep[:, j], j], prob[keep[:, j], j]) for j in range(len(FINDINGS))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", choices=["wm", "grid", "prior"], required=True)
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
    dtr, dva, dte = CXR(tr, a.crops), CXR(va, a.crops), CXR(te, a.crops)
    print(f"{a.crops}: train {len(dtr)}, validate {len(dva)}, test {len(dte)}", flush=True)
    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, num_workers=16, pin_memory=True, drop_last=True,
                     persistent_workers=True, prefetch_factor=4)
    lva = DataLoader(dva, batch_size=64, num_workers=8, pin_memory=True)
    lte = DataLoader(dte, batch_size=64, num_workers=8, pin_memory=True)
    model = TwoStream().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = a.epochs * len(ltr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    best, hist = -1.0, []
    for ep in range(a.epochs):
        t0 = time.time()
        for x, v, y, k, h in ltr:
            x, v = x.to(dev, non_blocking=True), v.to(dev, non_blocking=True)
            y, k, h = y.to(dev, non_blocking=True), k.to(dev, non_blocking=True), h.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(*prep(x, v, a.res, True), h)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * k).sum() / k.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
        val = mean_auc(predict(model, lva, a.res, dev), dva)
        hist.append({"epoch": ep, "val_mean_auroc": val})
        print(f"epoch {ep} validate mean AUROC {val:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if val > best:
            best = val
            torch.save(model.state_dict(), f"{out}/best.pt")
    json.dump(hist, open(f"{out}/history.json", "w"), indent=1)
    model.load_state_dict(torch.load(f"{out}/best.pt"))
    prob = predict(model, lte, a.res, dev)
    pd.DataFrame(prob, columns=list(FINDINGS)).assign(key=dte.keys)[["key"] + list(FINDINGS)].to_csv(
        f"/raid/home/CAMCA/hj880/perr/results/sota_cxr/{a.name}.csv", index=False)
    print(f"best validate mean AUROC {best:.4f}; test probabilities written", flush=True)


if __name__ == "__main__":
    main()
