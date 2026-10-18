"""Interval-change reader for MIMIC-CXR pairs (labels: label_cxr_change.py; worse / same / better for eight findings,
"none" carries no loss): DenseNet-121 on [current, registered prior, current - prior] (perr/cxr_course.py registration)
at 320 px, the comparison every design shares, plus with --views paired evidence views read by a second DenseNet with
attention per finding (as train_cxr_evidence.py):

  none     the comparison only
  wm       six views at the course world model's change addresses (its precision-weighted error between the current
           radiograph and what it expects from the registered prior, both signs, sigma 8, thorax domain, 40 px
           suppression); each view is the same 96 px square of current, prior and their difference, at 160 px
  random   control: six random thorax points, same views

Epoch chosen on the official validation split (mean over findings of the three-class macro F1); test predictions go to
results/cxr_course/<name>.csv.

    python train_cxr_course.py addresses            # precompute the view addresses
    CUDA_VISIBLE_DEVICES=5 python train_cxr_course.py --views wm --name course_wm
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_cxr_evidence import trunk

ROOT = "/raid/home/CAMCA/hj880/perr"
CACHE, REG, WM = f"{ROOT}/cache/cxr", f"{ROOT}/cache/cxr_reg", f"{ROOT}/cache/cxr_wm"
FINDINGS = ("consolidation", "pulmonary edema", "pleural effusion", "pneumonia", "pneumothorax", "atelectasis",
            "cardiomegaly", "support devices")
CLASSES = ("worse", "same", "better")
K, HALF, VIEW = 6, 48, 160
ADDR = f"{ROOT}/results/cxr_course_addresses.json"


def addresses_of(key):
    from perr.cxr_anat import search_domain
    from perr.cxr_wm import error_map, peaks, valid_region
    img = np.load(f"{CACHE}/{key}.npy")
    D = search_domain(key, valid_region(img))
    out = {}
    try:
        R = np.load(f"{WM}/{key}.cxrcR.npy").astype(np.float32)
        S = np.load(f"{WM}/{key}.cxrcS.npy").astype(np.float32)
        out["wm"] = [(r, c) for r, c, _ in peaks(error_map(R, S, 8.0, 0), D, k=K, nms=40)]
    except FileNotFoundError:
        out["wm"] = []
    rng = np.random.default_rng(int(key.split("-")[-1], 16) % 2 ** 32)
    out["random"] = [(r, c) for r, c, _ in peaks(rng.random((512, 512)).astype(np.float32), D, k=K, nms=40)]
    return key, out


class Pairs(Dataset):
    def __init__(self, df, views, addr):
        self.keys, self.views, self.addr = df.key.tolist(), views, addr
        y = np.full((len(df), len(FINDINGS)), -1, np.int64)
        for j, f in enumerate(FINDINGS):
            y[:, j] = df[f].map({c: i for i, c in enumerate(CLASSES)}).fillna(-1).astype(int)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        cur = np.load(f"{CACHE}/{k}.npy").astype(np.float32) / 255
        pri = np.load(f"{REG}/{k}.npy").astype(np.float32) / 255
        x = np.stack([cur, pri, np.clip(cur - pri + 0.5, 0, 1)])
        v = np.zeros((K, 3, 2 * HALF, 2 * HALF), np.float32)
        if self.views != "none":
            P = np.pad(x, ((0, 0), (HALF, HALF), (HALF, HALF)))
            for n, (r, c) in enumerate(self.addr.get(k, {}).get(self.views, [])[:K]):
                v[n] = P[:, r:r + 2 * HALF, c:c + 2 * HALF]
        return torch.from_numpy(x), torch.from_numpy(v), self.y[i]


class Reader(nn.Module):
    def __init__(self, views, n=len(FINDINGS) * len(CLASSES)):
        super().__init__()
        self.views = views
        self.g = trunk()
        self.head_g = nn.Linear(1024, n)
        if views != "none":
            self.l = trunk()
            self.att = nn.Linear(1024, len(FINDINGS))
            self.head_l = nn.Linear(1024, n)

    @staticmethod
    def pool(f):
        return F.adaptive_avg_pool2d(F.relu(f), 1).flatten(1)

    def forward(self, x, v):
        out = self.head_g(self.pool(self.g(x)))
        if self.views != "none":
            b = x.shape[0]
            fl = self.pool(self.l(v)).view(b, K, -1)                                     # (B, K, 1024)
            a = self.att(fl).softmax(1)                                                  # (B, K, findings)
            pooled = torch.einsum("bkf,bkd->bfd", a, fl)                                 # (B, findings, 1024)
            w = self.head_l.weight.view(len(FINDINGS), len(CLASSES), -1)
            local = torch.einsum("bfd,fcd->bfc", pooled, w) + self.head_l.bias.view(len(FINDINGS), len(CLASSES))
            out = out + local.reshape(b, -1)
        return out.view(x.shape[0], len(FINDINGS), len(CLASSES))


def affine(x):
    """The baseline's random affine (rotation 10 deg, scale 0.1, shift 0.1), one transform for all channels of a sample."""
    b, dev = x.shape[0], x.device
    u = lambda *s: torch.rand(*s, device=dev) * 2 - 1  # noqa: E731
    ang, sc = math.radians(10) * u(b), 1 + 0.1 * u(b)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, 0.1 * u(b)], 1), torch.stack([sin, cos, 0.1 * u(b)], 1)], 1)
    return F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def prep(x, v, res, train):
    x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    v = F.interpolate(v.flatten(0, 1), size=(VIEW, VIEW), mode="bilinear", align_corners=False)
    x = affine(x) if train else x
    m = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return ((x - m) / s).contiguous(memory_format=torch.channels_last), ((v - m) / s).contiguous(memory_format=torch.channels_last)


def macro_f1(pred, y):
    f1 = []
    for c in range(len(CLASSES)):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1))


def auroc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(prob, y):
    out = {}
    for j, f in enumerate(FINDINGS):
        m = y[:, j] >= 0
        if m.sum() < 10:
            continue
        out[f] = {"n": int(m.sum()), "macroF1": macro_f1(prob[m, j].argmax(1), y[m, j]),
                  "acc": float(np.mean(prob[m, j].argmax(1) == y[m, j])),
                  "AUROC worse": auroc(prob[m, j, 0], (y[m, j] == 0).astype(int)),
                  "AUROC better": auroc(prob[m, j, 2], (y[m, j] == 2).astype(int))}
    return out


@torch.no_grad()
def predict(model, loader, res, dev):
    model.eval()
    out = []
    for x, v, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(*prep(x.to(dev), v.to(dev), res, False)).float().softmax(-1).cpu())
    model.train()
    return torch.cat(out).numpy()


def main():
    if sys.argv[1:2] == ["addresses"]:
        keys = pd.read_csv(f"{ROOT}/results/cxr_change_labels.csv").key.tolist()
        with Pool(48) as p:
            json.dump(dict(p.map(addresses_of, keys, chunksize=8)), open(ADDR, "w"))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", choices=["none", "wm", "random"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda", 0)
    out = f"{ROOT}/runs_cls/{a.name}"
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(f"{ROOT}/results/cxr_change_labels.csv")
    d = d[d[list(FINDINGS)].ne("none").any(axis=1)]
    d = d[[os.path.exists(f"{REG}/{k}.npy") for k in d.key]].reset_index(drop=True)
    addr = json.load(open(ADDR)) if a.views != "none" else {}
    tr, va, te = (d[d.split == s].reset_index(drop=True) for s in ("train", "validate", "test"))
    dtr, dva, dte = Pairs(tr, a.views, addr), Pairs(va, a.views, addr), Pairs(te, a.views, addr)
    print(f"{a.views}: train {len(dtr)}, validate {len(dva)}, test {len(dte)}", flush=True)
    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, num_workers=16, pin_memory=True, drop_last=True, persistent_workers=True)
    lva = DataLoader(dva, batch_size=64, num_workers=8)
    lte = DataLoader(dte, batch_size=64, num_workers=8)
    model = Reader(a.views).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = a.epochs * len(ltr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    best, hist = -1.0, []
    for ep in range(a.epochs):
        t0 = time.time()
        for x, v, y in ltr:
            x, v, y = x.to(dev, non_blocking=True), v.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(*prep(x, v, a.res, True))
            loss = F.cross_entropy(logit.float().reshape(-1, len(CLASSES)), y.reshape(-1), ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
        mv = metrics(predict(model, lva, a.res, dev), dva.y.numpy())
        val = float(np.mean([m["macroF1"] for m in mv.values()]))
        hist.append({"epoch": ep, "val_macroF1": val})
        print(f"epoch {ep} validation mean macro-F1 {val:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if val > best:
            best = val
            torch.save(model.state_dict(), f"{out}/best.pt")
    json.dump(hist, open(f"{out}/history.json", "w"), indent=1)
    model.load_state_dict(torch.load(f"{out}/best.pt"))
    prob = predict(model, lte, a.res, dev)
    os.makedirs(f"{ROOT}/results/cxr_course", exist_ok=True)
    cols = {f"{f}|{c}": prob[:, j, i] for j, f in enumerate(FINDINGS) for i, c in enumerate(CLASSES)}
    pd.DataFrame(cols).assign(key=dte.keys).to_csv(f"{ROOT}/results/cxr_course/{a.name}.csv", index=False)
    mt = metrics(prob, dte.y.numpy())
    for f, m in mt.items():
        print(f"  {f:18s} n={m['n']:4d} macroF1 {m['macroF1']:.3f} acc {m['acc']:.3f} AUROC worse {m['AUROC worse']:.3f} better {m['AUROC better']:.3f}")
    print(f"TEST mean macro-F1 {np.mean([m['macroF1'] for m in mt.values()]):.4f}, mean AUROC worse "
          f"{np.nanmean([m['AUROC worse'] for m in mt.values()]):.4f}, better {np.nanmean([m['AUROC better'] for m in mt.values()]):.4f}", flush=True)


if __name__ == "__main__":
    main()
