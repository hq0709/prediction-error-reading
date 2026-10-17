"""Supervised DenseNet-121 on the official MIMIC-CXR-JPG train split: every frontal (PA / AP) image, 13 findings.

Labels (fixed convention): y = 1 if the CheXpert label is 1.0, y = 0 if 0.0 or missing; uncertain (-1.0) labels are
masked out of the loss for that finding (multi-label BCE over the remaining entries).  ImageNet-initialised
torchvision DenseNet-121, 320 x 320 input, Adam 1e-4, cosine schedule, bf16 autocast.  After every epoch the model is
scored on the official validate split (all frontal images, same label convention); the epoch with the best mean
AUROC over the 13 findings is the one used on the test split (predict.py).  The test split is never read here.

    CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node 2 train.py --out ckpt
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/raid/home/CAMCA/hj880/perr")
from common import CACHE, FINDINGS, targets  # noqa: E402
from perr.cxr import mimic_frontal  # noqa: E402  (read-only use of the project's table builder)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def letterbox_jpg(path: str, size: int = 512) -> np.ndarray:
    """perr.cxr.letterbox for images not in the cache: JPEG DCT-domain downscale (draft, still >= size on the longer
    side), then LANCZOS to `size` on the longer side, zero padding to size x size.  Mean |diff| vs the cache 0.5/255."""
    im = Image.open(path)
    im.draft("L", (size, size))
    im = im.convert("L")
    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = np.asarray(im.resize((nw, nh), Image.LANCZOS), np.uint8)
    out = np.zeros((size, size), np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = small
    return out


class CXR(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.keys, self.paths = df.key.tolist(), df.path.tolist()
        y, keep = targets(df)
        self.y, self.keep = torch.from_numpy(y), torch.from_numpy(keep.astype(np.float32))
        have = {f[:-4] for f in os.listdir(f"{CACHE}/mimic")}
        self.cached = np.array([k.split("/", 1)[1] in have for k in self.keys])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        try:
            a = np.load(f"{CACHE}/{self.keys[i]}.npy") if self.cached[i] else letterbox_jpg(self.paths[i])
            keep = self.keep[i]
        except Exception as e:  # unreadable file: zero image, masked out of the loss
            print(f"unreadable {self.paths[i]}: {e!r}", flush=True)
            a, keep = np.zeros((512, 512), np.uint8), torch.zeros(len(FINDINGS))
        return torch.from_numpy(a), self.y[i], keep


def prep(x: torch.Tensor, res: int, train: bool) -> torch.Tensor:
    """uint8 (B, 512, 512) on GPU -> normalised (B, 3, res, res).  Train: brightness / contrast jitter and a random
    affine (rotation +-10 deg, scale 0.9-1.1, translation +-5 %), zero padding as in the letterbox."""
    x = x.unsqueeze(1).float().div_(255)
    x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    if train:
        b, dev = x.shape[0], x.device
        u = lambda *s: torch.rand(*s, device=dev) * 2 - 1  # noqa: E731
        x = ((x - 0.5) * (1 + 0.1 * u(b, 1, 1, 1)) + 0.5 + 0.05 * u(b, 1, 1, 1)).clamp_(0, 1)
        ang, sc = math.radians(10) * u(b), 1 + 0.1 * u(b)
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, 0.1 * u(b)], 1), torch.stack([sin, cos, 0.1 * u(b)], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    x = (x.expand(-1, 3, -1, -1) - MEAN.to(x.device)) / STD.to(x.device)
    return x.contiguous(memory_format=torch.channels_last)


def build_model() -> nn.Module:
    m = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, len(FINDINGS))
    return m


def auroc(y, s):
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


@torch.no_grad()
def predict(model, loader, res, dev):
    model.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(prep(x.to(dev, non_blocking=True), res, False)).float()).cpu())
    return torch.cat(out).numpy()


def val_auc(prob, ds):
    y, keep = ds.y.numpy(), ds.keep.numpy() > 0
    per = {f: auroc(y[keep[:, j], j], prob[keep[:, j], j]) for j, f in enumerate(FINDINGS)}
    return float(np.nanmean(list(per.values()))), per


def worker_init(_):
    torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/ckpt")
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64, help="global batch size")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300, help="linear warm-up steps")
    ap.add_argument("--workers", type=int, default=16, help="DataLoader workers per process")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many training images")
    ap.add_argument("--keys", default="", help="train only on these image keys (same-data comparison with the reader)")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed + rank)
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)

    m = mimic_frontal()
    m["key"] = "mimic/" + m.dicom_id
    tr, va = m[m.split == "train"].reset_index(drop=True), m[m.split == "validate"].reset_index(drop=True)
    if a.limit:
        tr = tr.sample(a.limit, random_state=a.seed).reset_index(drop=True)
    if a.keys:
        tr = tr[tr.key.isin(set(open(a.keys).read().split()))].reset_index(drop=True)
    dtr, dva = CXR(tr), CXR(va)
    if rank == 0:
        y, keep = targets(tr)
        print(f"train {len(dtr)} frontal images ({int(dtr.cached.sum())} cached), validate {len(dva)}", flush=True)
        print("train positives:", dict(zip(FINDINGS, (y * keep).sum(0).astype(int).tolist())), flush=True)
    per_gpu = a.batch // world
    sampler = DistributedSampler(dtr, num_replicas=world, rank=rank, shuffle=True, seed=a.seed, drop_last=True)
    ltr = DataLoader(dtr, batch_size=per_gpu, sampler=sampler, num_workers=a.workers, pin_memory=True,
                     drop_last=True, persistent_workers=True, prefetch_factor=4, worker_init_fn=worker_init)
    lva = DataLoader(dva, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init)

    model = build_model().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = a.epochs * len(ltr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    start, hist = 0, []
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, hist = ck["epoch"] + 1, ck["hist"]
    net = nn.parallel.DistributedDataParallel(model, device_ids=[dev.index]) if ddp else model

    for ep in range(start, a.epochs):
        sampler.set_epoch(ep)
        net.train()
        t0, tl, nl = time.time(), 0.0, 0
        for it, (x, y, k) in enumerate(ltr):
            x, y, k = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), k.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = net(prep(x, a.res, True))
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * k).sum() / k.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl, nl = tl + loss.item(), nl + 1
            if rank == 0 and (it + 1) % 200 == 0:
                print(f"ep {ep} it {it + 1}/{len(ltr)} loss {tl / nl:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(it + 1) * a.batch / (time.time() - t0):.0f} img/s", flush=True)
                tl, nl = 0.0, 0
        if rank == 0:
            prob = predict(model, lva, a.res, dev)
            mean, per = val_auc(prob, dva)
            hist.append({"epoch": ep, "val_mean_auroc": mean, "val_auroc": per, "train_min": (time.time() - t0) / 60})
            print(f"== epoch {ep} val mean AUROC {mean:.4f} ({(time.time() - t0) / 60:.1f} min) "
                  + " ".join(f"{f[:6]}={v:.3f}" for f, v in per.items()), flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                        "hist": hist, "args": vars(a)}, f"{a.out}/epoch{ep}.pt")
            json.dump(hist, open(f"{a.out}/history.json", "w"), indent=1)
        if ddp:
            dist.barrier()
    if rank == 0:
        best = max(hist, key=lambda h: h["val_mean_auroc"])
        json.dump({"best_epoch": best["epoch"], "val_mean_auroc": best["val_mean_auroc"]}, open(f"{a.out}/best.json", "w"))
        print(f"best epoch {best['epoch']} val mean AUROC {best['val_mean_auroc']:.4f}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
