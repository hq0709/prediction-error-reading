"""Train the chest-radiograph world model (perr/cxr_wm.py) on MIMIC-CXR training images labelled "No Finding".
Validation loss: hidden-part MSE on windows of MIMIC validation "No Finding" images.
With --course: the disease-course variant, which also sees the patient's previous radiograph registered onto the
current one (a third input channel), trained on training pairs without any new finding (results/cxr_pairs.csv).

    CUDA_VISIBLE_DEVICES=0 python train_cxr_wm.py --steps 40000 --out runs_wm/cxr_nll
    CUDA_VISIBLE_DEVICES=0 python train_cxr_wm.py --course --steps 30000 --out runs_wm/cxr_course
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

from perr.cxr import CACHE
from perr.cxr_course import LABELS, REG
from perr.cxr_wm import CTX, hole_mask, make_unet2d, norm


class Windows(IterableDataset):
    """Windows of one image, or with course=True stacked (current, registered prior) windows of a pair."""

    def __init__(self, keys, seed, course=False):
        self.keys, self.seed, self.course = keys, seed, course

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        h = CTX // 2
        while True:
            key = self.keys[rng.integers(len(self.keys))]
            img = np.load(f"{CACHE}/{key}.npy")
            rows, cols = np.nonzero(img.any(1))[0], np.nonzero(img.any(0))[0]
            r = rng.integers(rows.min(), rows.max() + 1)
            c = rng.integers(cols.min(), cols.max() + 1)
            P = np.pad(img, h)
            w = norm(P[r:r + CTX, c:c + CTX])
            if self.course:
                Q = np.pad(np.load(f"{REG}/{key}.npy"), h)
                w = np.stack([w, norm(Q[r:r + CTX, c:c + CTX])])
            yield torch.from_numpy(np.ascontiguousarray(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs_wm/cxr_nll")
    ap.add_argument("--course", action="store_true", help="also condition on the registered prior radiograph")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.course:
        d = pd.read_csv("results/cxr_pairs.csv")
        stable = d[(d[[f"new_{f}" for f in LABELS]] != 1).all(axis=1) & (d.reg_ncc >= 0.5)]
        train, val = stable[stable.split == "train"].key.tolist(), stable[stable.split == "validate"].key.tolist()
    else:
        d = pd.read_csv("results/cxr_mimic.csv")
        normal = d[d["No Finding"] == 1.0]
        train, val = normal[normal.split == "train"].key.tolist(), normal[normal.split == "validate"].key.tolist()
    print(f"{len(train)} training images, {len(val)} validation images", flush=True)
    dl = DataLoader(Windows(train, 0, a.course), batch_size=a.bs, num_workers=12, pin_memory=True, persistent_workers=True)
    vx = torch.stack([w for w, _ in zip(Windows(val, 12345, a.course), range(1024))])
    model = make_unet2d(in_ch=3 if a.course else 2).cuda()
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    M = torch.from_numpy(hole_mask()).cuda()[None, None]

    def loss_of(x, mse_only=False):
        x = x.cuda(non_blocking=True)
        x, prior = (x[:, :1], x[:, 1:]) if a.course else (x[:, None], None)
        inp = [x * (1 - M), M.expand_as(x)] + ([prior] if a.course else [])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(torch.cat(inp, 1)).float()
        se = (pred[:, :1] - x) ** 2
        if mse_only:
            return (se * M).sum() / (M.sum() * x.shape[0])
        s = pred[:, 1:].clamp(-12, 3)
        nll = 0.5 * (se * torch.exp(-s) + s) * torch.exp(s).detach() ** 0.5          # beta-NLL, beta = 0.5
        return (nll * M).sum() / (M.sum() * x.shape[0])

    @torch.no_grad()
    def validate():
        model.eval()
        v = np.mean([loss_of(vx[i:i + 128], mse_only=True).item() for i in range(0, len(vx), 128)])
        model.train()
        return v

    m0 = M.cpu()[0, 0]
    cur = vx[:, 0] if a.course else vx
    vm = ((cur * (1 - m0)).sum((1, 2)) / (1 - m0).sum())[:, None, None]
    print(f"val MSE of the context-mean predictor {(((cur - vm) ** 2) * m0).sum().item() / (m0.sum().item() * len(vx)):.5f}", flush=True)
    if a.course:
        print(f"val MSE of 'hidden = registered prior' {(((cur - vx[:, 1]) ** 2) * m0).sum().item() / (m0.sum().item() * len(vx)):.5f}", flush=True)
    best, t0, run = 1e9, time.time(), []
    for step, x in enumerate(dl):
        loss = loss_of(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if (step + 1) % 2000 == 0 or step + 1 == a.steps:
            v = validate()
            print(f"step {step + 1} train {np.mean(run):.5f} val {v:.5f} ({time.time() - t0:.0f}s)", flush=True)
            run = []
            if v < best:
                best = v
                torch.save(model.state_dict(), f"{a.out}/model.pt")
            if (step + 1) % 10000 == 0:
                torch.save(model.state_dict(), f"{a.out}/model_{(step + 1) // 1000}k.pt")
        if step + 1 == a.steps:
            break
    print(f"best val {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
