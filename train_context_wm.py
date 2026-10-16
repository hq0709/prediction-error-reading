"""Train the learned anatomy world model (see perr/context_wm.py) on normal anatomy of the training pool.

World-model dev split: 60 pool scans with a nodule >= 6 mm (liver: 15 pool cases with a tumour >= 10 mm) are held
out of training; their recall decides how the world model is used. No evaluation scan is read. Validation loss is the hidden-part MSE on dev-scan cubes (also for --nll).

    CUDA_VISIBLE_DEVICES=7 python train_context_wm.py --steps 40000 --out runs_wm/ctx
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from perr.context_wm import CTX, ORGANS, hole_mask, make_unet, norm


def dev_split(organ="lung"):
    if organ == "lung":
        big = json.load(open("results/pool_scans_with_big.json"))
        dev = sorted(np.random.default_rng(0).choice(sorted(big), size=60, replace=False).tolist())
        pool = open("results/train_uids.txt").read().split()
    else:              # liver: every MSD Task03 case outside the evaluation half; 15 cases with a target held out
        ev = set(open("sota/eval_liver.txt").read().split())
        src = ORGANS["liver"]["src"]
        pool = sorted({f.split(".")[0] for f in os.listdir(src) if f.endswith(".meta.npy")} - ev, key=lambda c: int(c.split("_")[1]))
        big = [c for c in pool if any(d >= 10 for _, d in np.load(f"{src}/{c}.meta.npy", allow_pickle=True).item()["nodules"])]
        dev = sorted(np.random.default_rng(0).choice(big, size=15, replace=False).tolist())
    return [u for u in pool if u not in set(dev)], dev


class Cubes(IterableDataset):
    """64 mm cubes centred on the cached training centres (half anywhere in the grown domain, half on tissue)."""

    def __init__(self, uids, seed, flip=True, organ="lung"):
        self.uids, self.seed, self.flip, self.organ = uids, seed, flip, organ

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        cache, h = {}, CTX // 2
        while True:
            u = self.uids[rng.integers(len(self.uids))]
            if u not in cache:
                if len(cache) >= 48:
                    cache.pop(next(iter(cache)))
                iso = ORGANS[self.organ]["iso"]
                cache[u] = (np.load(f"{iso}/{u}.vol.npy", mmap_mode="r"), np.load(f"{iso}/{u}.centres.npy"))
            V, C = cache[u]
            z, y, x = (int(v) for v in C[rng.integers(len(C))])
            cube = norm(V[z - h:z + h, y - h:y + h, x - h:x + h], self.organ)
            if self.flip and rng.random() < 0.5:
                cube = cube[:, :, ::-1]
            yield torch.from_numpy(np.ascontiguousarray(cube))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs_wm/ctx")
    ap.add_argument("--nll", action="store_true", help="also predict a log-variance; beta-NLL loss (beta 0.5)")
    ap.add_argument("--organ", choices=sorted(ORGANS), default="lung")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train, dev = dev_split(a.organ)
    json.dump({"train": train, "dev": dev}, open(f"{a.out}/split.json", "w"))
    dl = DataLoader(Cubes(train, 0, organ=a.organ), batch_size=a.bs, num_workers=12, pin_memory=True, persistent_workers=True)
    val = torch.stack([c for c, _ in zip(Cubes(dev, 12345, flip=False, organ=a.organ), range(512))])
    model = make_unet(out_ch=2 if a.nll else 1).cuda()
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    M = torch.from_numpy(hole_mask()).cuda()[None, None]

    def loss_of(x, mse_only=False):
        x = x.cuda(non_blocking=True)[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(torch.cat([x * (1 - M), M.expand_as(x)], 1)).float()
        se = (pred[:, :1] - x) ** 2
        if not a.nll or mse_only:
            return (se * M).sum() / (M.sum() * x.shape[0])
        s = pred[:, 1:].clamp(-12, 3)
        nll = 0.5 * (se * torch.exp(-s) + s) * torch.exp(s).detach() ** 0.5      # beta-NLL, beta = 0.5
        return (nll * M).sum() / (M.sum() * x.shape[0])

    @torch.no_grad()
    def validate():
        model.eval()
        v = np.mean([loss_of(val[i:i + 64], mse_only=True).item() for i in range(0, len(val), 64)])
        model.train()
        return v

    # reference: predict the hidden part as the mean of the visible part
    vm = ((val * (1 - M.cpu()[0, 0])).sum((1, 2, 3)) / (1 - M.cpu()[0, 0]).sum())[:, None, None, None]
    print(f"val MSE of the context-mean predictor {(((val - vm) ** 2) * M.cpu()[0, 0]).sum().item() / (M.sum().item() * len(val)):.5f}", flush=True)
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
        if step + 1 == a.steps:
            break
    print(f"best val {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
