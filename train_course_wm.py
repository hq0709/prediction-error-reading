"""Train the learned disease-course world model (see perr/course_wm.py) on autoPET training patients.

World-model dev split: 30 training patients are held out of training; their recall decides how the model is used.
Validation and test patients (the evaluation set) are never read. Validation loss: hidden-part MSE on dev cubes.

    CUDA_VISIBLE_DEVICES=0 python train_course_wm.py --steps 30000 --out runs_wm/course_nll
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from perr.context_wm import hole_mask, make_unet
from perr.course_wm import CTX, ISO, norm

AUTOPET = "/raid/home/CAMCA/hj880/data_ext/autopet4_longitudinal_ct"


def dev_split():
    """Pair keys "<pid>_<NN>" of training patients, split by patient (no SimpleITK needed here)."""
    pts = sorted(json.load(open(f"{AUTOPET}/data_split.json"))["train"])
    dev = set(np.random.default_rng(0).choice(pts, size=30, replace=False).tolist())
    keys = sorted(f[:-len(".meta.npy")] for f in os.listdir(ISO) if f.endswith(".meta.npy"))
    keys = [k for k in keys if k.rsplit("_", 1)[0] in set(pts)
            and not np.load(f"{ISO}/{k}.meta.npy", allow_pickle=True).item().get("empty")]
    return [k for k in keys if k.rsplit("_", 1)[0] not in dev], [k for k in keys if k.rsplit("_", 1)[0] in dev]


class Cubes(IterableDataset):
    """(follow-up cube, registered-baseline cube) pairs centred on the cached training centres."""

    def __init__(self, keys, seed, flip=True):
        self.keys, self.seed, self.flip = keys, seed, flip

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        cache, h = {}, CTX // 2
        while True:
            k = self.keys[rng.integers(len(self.keys))]
            if k not in cache:
                if len(cache) >= 32:
                    cache.pop(next(iter(cache)))
                C = np.load(f"{ISO}/{k}.centres.npy")
                cache[k] = (np.load(f"{ISO}/{k}.fu.npy", mmap_mode="r"), np.load(f"{ISO}/{k}.bl.npy", mmap_mode="r"), C)
            FU, BL, C = cache[k]
            if len(C) == 0:
                continue
            z, y, x = (int(v) for v in C[rng.integers(len(C))])
            sl = (slice(z - h, z + h), slice(y - h, y + h), slice(x - h, x + h))
            cube = np.stack([norm(FU[sl]), norm(BL[sl])])
            if self.flip and rng.random() < 0.5:
                cube = cube[:, :, :, ::-1]
            yield torch.from_numpy(np.ascontiguousarray(cube))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs_wm/course_nll")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train, dev = dev_split()
    json.dump({"train": train, "dev": dev}, open(f"{a.out}/split.json", "w"))
    print(f"{len(train)} training pairs, {len(dev)} dev pairs", flush=True)
    dl = DataLoader(Cubes(train, 0), batch_size=a.bs, num_workers=12, pin_memory=True, persistent_workers=True)
    val = torch.stack([c for c, _ in zip(Cubes(dev, 12345, flip=False), range(512))])
    model = make_unet(out_ch=2, in_ch=3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    M = torch.from_numpy(hole_mask()).cuda()[None, None]

    def loss_of(cube, mse_only=False):
        cube = cube.cuda(non_blocking=True)
        fu, bl = cube[:, :1], cube[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(torch.cat([fu * (1 - M), M.expand_as(fu), bl], 1)).float()
        se = (pred[:, :1] - fu) ** 2
        if mse_only:
            return (se * M).sum() / (M.sum() * fu.shape[0])
        s = pred[:, 1:].clamp(-12, 3)
        nll = 0.5 * (se * torch.exp(-s) + s) * torch.exp(s).detach() ** 0.5      # beta-NLL, beta = 0.5
        return (nll * M).sum() / (M.sum() * fu.shape[0])

    @torch.no_grad()
    def validate():
        model.eval()
        v = np.mean([loss_of(val[i:i + 64], mse_only=True).item() for i in range(0, len(val), 64)])
        model.train()
        return v

    m0 = M.cpu()[0, 0]
    print(f"val MSE of 'hidden = baseline' {(((val[:, 0] - val[:, 1]) ** 2) * m0).sum().item() / (m0.sum().item() * len(val)):.5f}",
          flush=True)
    best, t0, run = 1e9, time.time(), []
    for step, cube in enumerate(dl):
        loss = loss_of(cube)
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
