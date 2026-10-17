"""Train the latent radiograph world model (perr/cxr_latent_wm.py) on MIMIC-CXR "No Finding" training images, then
read MIMIC validation and test images densely and score unsupervised abnormality detection.

The hidden region never reaches the context: context features come from the encoder run on the image with the
hidden blocks blanked out in pixels (a ViT mixes information across patches, so masking only in feature space would
leak the hidden content); target features come from the encoder run on the full image.

    CUDA_VISIBLE_DEVICES=2 python train_cxr_latent_wm.py --encoder dinov2 --steps 8000 --out runs_wm/cxr_latent_dinov2

--masking phase --scales 3,6,12: multi-scale world model. Each training image hides one pass of a block lattice
(perr.cxr_latent_wm.phase_masks) at a random scale; dense reading hides the four passes at every scale (4 encoder
runs per scale instead of 64 for the sliding 6 x 6 block), giving one error map per scale.
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from perr.cxr import CACHE, FINDINGS
from perr.cxr_latent_wm import BLOCK, GRID, block_masks, make_encoder, make_predictor, phase_masks, random_phase

PATCH = 14
SIDE = GRID * PATCH


def load_batch(keys, device):
    x = torch.from_numpy(np.stack([np.load(f"{CACHE}/{k}.npy") for k in keys])).to(device).float()[:, None] / 255.0
    return F.interpolate(x, size=(SIDE, SIDE), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)


def blank(img, hidden, mean):
    """Set the pixels of hidden patches to the encoder's mean intensity."""
    m = hidden.view(-1, 1, GRID, GRID).float()
    m = F.interpolate(m, scale_factor=PATCH, mode="nearest")
    return img * (1 - m) + mean.to(img.device) * m


@torch.no_grad()
def encode(enc, img, mean, std):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(pixel_values=((img - mean.to(img.device)) / std.to(img.device)).to(torch.bfloat16)).last_hidden_state
    return h[:, 1:1 + GRID * GRID].float()


def random_hidden(b, rng, device):
    m = np.zeros((b, GRID, GRID), bool)
    for i in range(b):
        for _ in range(rng.integers(2, 7)):
            r, c = rng.integers(0, GRID - BLOCK + 1, 2)
            m[i, r:r + BLOCK, c:c + BLOCK] = True
    return torch.from_numpy(m.reshape(b, -1)).to(device)


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def read_phase(enc, pred, mean, std, key, scales, dev="cuda"):
    """Dense reading at every scale: (len(scales), 2, GRID, GRID) raw and precision-weighted per-patch errors."""
    img = load_batch([key], dev)
    target = encode(enc, img, mean, std)
    out = []
    for sc in scales:
        masks = torch.from_numpy(phase_masks(sc).reshape(4, -1)).to(dev)
        context = encode(enc, blank(img.expand(4, -1, -1, -1), masks, mean), mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, s = pred(context, masks)
        se = ((mu.float() - target) ** 2).mean(-1)
        s = s.float().clamp(-10, 6)
        m = masks.float()
        out.append(torch.stack([(se * m).sum(0), (se * torch.exp(-s) * m).sum(0)]).view(2, GRID, GRID))
    return torch.stack(out).cpu().numpy()


def evaluate_phase(enc, pred, mean, std, scales, out):
    """MIMIC validation / test normal vs abnormal from the multi-scale maps: per scale and error type the mean of the
    top-t patches, and the sum over scales of maps z-scored by "No Finding" validation statistics per patch."""
    pred.eval()
    d = pd.read_csv("results/cxr_mimic.csv")
    ev = d[d.split.isin(["validate", "test"])].copy()
    others = [f for f in FINDINGS if f != "No Finding"]
    ev["abnormal"] = (ev[others] == 1).any(axis=1).astype(int)
    ev = ev[(ev["No Finding"] == 1) | (ev.abnormal == 1)].reset_index(drop=True)
    maps = np.stack([read_phase(enc, pred, mean, std, k, scales) for k in ev.key])        # (n, S, 2, G, G)
    np.save(f"{out}/maps_eval.npy", maps.astype(np.float16))
    ev[["key", "split", "abnormal", "No Finding"] + others].to_csv(f"{out}/maps_eval_keys.csv", index=False)
    nf = ((ev.split == "validate") & (ev["No Finding"] == 1)).to_numpy()
    mu, sd = maps[nf].mean(0), maps[nf].std(0) + 1e-6
    z = (maps - mu) / sd
    top = lambda m, t: np.sort(m.reshape(len(m), -1), 1)[:, ::-1][:, :t].mean(1)  # noqa: E731
    cols = {}
    for i, sc in enumerate(scales):
        for j, nm in enumerate(("raw", "pw")):
            for t in (5, 14, 50):
                cols[f"{nm}{sc}_top{t}"] = top(maps[:, i, j], t)
                cols[f"z{nm}{sc}_top{t}"] = top(z[:, i, j], t)
    for j, nm in enumerate(("raw", "pw")):
        for t in (5, 14, 50):
            cols[f"zsum_{nm}_top{t}"] = top(z[:, :, j].sum(1), t)
    sc_df = pd.concat([ev, pd.DataFrame(cols)], axis=1)
    sc_df.to_csv(f"{out}/scores.csv", index=False)
    val, test = sc_df[sc_df.split == "validate"], sc_df[sc_df.split == "test"]
    best = max(cols, key=lambda c: auroc(val[c], val.abnormal))
    print("validation, best 8:", " | ".join(f"{c} {auroc(val[c], val.abnormal):.3f}" for c in sorted(cols, key=lambda c: -auroc(val[c], val.abnormal))[:8]))
    print(f"TEST normal vs abnormal, configuration chosen on validation ({best}): {auroc(test[best], test.abnormal):.3f}")
    nfr = test[test["No Finding"] == 1]
    for f in others:
        pos = test[test[f] == 1]
        if len(pos) >= 20:
            sub = pd.concat([pos.assign(y=1), nfr.assign(y=0)])
            print(f"  {f:28s} n={len(pos):4d} AUROC vs No Finding {auroc(sub[best], sub.y):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["dinov2", "raddino"], default="dinov2")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masking", choices=["blocks", "phase"], default="blocks")
    ap.add_argument("--scales", default="3,6,12")
    a = ap.parse_args()
    scales = [int(x) for x in a.scales.split(",")]
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    enc, mean, std = make_encoder(a.encoder)
    enc = enc.to(dev)
    pred = make_predictor().to(dev)
    d = pd.read_csv("results/cxr_mimic.csv")
    train = d[(d.split == "train") & (d["No Finding"] == 1)].key.tolist()
    opt = torch.optim.AdamW(pred.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    rng = np.random.default_rng(0)
    t0, run = time.time(), []
    for step in range(a.steps):
        img = load_batch([train[i] for i in rng.integers(0, len(train), a.bs)], dev)
        hidden = (random_hidden(a.bs, rng, dev) if a.masking == "blocks" else
                  torch.from_numpy(random_phase(a.bs, rng, scales).reshape(a.bs, -1)).to(dev))
        target = encode(enc, img, mean, std)
        context = encode(enc, blank(img, hidden, mean), mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, s = pred(context, hidden)
        mu, s = mu.float(), s.float().clamp(-10, 6)
        se = ((mu - target) ** 2).mean(-1)
        nll = 0.5 * (se * torch.exp(-s) + s) * torch.exp(s).detach() ** 0.5           # beta-NLL, beta = 0.5
        loss = (nll * hidden).sum() / hidden.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        opt.step(); sched.step()
        run.append(((se * hidden).sum() / hidden.sum()).item())
        if (step + 1) % 1000 == 0:
            print(f"step {step + 1} hidden-patch MSE {np.mean(run):.4f} ({time.time() - t0:.0f}s)", flush=True)
            run = []
    torch.save(pred.state_dict(), f"{a.out}/predictor.pt")

    if a.masking == "phase":
        return evaluate_phase(enc, pred, mean, std, scales, a.out)
    # dense reading of validation and test images
    pred.eval()
    masks = torch.from_numpy(block_masks().reshape(len(block_masks()), -1)).to(dev)
    ev = d[d.split.isin(["validate", "test"])].copy()
    others = [f for f in FINDINGS if f != "No Finding"]
    ev["abnormal"] = (ev[others] == 1).any(axis=1).astype(int)
    ev = ev[(ev["No Finding"] == 1) | (ev.abnormal == 1)]
    rows = []
    with torch.no_grad():
        for k in ev.key:
            img = load_batch([k], dev)
            target = encode(enc, img, mean, std)
            context = encode(enc, blank(img.expand(len(masks), -1, -1, -1), masks, mean), mean, std)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, s = pred(context, masks)
            se = ((mu.float() - target) ** 2).mean(-1)
            s = s.float().clamp(-10, 6)
            m = masks.float()
            raw = (se * m).sum(0) / m.sum(0)
            pw = (se * torch.exp(-s) * m).sum(0) / m.sum(0)
            np.save(f"{a.out}/map_{k.replace('/', '_')}.npy", torch.stack([raw, pw]).view(2, GRID, GRID).cpu().numpy().astype(np.float16))
            top = lambda v: v.sort(descending=True).values
            rows.append({"key": k, **{f"{n}_top{t}": top(v)[:t].mean().item() for n, v in (("raw", raw), ("pw", pw)) for t in (5, 14, 50)}})
    sc = ev.merge(pd.DataFrame(rows), on="key")
    sc.to_csv(f"{a.out}/scores.csv", index=False)
    cfg = [c for c in rows[0] if c != "key"]
    val, test = sc[sc.split == "validate"], sc[sc.split == "test"]
    best = max(cfg, key=lambda c: auroc(val[c], val.abnormal))
    print("validation:", " | ".join(f"{c} {auroc(val[c], val.abnormal):.3f}" for c in cfg))
    print(f"TEST normal vs abnormal, configuration chosen on validation ({best}): {auroc(test[best], test.abnormal):.3f}")
    nf = test[test["No Finding"] == 1]
    for f in others:
        pos = test[test[f] == 1]
        if len(pos) >= 20:
            sub = pd.concat([pos.assign(y=1), nf.assign(y=0)])
            print(f"  {f:28s} n={len(pos):4d} AUROC vs No Finding {auroc(sub[best], sub.y):.3f}")


if __name__ == "__main__":
    main()
