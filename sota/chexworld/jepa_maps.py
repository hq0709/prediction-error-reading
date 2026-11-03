"""Shared core of the world-model anomaly maps (CheXWorld: sota/chexworld/make_maps.py, RadJEPA: sota/radjepa/make_maps.py).

The map is the JEPA analogue of our prediction error. The 512 canvas is resized to the model's input resolution R,
cut into its G x G patch tokens, and every target block of bh x bw tokens on a stride-1 lattice that tiles the grid is
predicted in turn: the block is removed from the context (the context is every other token), the model's context
encoder encodes the context, the model's predictor predicts the block's target latents, and the per-token squared
error against the target encoder's latents (layer-normalised as the model's own loss does) is recorded. A token's
anomaly is its mean error over the blocks that cover it, a G x G float16 map over the whole canvas (higher = more
anomalous), cache/cxr_maps/<name>/<key>.npy.

An adapter wraps one model: it exposes the context-encoder input tokens, the context encoder applied to a subset of
tokens, the target encoder and the predictor, all with the model's own modules; positional embeddings at the native
resolution are the model's own, at other resolutions they are bicubically interpolated as the model's own
interpolate_pos_encoding does. verify() checks the adapter against the model's own forward at the native resolution.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/raid/home/CAMCA/hj880/perr"
CACHE = f"{ROOT}/cache/cxr"
MAPS = f"{ROOT}/cache/cxr_maps"
NORMS = ("loss", "both")   # loss: target layer-normalised (the model's loss); both: prediction layer-normalised as well


def canvas(key):
    return np.load(f"{CACHE}/{key}.npy")


def to_input(img, R, interp, mean, std, contrast=1.0):
    """512 canvas (uint8) -> (3, R, R) float tensor: resize, grey to three channels, [0, 1], mean/std normalisation
    (contrast < 1 only for the low-contrast control of verify(): [0, 1] values pulled towards 0.5)."""
    im = Image.fromarray(img, "L").resize((R, R), interp)
    x = torch.from_numpy(np.asarray(im, np.float32) / 255.0)[None].repeat(3, 1, 1)
    if contrast != 1.0:
        x = 0.5 + contrast * (x - 0.5)
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


def interp_pos(pos, n_new):
    """(1, N, D) grid embedding -> (1, n_new, D), bicubic, scale factor sqrt(n_new / N) (as interpolate_pos_encoding)."""
    N, D = pos.shape[1], pos.shape[2]
    if n_new == N:
        return pos
    g = int(round(N ** 0.5))
    out = F.interpolate(pos.reshape(1, g, g, D).permute(0, 3, 1, 2).float(), scale_factor=(n_new / N) ** 0.5,
                        mode="bicubic")
    assert out.shape[-1] ** 2 == n_new, out.shape
    return out.permute(0, 2, 3, 1).reshape(1, n_new, D)


def blocks(G, bh, bw, stride=1):
    """Top-left corners of a bh x bw block lattice tiling a G x G grid (the last row/column of blocks always included)."""
    tops = sorted(set(list(range(0, G - bh + 1, stride)) + [G - bh]))
    lefts = sorted(set(list(range(0, G - bw + 1, stride)) + [G - bw]))
    return [(t, l) for t in tops for l in lefts]


def block_indices(G, bh, bw, stride=1, device="cpu"):
    """(P, bh * bw) target and (P, G * G - bh * bw) context token indices (ascending), one row per block."""
    tgt, ctx = [], []
    for t, l in blocks(G, bh, bw, stride):
        m = torch.zeros(G, G, dtype=torch.bool)
        m[t:t + bh, l:l + bw] = True
        m = m.flatten()
        tgt.append(torch.nonzero(m).flatten())
        ctx.append(torch.nonzero(~m).flatten())
    return torch.stack(tgt).to(device), torch.stack(ctx).to(device)


def gather(x, idx):
    """x (B, N, D), idx (B, K) -> (B, K, D)."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


@torch.no_grad()
def map_batch(ad, x, bh, bw, stride=1, budget=160_000, raw=False):
    """Anomaly maps of a batch of model inputs x (B, 3, R, R): {norm: (B, G, G) float32} (raw: the per-token error sums
    {norm: (B, N)} and the number of blocks covering each token (N,))."""
    B = x.shape[0]
    G = x.shape[-1] // ad.patch
    N = G * G
    tgt_idx, ctx_idx = block_indices(G, bh, bw, stride, x.device)
    P, Nt = tgt_idx.shape
    h = ad.targets(x)                                   # (B, N, D) target-encoder latents
    hn = F.layer_norm(h, (h.shape[-1],))
    t0 = ad.tokens_ctx(x)                               # (B, N, D) context-encoder input tokens
    s = {n: torch.zeros(B, N, device=x.device) for n in NORMS}
    cnt = torch.zeros(N, device=x.device)
    chunk = max(1, budget // (B * N))
    for i in range(0, P, chunk):
        ti, ci = tgt_idx[i:i + chunk], ctx_idx[i:i + chunk]
        p = len(ti)
        ti_r = ti.repeat(B, 1)                          # (B * p, Nt), image-major
        ci_r = ci.repeat(B, 1)
        img = torch.arange(B, device=x.device).repeat_interleave(p)
        z = ad.encode_ctx(gather(t0[img], ci_r))        # (B * p, Nc, D)
        pred = ad.predict(z, ci_r, ti_r).float()        # (B * p, Nt, D)
        tgt = gather(hn[img], ti_r)
        err = {"loss": ((pred - tgt) ** 2).mean(-1),
               "both": ((F.layer_norm(pred, (pred.shape[-1],)) - tgt) ** 2).mean(-1)}
        for n in NORMS:
            s[n].view(-1).scatter_add_(0, (img[:, None] * N + ti_r).flatten(), err[n].flatten())
        cnt.scatter_add_(0, ti.flatten(), torch.ones(ti.numel(), device=x.device))
    assert bool((cnt > 0).all())
    if raw:
        return s, cnt
    return {n: (s[n] / cnt).view(B, G, G) for n in NORMS}


@torch.no_grad()
def map_tiles(ad, x, bh, bw, crop, stride, budget=160_000):
    """Native-resolution crops of a larger input: x (B, 3, Rb, Rb) is cut into crop x crop windows (the model's native
    input) every `stride` pixels; each window is mapped as a whole image (its own context), and a token's anomaly is
    its mean error over every (window, block) pair that covers it. {norm: (B, Rb / patch, Rb / patch)}."""
    B, p, Rb = x.shape[0], ad.patch, x.shape[-1]
    assert crop % p == 0 and stride % p == 0 and Rb % p == 0
    Gb, Gc = Rb // p, crop // p
    offs = sorted(set(list(range(0, Rb - crop + 1, stride)) + [Rb - crop]))
    S = {n: torch.zeros(B, Gb, Gb, device=x.device) for n in NORMS}
    C = torch.zeros(Gb, Gb, device=x.device)
    for oy in offs:
        for ox in offs:
            s, cnt = map_batch(ad, x[..., oy:oy + crop, ox:ox + crop], bh, bw, budget=budget, raw=True)
            ty, tx = oy // p, ox // p
            for n in NORMS:
                S[n][:, ty:ty + Gc, tx:tx + Gc] += s[n].view(B, Gc, Gc)
            C[ty:ty + Gc, tx:tx + Gc] += cnt.view(Gc, Gc)
    assert bool((C > 0).all())
    return {n: S[n] / C for n in NORMS}


def save(name, key, m):
    p = f"{MAPS}/{name}/{key}.npy"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, m.astype(np.float16))


def run(ad, keys, configs, names, bs=4, log=print):
    """configs: list of (R, bh, bw, tile), tile None (the whole canvas resized to R) or (crop, stride) (native crops of
    the canvas resized to R, map_tiles); names(cfg) -> {norm: map name}: writes cache/cxr_maps/<name>/<key>.npy."""
    for cfg in configs:
        R, bh, bw, tile = cfg
        ad.set_resolution(tile[0] if tile else R)
        nm = names(cfg)
        todo = [k for k in keys if not all(os.path.exists(f"{MAPS}/{v}/{k}.npy") for v in nm.values())]
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        budget = getattr(ad, "budget", 160_000)
        for i in range(0, len(todo), bs):
            ks = todo[i:i + bs]
            x = torch.stack([ad.preprocess(canvas(k), R) for k in ks]).to(ad.device)
            out = map_tiles(ad, x, bh, bw, *tile, budget=budget) if tile else map_batch(ad, x, bh, bw, budget=budget)
            for n, v in nm.items():
                for k, m in zip(ks, out[n].cpu().numpy()):
                    save(v, k, m)
        log(f"R={R}{f' (native {tile[0]} crops every {tile[1]} px)' if tile else ''} block {bh}x{bw}: {len(todo)} images "
            f"in {time.time() - t:.0f} s ({(time.time() - t) / max(1, len(todo)):.2f} s/image), "
            f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
