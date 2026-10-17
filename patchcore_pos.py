"""Does knowing where a patch sits help a memory of normal patches? PatchCore on RAD-DINO tokens (patchcore_cxr.py) with
each token's position in the thorax (u, v in [0, 1] across the bounding box of the thorax hull, perr.cxr_anat) appended
to its feature with weight lam: distance^2 = |f - f'|^2 + (lam * D)^2 |p - p'|^2, D = the median nearest-neighbour
feature distance of the bank. lam = 0 is PatchCore with a thorax-only bank. Maps (37 x 37, float16) for MIMIC
validation/test, NIH box images and ChestX-Det go to cache/cxr_maps/patchcore_pos<lam>/<key>.npy.

    CUDA_VISIBLE_DEVICES=6 python patchcore_pos.py
"""
import os

import numpy as np
import pandas as pd
import torch

from cxr_maps import keys_for
from perr.cxr_anat import bits
from perr.cxr_latent_wm import GRID, make_encoder
from patchcore_cxr import feats

ROOT = "/raid/home/CAMCA/hj880/perr"
OUT = f"{ROOT}/cache/cxr_maps"
LAMS = (0, 1, 3, 10)


def positions(key):
    """(GRID*GRID, 2) token positions (u, v) in the thorax box, and (GRID*GRID,) inside-thorax flags."""
    hull = ((bits(key) >> 15) & 1).astype(bool)                                  # 128 grid
    c = (np.arange(GRID) + 0.5) * 128 / GRID
    if not hull.any():
        y0, y1, x0, x1 = 0, 128, 0, 128
    else:
        ys, xs = np.nonzero(hull)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    v, u = np.meshgrid((c - y0) / max(1, y1 - y0), (c - x0) / max(1, x1 - x0), indexing="ij")
    inside = hull[np.clip(c.astype(int), 0, 127)][:, np.clip(c.astype(int), 0, 127)]
    return np.stack([u.ravel(), v.ravel()], 1).astype(np.float32), inside.ravel()


@torch.no_grad()
def nn_dist(x, bank, chunk=4096):
    out = []
    for i in range(0, len(x), chunk):
        out.append(torch.cdist(x[i:i + chunk].float(), bank.float()).min(1).values)
    return torch.cat(out)


def main():
    enc, mean, std = make_encoder("raddino")
    enc = enc.cuda()
    d = pd.read_csv(f"{ROOT}/results/cxr_mimic.csv")
    train = set(open(f"{ROOT}/results/cxr_train60k_keys.txt").read().split())
    nf = d[(d["No Finding"] == 1) & d.key.isin(train)].key.tolist()
    rng = np.random.default_rng(0)
    nf = list(rng.choice(nf, 3000, replace=False))
    bf, bp = [], []
    for ks, f in feats(enc, mean, std, nf):
        for k, x in zip(ks, f):
            p, ins = positions(k)
            idx = np.nonzero(ins)[0]
            idx = rng.choice(idx, min(200, len(idx)), replace=False)
            bf.append(x[torch.from_numpy(idx).cuda()].half())
            bp.append(torch.from_numpy(p[idx]).cuda().half())
    bf, bp = torch.cat(bf), torch.cat(bp)
    sub = torch.randperm(len(bf), device="cuda")[:20000]
    D = float(nn_dist(bf[sub], bf[torch.randperm(len(bf), device="cuda")[:200000]]).median())
    print(f"bank {tuple(bf.shape)} thorax tokens from {len(nf)} images; median NN distance {D:.2f}", flush=True)
    keys = keys_for("mimic_eval") + keys_for("nih") + keys_for("cxdet")
    for lam in LAMS:
        name = f"patchcore_pos{lam}"
        bank = torch.cat([bf.float(), lam * D * bp.float()], 1).half()
        todo = [k for k in keys if not os.path.exists(f"{OUT}/{name}/{k}.npy")]
        for ks, f in feats(enc, mean, std, todo):
            P = torch.from_numpy(np.stack([positions(k)[0] for k in ks])).cuda()
            q = torch.cat([f.float(), lam * D * P], 2)
            dist = nn_dist(q.reshape(-1, q.shape[-1]), bank).view(len(ks), GRID, GRID).cpu().numpy()
            for k, m in zip(ks, dist):
                os.makedirs(os.path.dirname(f"{OUT}/{name}/{k}.npy"), exist_ok=True)
                np.save(f"{OUT}/{name}/{k}.npy", m.astype(np.float16))
        print(f"{name}: {len(keys)} images mapped", flush=True)


if __name__ == "__main__":
    main()
