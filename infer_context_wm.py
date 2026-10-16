"""Dense prediction of the learned anatomy world model: every 12 mm block of the grown lung domain is predicted
from a 64 mm cube centred on it with the central 24 mm hidden (so every voxel is predicted from tissue at least
6 mm away, and a nodule up to 12 mm across is wholly hidden from the cube of the block holding its centre);
writes the error R = observed - expected (normalised units, 1 mm grid) to cache/iso/<uid>.<tag>R.npy, and for a
model with a variance head its log-variance to <tag>S.npy.

    CUDA_VISIBLE_DEVICES=7 python infer_context_wm.py runs_wm/ctx/model.pt ctx uids.txt --shard 0 4
"""
import argparse
import os

import numpy as np
import torch

from perr.context_wm import CTX, ORGANS, hole_mask, load_iso, make_unet, norm

BLOCK = 12


@torch.no_grad()
def predict(model, uid, bs=128, organ="lung"):
    V, D, _ = load_iso(uid, organ)
    X = torch.from_numpy(norm(V, organ)).cuda()
    h = CTX // 2
    P = torch.nn.functional.pad(X[None, None], (h, h + BLOCK) * 3, mode="replicate")[0, 0]
    S = X.shape
    nb = [-(-n // BLOCK) for n in S]
    Dt = torch.from_numpy(np.asarray(D) > 0).cuda()
    Dp = torch.nn.functional.pad(Dt[None, None].float(), [0, nb[2] * BLOCK - S[2], 0, nb[1] * BLOCK - S[1], 0, nb[0] * BLOCK - S[0]])[0, 0]
    occ = Dp.reshape(nb[0], BLOCK, nb[1], BLOCK, nb[2], BLOCK).amax((1, 3, 5)) > 0
    blocks = torch.nonzero(occ).cpu().numpy() * BLOCK
    M = torch.from_numpy(hole_mask()).cuda()[None, None]
    E = X.clone()
    two = model.out.out_channels == 2
    L = torch.full_like(X, -12.0) if two else None
    a, b = (CTX - BLOCK) // 2, (CTX + BLOCK) // 2
    for i in range(0, len(blocks), bs):
        chunk = blocks[i:i + bs]
        # cube centred on the block centre c = b0 + B/2: original [c - 32, c + 32) = padded [b0 + B/2, b0 + B/2 + 64)
        x = torch.stack([P[z + BLOCK // 2:z + BLOCK // 2 + CTX, y + BLOCK // 2:y + BLOCK // 2 + CTX,
                           w + BLOCK // 2:w + BLOCK // 2 + CTX] for z, y, w in chunk])[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(torch.cat([x * (1 - M), M.expand_as(x)], 1)).float()[:, :, a:b, a:b, a:b]
        for (z, y, w), p in zip(chunk, out):
            ez, ey, ew = min(BLOCK, S[0] - z), min(BLOCK, S[1] - y), min(BLOCK, S[2] - w)
            E[z:z + ez, y:y + ey, w:w + ew] = p[0, :ez, :ey, :ew]
            if two:
                L[z:z + ez, y:y + ey, w:w + ew] = p[1, :ez, :ey, :ew].clamp(-12, 3)
    return (X - E).half().cpu().numpy(), (L.half().cpu().numpy() if two else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("tag")
    ap.add_argument("uids")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--organ", choices=sorted(ORGANS), default="lung")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    sd = torch.load(a.ckpt, map_location="cuda")
    model = make_unet(out_ch=sd["out.weight"].shape[0]).cuda().eval()
    model.load_state_dict(sd)
    uids = open(a.uids).read().split()[a.shard[0]::a.shard[1]]
    iso = ORGANS[a.organ]["iso"]
    for n, u in enumerate(uids):
        out = f"{iso}/{u}.{a.tag}R.npy"
        if os.path.exists(out):
            continue
        R, L = predict(model, u, organ=a.organ)
        if L is not None:
            np.save(f"{iso}/{u}.{a.tag}S.npy", L)
        np.save(out + ".tmp.npy", R)
        os.replace(out + ".tmp.npy", out)
        print(n, u, flush=True)
