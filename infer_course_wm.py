"""Dense prediction of the learned disease-course world model on the 1.5 mm grid of each autoPET pair: every
12-voxel (18 mm) block of the domain is predicted from a cube centred on it (follow-up with the central 24 voxels
hidden, plus the registered baseline). Writes the error R = follow-up - expected (normalised units) to
cache/iso_course/<key>.<tag>R.npy and the predicted log-variance to <tag>S.npy.

    CUDA_VISIBLE_DEVICES=0 python infer_course_wm.py runs_wm/course_nll/model.pt cnll keys.txt --shard 0 4
"""
import argparse
import os

import numpy as np
import torch

from perr.context_wm import hole_mask, make_unet
from perr.course_wm import CTX, ISO, load_iso, norm

BLOCK = 12


@torch.no_grad()
def predict(model, key, bs=96):
    FUa, BLa, D, _ = load_iso(key)
    X, B = torch.from_numpy(norm(FUa)).cuda(), torch.from_numpy(norm(BLa)).cuda()
    h = CTX // 2
    pad = lambda t: torch.nn.functional.pad(t[None, None], (h, h + BLOCK) * 3, mode="replicate")[0, 0]
    P, Q = pad(X), pad(B)
    S = X.shape
    nb = [-(-n // BLOCK) for n in S]
    Dt = torch.from_numpy(np.asarray(D) > 0).cuda().float()
    Dp = torch.nn.functional.pad(Dt[None, None], [0, nb[2] * BLOCK - S[2], 0, nb[1] * BLOCK - S[1], 0, nb[0] * BLOCK - S[0]])[0, 0]
    occ = Dp.reshape(nb[0], BLOCK, nb[1], BLOCK, nb[2], BLOCK).amax((1, 3, 5)) > 0
    blocks = torch.nonzero(occ).cpu().numpy() * BLOCK
    M = torch.from_numpy(hole_mask()).cuda()[None, None]
    E, L = X.clone(), torch.full_like(X, -12.0)
    a, b = (CTX - BLOCK) // 2, (CTX + BLOCK) // 2
    o = BLOCK // 2
    for i in range(0, len(blocks), bs):
        chunk = blocks[i:i + bs]
        cut = lambda T: torch.stack([T[z + o:z + o + CTX, y + o:y + o + CTX, w + o:w + o + CTX] for z, y, w in chunk])[:, None]
        x, q = cut(P), cut(Q)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(torch.cat([x * (1 - M), M.expand_as(x), q], 1)).float()[:, :, a:b, a:b, a:b]
        for (z, y, w), p in zip(chunk, out):
            ez, ey, ew = min(BLOCK, S[0] - z), min(BLOCK, S[1] - y), min(BLOCK, S[2] - w)
            E[z:z + ez, y:y + ey, w:w + ew] = p[0, :ez, :ey, :ew]
            L[z:z + ez, y:y + ey, w:w + ew] = p[1, :ez, :ey, :ew].clamp(-12, 3)
    return (X - E).half().cpu().numpy(), L.half().cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("tag")
    ap.add_argument("keys")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    model = make_unet(out_ch=2, in_ch=3).cuda().eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cuda"))
    keys = open(a.keys).read().split()[a.shard[0]::a.shard[1]]
    for n, k in enumerate(keys):
        out = f"{ISO}/{k}.{a.tag}R.npy"
        if os.path.exists(out) or np.load(f"{ISO}/{k}.meta.npy", allow_pickle=True).item().get("empty"):
            continue
        R, L = predict(model, k)
        np.save(f"{ISO}/{k}.{a.tag}S.npy", L)
        np.save(out + ".tmp.npy", R)
        os.replace(out + ".tmp.npy", out)
        print(n, k, flush=True)
