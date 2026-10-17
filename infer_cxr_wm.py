"""Dense prediction of the chest-radiograph world model: every 16 x 16 block of the canonical radiograph is predicted
from a 128 x 128 window centred on it (central 48 x 48 hidden). Writes the error R = observed - expected and the
predicted log-variance S (float16, 512 x 512) to cache/cxr_wm/<key>.<tag>{R,S}.npy.

    CUDA_VISIBLE_DEVICES=0 python infer_cxr_wm.py runs_wm/cxr_nll/model.pt cxr keys.txt --shard 0 2
    CUDA_VISIBLE_DEVICES=0 python infer_cxr_wm.py runs_wm/cxr_nll_s4/model.pt cxr4 keys.txt --scale 4

With --scale s the model reads the radiograph downscaled by s (train_cxr_wm.py --scale) and R, S are upsampled back to
the 512 canvas (R in the same intensity units; S the log-variance of a downscaled pixel). With --canvas 1024 the model
reads the 1024 grid (build_cxr1024.py) and R, S are pooled 2 x 2 onto the 512 canvas (S as the log of the mean variance).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from perr.cxr import CACHE, SIZE
from perr.cxr_course import REG
from perr.cxr_wm import BLOCK, CTX, downscale, hole_mask, make_unet2d, norm

OUT = "/raid/home/CAMCA/hj880/perr/cache/cxr_wm"


@torch.no_grad()
def predict(model, key, bs=512, course=False, scale=1, canvas=512):
    """All 32 x 32 windows at once: unfold extracts them, and the predicted central blocks tile the image exactly.
    With course=True the registered prior is a third input channel."""
    cache = CACHE if canvas == 512 else f"{CACHE}1024"
    X = torch.from_numpy(norm(downscale(np.load(f"{cache}/{key}.npy"), scale))).cuda()
    size = canvas // scale
    h, o, n = CTX // 2, BLOCK // 2, size // BLOCK
    windows = lambda img: torch.nn.functional.unfold(torch.nn.functional.pad(img[None, None], (h - o,) * 4), CTX,
                                                     stride=BLOCK)[0].T.reshape(-1, 1, CTX, CTX)
    win = windows(X)
    pw = windows(torch.from_numpy(norm(np.load(f"{REG}/{key}.npy"))).cuda()) if course else None
    M = torch.from_numpy(hole_mask()).cuda()[None, None]
    a, b = (CTX - BLOCK) // 2, (CTX + BLOCK) // 2
    outs = []
    for i in range(0, len(win), bs):
        x = win[i:i + bs]
        inp = [x * (1 - M), M.expand_as(x)] + ([pw[i:i + bs]] if course else [])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(model(torch.cat(inp, 1)).float()[:, :, a:b, a:b])
    out = torch.cat(outs)                                            # (n*n, 2, BLOCK, BLOCK), row-major blocks
    tile = lambda t: t.reshape(n, n, BLOCK, BLOCK).permute(0, 2, 1, 3).reshape(size, size)
    E, L = tile(out[:, 0]), tile(out[:, 1].clamp(-12, 3))
    R = X - E
    if canvas == 1024:
        R = F.avg_pool2d(R[None, None], 2)[0, 0]
        L = torch.log(F.avg_pool2d(torch.exp(L)[None, None], 2)[0, 0])
    if scale > 1:
        R, L = (F.interpolate(t[None, None], size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0] for t in (R, L))
    return R.half().cpu().numpy(), L.half().cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("tag")
    ap.add_argument("keys")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--canvas", type=int, default=512)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    sd = torch.load(a.ckpt, map_location="cuda")
    course = sd["enc.0.net.0.weight"].shape[1] == 3
    model = make_unet2d(in_ch=3 if course else 2).cuda().eval()
    model.load_state_dict(sd)
    keys = open(a.keys).read().split()[a.shard[0]::a.shard[1]]
    for n, k in enumerate(keys):
        out = f"{OUT}/{k}.{a.tag}R.npy"
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        R, L = predict(model, k, course=course, scale=a.scale, canvas=a.canvas)
        np.save(f"{OUT}/{k}.{a.tag}S.npy", L)
        np.save(out, R)
        if n % 200 == 0:
            print(n, k, flush=True)
