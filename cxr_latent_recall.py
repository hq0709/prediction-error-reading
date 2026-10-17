"""Pointing at NIH ChestX-ray14 boxes (984 boxes, 880 images, external to MIMIC) with the latent radiograph world model
(train_cxr_latent_wm.py): dense reading of each image (every 6 x 6 block of the 37 x 37 patch grid hidden in turn),
per-patch error raw (squared error of the predicted features) or precision-weighted (divided by the predicted
variance), upsampled to the 512 grid; top-K addresses with square suppression, as cxr_recall.py does for the pixel
world model.

    CUDA_VISIBLE_DEVICES=2 python cxr_latent_recall.py raddino runs_wm/cxr_latent_raddino
    CUDA_VISIBLE_DEVICES=2 python cxr_latent_recall.py raddino runs_wm/cxr_latent_ms phase     # multi-scale model
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from cxr_recall import KS, boxes, first_hit
from perr.cxr import CACHE
from perr.cxr_latent_wm import GRID, block_masks, make_encoder, make_predictor
from perr.cxr_wm import peaks, valid_region
from train_cxr_latent_wm import blank, encode, load_batch


@torch.no_grad()
def dense(enc, pred, mean, std, key, masks):
    img = load_batch([key], "cuda")
    target = encode(enc, img, mean, std)
    context = encode(enc, blank(img.expand(len(masks), -1, -1, -1), masks, mean), mean, std)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mu, s = pred(context, masks)
    se = ((mu.float() - target) ** 2).mean(-1)
    s = s.float().clamp(-10, 6)
    m = masks.float()
    raw = (se * m).sum(0) / m.sum(0)
    pw = (se * torch.exp(-s) * m).sum(0) / m.sum(0)
    up = lambda v: F.interpolate(v.view(1, 1, GRID, GRID), size=(512, 512), mode="bilinear", align_corners=False)[0, 0]  # noqa: E731
    return up(raw).cpu().numpy(), up(pw).cpu().numpy()


def main_phase(name, run):
    """Multi-scale model: one map per scale (3, 6, 12 patches) and their sum after scaling each by its median."""
    from cxr_latent_read import SCALES, read_batch
    enc, mean, std = make_encoder(name)
    enc = enc.cuda()
    pred = make_predictor().cuda().eval()
    pred.load_state_dict(torch.load(f"{run}/predictor.pt"))
    b = boxes()
    keys = sorted(b.key.unique())
    up = lambda v: F.interpolate(torch.from_numpy(v).float()[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0].numpy()  # noqa: E731
    res = {}
    for i in range(0, len(keys), 8):
        ks = keys[i:i + 8]
        for k, mp in zip(ks, read_batch(enc, pred, mean, std, ks)):          # (S, 2, G, G)
            V = valid_region(np.load(f"{CACHE}/{k}.npy"))
            r = {"valid_area": int(V.sum())}
            for j, nm in enumerate(("raw", "pw")):
                for si, sc in enumerate(SCALES):
                    r[f"{nm}{sc}"] = peaks(up(mp[si, j]), V, k=20, nms=24)
                r[f"{nm}_sum"] = peaks(up(sum(mp[si, j] / np.median(mp[si, j]) for si in range(len(SCALES)))), V, k=20, nms=24)
            res[k] = r
    names = [k for k in next(iter(res.values())) if k != "valid_area"]
    print(f"{len(b)} boxes on {b.key.nunique()} images")
    for nm in names:
        f = [first_hit(res[r.key][nm], (r.x, r.y, r.w, r.h)) for r in b.itertuples()]
        print(f"{nm:10s}", " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in f]):.2f}" for K in KS))


def main():
    name, run = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "phase":
        return main_phase(name, run)
    enc, mean, std = make_encoder(name)
    enc = enc.cuda()
    pred = make_predictor().cuda().eval()
    pred.load_state_dict(torch.load(f"{run}/predictor.pt"))
    masks = torch.from_numpy(block_masks().reshape(len(block_masks()), -1)).cuda()
    b = boxes()
    res = {}
    for key in sorted(b.key.unique()):
        img = np.load(f"{CACHE}/{key}.npy")
        V = valid_region(img)
        raw, pw = dense(enc, pred, mean, std, key, masks)
        res[key] = {"latent_raw": peaks(raw, V, k=20, nms=24), "latent_pw": peaks(pw, V, k=20, nms=24),
                    "latent_raw_nms40": peaks(raw, V, k=20, nms=40), "valid_area": int(V.sum())}
    names = [k for k in next(iter(res.values())) if k != "valid_area"]
    print(f"{len(b)} boxes on {b.key.nunique()} images")
    for nm in names:
        f = [first_hit(res[r.key][nm], (r.x, r.y, r.w, r.h)) for r in b.itertuples()]
        print(f"{nm:18s}", " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in f]):.2f}" for K in KS))
        per = {}
        for r, x in zip(b.itertuples(), f):
            per.setdefault(r.finding, []).append(x is not None and x <= 10)
        print("   K=10 per finding:", " ".join(f"{k[:10]}={np.mean(v):.2f}" for k, v in sorted(per.items())))


if __name__ == "__main__":
    main()
