"""Unsupervised baseline for the radiograph world models: PatchCore (Roth et al., CVPR 2022) on frozen RAD-DINO patch
features. Memory bank: patch features of "No Finding" MIMIC training images (random subsample of patches); anomaly
of a patch = distance to its nearest bank patch; image score = mean of the top-t patch anomalies.

Same protocol as train_cxr_latent_wm.py: MIMIC validation and test images that are "No Finding" or have any positive
finding; configuration (t) chosen on validation; test AUROC normal vs abnormal and per finding vs "No Finding". Also
pointing at the NIH ChestX-ray14 boxes (top-K peaks of the upsampled anomaly map, as cxr_recall.py).

    CUDA_VISIBLE_DEVICES=2 python patchcore_cxr.py --bank-images 3000 --per-image 200
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cxr_recall import KS, boxes, first_hit
from perr.cxr import CACHE, FINDINGS
from perr.cxr_latent_wm import GRID, make_encoder
from perr.cxr_wm import peaks, valid_region
from train_cxr_latent_wm import auroc, encode, load_batch


@torch.no_grad()
def feats(enc, mean, std, keys, bs=16):
    for i in range(0, len(keys), bs):
        yield keys[i:i + bs], encode(enc, load_batch(keys[i:i + bs], "cuda"), mean, std)      # (b, N, D)


@torch.no_grad()
def nn_dist(x, bank, chunk=4096):
    """x (n, D), bank (M, D) -> distance of each row of x to its nearest bank row."""
    out = []
    for i in range(0, len(x), chunk):
        out.append(torch.cdist(x[i:i + chunk].float(), bank.float()).min(1).values)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-images", type=int, default=3000)
    ap.add_argument("--per-image", type=int, default=200)
    a = ap.parse_args()
    enc, mean, std = make_encoder("raddino")
    enc = enc.cuda()
    d = pd.read_csv("results/cxr_mimic.csv")
    train = set(open("results/cxr_train60k_keys.txt").read().split())
    nf = d[(d["No Finding"] == 1) & d.key.isin(train)].key.tolist()
    rng = np.random.default_rng(0)
    nf = list(rng.choice(nf, a.bank_images, replace=False))
    bank = []
    for _, f in feats(enc, mean, std, nf):
        for x in f:
            bank.append(x[torch.from_numpy(rng.choice(len(x), a.per_image, replace=False)).cuda()].half())
    bank = torch.cat(bank)
    print(f"memory bank {tuple(bank.shape)} from {len(nf)} No Finding training images", flush=True)

    others = [f for f in FINDINGS if f != "No Finding"]
    ev = d[d.split.isin(["validate", "test"])].copy()
    ev["abnormal"] = (ev[others] == 1).any(axis=1).astype(int)
    ev = ev[(ev["No Finding"] == 1) | (ev.abnormal == 1)].reset_index(drop=True)
    rows = []
    for ks, f in feats(enc, mean, std, ev.key.tolist()):
        dist = nn_dist(f.reshape(-1, f.shape[-1]), bank).view(len(ks), -1)
        srt = dist.sort(1, descending=True).values
        for k, s in zip(ks, srt):
            rows.append({"key": k, **{f"top{t}": s[:t].mean().item() for t in (1, 5, 14, 50, 200)}})
    sc = ev.merge(pd.DataFrame(rows), on="key")
    sc.to_csv("results/patchcore_cxr_scores.csv", index=False)
    cfg = [c for c in rows[0] if c != "key"]
    val, test = sc[sc.split == "validate"], sc[sc.split == "test"]
    best = max(cfg, key=lambda c: auroc(val[c], val.abnormal))
    print("validation:", " | ".join(f"{c} {auroc(val[c], val.abnormal):.3f}" for c in cfg))
    print(f"TEST normal vs abnormal, configuration chosen on validation ({best}): {auroc(test[best], test.abnormal):.3f}")
    nfr = test[test["No Finding"] == 1]
    for f in others:
        pos = test[test[f] == 1]
        if len(pos) >= 20:
            sub = pd.concat([pos.assign(y=1), nfr.assign(y=0)])
            print(f"  {f:28s} n={len(pos):4d} AUROC vs No Finding {auroc(sub[best], sub.y):.3f}")

    # pointing at NIH boxes
    b = boxes()
    res = {}
    for ks, f in feats(enc, mean, std, sorted(b.key.unique())):
        dist = nn_dist(f.reshape(-1, f.shape[-1]), bank).view(len(ks), 1, GRID, GRID)
        up = F.interpolate(dist.float(), size=(512, 512), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        for k, m in zip(ks, up):
            res[k] = peaks(m, valid_region(np.load(f"{CACHE}/{k}.npy")), k=20, nms=24)
    f = [first_hit(res[r.key], (r.x, r.y, r.w, r.h)) for r in b.itertuples()]
    print("NIH pointing, PatchCore map:", " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in f]):.2f}" for K in KS))


if __name__ == "__main__":
    main()
