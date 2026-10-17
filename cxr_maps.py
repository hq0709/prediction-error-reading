"""Address maps of external radiographs for pointing evaluations (NIH boxes, ChestX-Det): the multi-scale latent world
model's per-patch errors (cxr_latent_read.read_batch, (3, 2, 37, 37)) and the PatchCore anomaly map (37, 37; the bank of
patchcore_cxr.py, rebuilt with the same seed and saved), float16, cache/cxr_maps/<method>/<key>.npy.

    CUDA_VISIBLE_DEVICES=4 python cxr_maps.py nih
    CUDA_VISIBLE_DEVICES=4 python cxr_maps.py cxdet
    CUDA_VISIBLE_DEVICES=4 python cxr_maps.py mimic_eval
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from cxr_latent_read import read_batch
from perr.cxr_latent_wm import GRID, make_encoder, make_predictor
from patchcore_cxr import feats, nn_dist

ROOT = "/raid/home/CAMCA/hj880/perr"
OUT = f"{ROOT}/cache/cxr_maps"


def keys_for(which):
    if which == "nih":
        return pd.read_csv(f"{ROOT}/results/cxr_nih.csv").key.tolist()
    if which == "mimic_eval":                                               # MIMIC validation and test images
        d = pd.read_csv(f"{ROOT}/results/cxr_mimic.csv")
        return d[d.split.isin(["validate", "test"])].key.tolist()
    return sorted("cxdet/" + f[:-4] for f in os.listdir(f"{ROOT}/cache/cxr/cxdet") if f.endswith(".npy"))


def patchcore_bank(enc, mean, std):
    path = f"{OUT}/patchcore_bank.pt"
    if os.path.exists(path):
        return torch.load(path).cuda()
    d = pd.read_csv(f"{ROOT}/results/cxr_mimic.csv")
    train = set(open(f"{ROOT}/results/cxr_train60k_keys.txt").read().split())
    nf = d[(d["No Finding"] == 1) & d.key.isin(train)].key.tolist()
    rng = np.random.default_rng(0)
    nf = list(rng.choice(nf, 3000, replace=False))
    bank = []
    for _, f in feats(enc, mean, std, nf):
        for x in f:
            bank.append(x[torch.from_numpy(rng.choice(len(x), 200, replace=False)).cuda()].half())
    bank = torch.cat(bank)
    os.makedirs(OUT, exist_ok=True)
    torch.save(bank.cpu(), path)
    return bank


def save(method, key, a):
    p = f"{OUT}/{method}/{key}.npy"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, a.astype(np.float16))


def main():
    keys = keys_for(sys.argv[1])
    enc, mean, std = make_encoder("raddino")
    enc = enc.cuda()
    bank = patchcore_bank(enc, mean, std)
    todo = [k for k in keys if not os.path.exists(f"{OUT}/patchcore/{k}.npy")]
    for ks, f in feats(enc, mean, std, todo):
        dist = nn_dist(f.reshape(-1, f.shape[-1]), bank).view(len(ks), GRID, GRID).cpu().numpy()
        for k, m in zip(ks, dist):
            save("patchcore", k, m)
    pred = make_predictor().cuda().eval()
    pred.load_state_dict(torch.load(f"{ROOT}/runs_wm/cxr_latent_ms/predictor.pt"))
    todo = [k for k in keys if not os.path.exists(f"{OUT}/latent_ms/{k}.npy")]
    for i in range(0, len(todo), 8):
        ks = todo[i:i + 8]
        for k, mp in zip(ks, read_batch(enc, pred, mean, std, ks)):
            save("latent_ms", k, mp)
    print(f"{len(keys)} images mapped", flush=True)


if __name__ == "__main__":
    main()
