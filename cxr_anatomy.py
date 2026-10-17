"""Anatomy of every cached radiograph (512 letterbox, perr/cxr.py): the ChestX-Det PSPNet released with torchxrayvision
(14 structures: clavicles, scapulae, lungs, hila, heart, aorta, diaphragm, mediastinum, trachea, spine), stored as a
128 x 128 uint16 bit field, cache/cxr_anat/<key>.npy: bit i = structure i (perr.cxr_anat.STRUCTURES), bit 15 = the
thorax, the convex hull of lungs, hila, heart, aorta, mediastinum and diaphragm (the hull keeps lung regions that a
segmenter trained on clear lungs drops under dense opacity or effusion).

    CUDA_VISIBLE_DEVICES=4 /raid/home/CAMCA/hj880/venvs/sota_cxr_xrv/bin/python cxr_anatomy.py nih
    CUDA_VISIBLE_DEVICES=4 /raid/home/CAMCA/hj880/venvs/sota_cxr_xrv/bin/python cxr_anatomy.py mimic 0 2   # shard 0 of 2
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from skimage.morphology import convex_hull_image

ROOT = "/raid/home/CAMCA/hj880/perr"
CACHE, OUT = f"{ROOT}/cache/cxr", f"{ROOT}/cache/cxr_anat"
THORAX = (4, 5, 6, 7, 8, 9, 10, 11)          # lungs, hila, heart, aorta, diaphragm, mediastinum
GRID = 128


def keys_for(which):
    if which == "nih":
        return pd.read_csv(f"{ROOT}/results/cxr_nih.csv").key.tolist()
    if which == "cxdet":
        return sorted("cxdet/" + f[:-4] for f in os.listdir(f"{CACHE}/cxdet") if f.endswith(".npy"))
    d = pd.read_csv(f"{ROOT}/results/cxr_mimic.csv")             # validation / test images and "No Finding" training images
    train = open(f"{ROOT}/results/cxr_train60k_keys.txt").read().split()
    return d[d.split.isin(["validate", "test"])].key.tolist() + train + d[d.split == "train"].key.tolist()


def encode(prob):
    """(14, G, G) probabilities -> (G, G) uint16 bit field."""
    m = prob > 0.5
    out = np.zeros(m.shape[1:], np.uint16)
    for i in range(m.shape[0]):
        out |= m[i].astype(np.uint16) << i
    th = m[list(THORAX)].any(0)
    if th.any():
        out |= convex_hull_image(th).astype(np.uint16) << 15
    return out


def main():
    which = sys.argv[1]
    shard, nshard = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    keys = [k for k in keys_for(which)[shard::nshard] if not os.path.exists(f"{OUT}/{k}.npy")]
    print(f"{which}: {len(keys)} images to segment", flush=True)
    net = xrv.baseline_models.chestx_det.PSPNet().cuda().eval()
    pool = ThreadPoolExecutor(16)
    load = lambda k: np.load(f"{CACHE}/{k}.npy")  # noqa: E731

    def save(k, a):
        os.makedirs(os.path.dirname(f"{OUT}/{k}.npy"), exist_ok=True)
        np.save(f"{OUT}/{k}.npy", a)

    bs = 32
    for i in range(0, len(keys), bs):
        ks = keys[i:i + bs]
        x = torch.from_numpy(np.stack(list(pool.map(load, ks)))).cuda().float()[:, None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = net(x / 255 * 2048 - 1024)
        p = F.interpolate(torch.sigmoid(y.float()), size=(GRID, GRID), mode="area").cpu().numpy()
        list(pool.map(lambda a: save(a[0], encode(a[1])), zip(ks, p)))
        if i % (bs * 100) == 0:
            print(f"  {i + len(ks)}/{len(keys)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
