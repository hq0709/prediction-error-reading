"""Dense reading of MIMIC radiographs with the multi-scale latent world model (train_cxr_latent_wm.py --masking phase):
per image a (scales, 2, 37, 37) float16 map of raw and precision-weighted per-patch errors, written to
cache/cxr_latent/<tag>/<key>.npy. Shards the key list across processes (one per GPU).

    CUDA_VISIBLE_DEVICES=2 python cxr_latent_read.py runs_wm/cxr_latent_ms ms 0 2     # shard 0 of 2
"""
import os
import sys

import numpy as np
import torch

from perr.cxr import mimic_frontal
from perr.cxr_latent_wm import GRID, make_encoder, make_predictor, phase_masks
from train_cxr_latent_wm import blank, encode, load_batch

OUT = "/raid/home/CAMCA/hj880/perr/cache/cxr_latent"
SCALES = (3, 6, 12)


def keys():
    m = mimic_frontal()
    m["key"] = "mimic/" + m.dicom_id
    train = set(open("results/cxr_train60k_keys.txt").read().split())
    return sorted(m[m.key.isin(train) | m.split.isin(["validate", "test"])].key)


@torch.no_grad()
def read_batch(enc, pred, mean, std, ks):
    """train_cxr_latent_wm.read_phase for several images at once: (len(ks), scales, 2, GRID, GRID)."""
    b = len(ks)
    img = load_batch(ks, "cuda")
    target = encode(enc, img, mean, std)                                            # (b, N, D)
    out = []
    for sc in SCALES:
        masks = torch.from_numpy(phase_masks(sc).reshape(4, -1)).cuda()
        mrep = masks.repeat(b, 1)                                                   # (b*4, N), image-major
        context = encode(enc, blank(img.repeat_interleave(4, 0), mrep, mean), mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, s = pred(context, mrep)
        se = ((mu.float() - target.repeat_interleave(4, 0)) ** 2).mean(-1).view(b, 4, -1)
        s = s.float().clamp(-10, 6).view(b, 4, -1)
        m = masks.float()[None]
        out.append(torch.stack([(se * m).sum(1), (se * torch.exp(-s) * m).sum(1)], 1).view(b, 2, GRID, GRID))
    return torch.stack(out, 1).cpu().numpy()


def main():
    run, tag, shard, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    enc, mean, std = make_encoder("raddino")
    enc = enc.cuda()
    pred = make_predictor().cuda().eval()
    pred.load_state_dict(torch.load(f"{run}/predictor.pt"))
    todo = [k for k in keys()[shard::n] if not os.path.exists(f"{OUT}/{tag}/{k}.npy")]
    print(f"shard {shard}/{n}: {len(todo)} images to read", flush=True)
    for i in range(0, len(todo), 8):
        ks = todo[i:i + 8]
        for k, mp in zip(ks, read_batch(enc, pred, mean, std, ks)):
            out = f"{OUT}/{tag}/{k}.npy"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, mp.astype(np.float16))
        if (i // 8 + 1) % 250 == 0:
            print(f"{i + len(ks)} read", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
