"""Slice features for the SliceWorld comparison (Tian & Song, NeurIPS 2026, arXiv 2605.24371; no code released, so
its world-state model is re-implemented from the paper's appendix): every scan of the 1 mm lung grid (cache/iso) as an
ordered sequence of at most 196 axial slices (evenly sampled), each slice a 3-channel image of the paper's windows
(lung WL -600 / WW 1500, soft tissue 40 / 400, bone 400 / 1500) resized to 224 x 224 by area interpolation, encoded by
a frozen DINOv2 ViT-L/14 (CLS + 256 patch tokens, fp16).

    python sliceworld_feats.py runs_wm/ctx_nll/all_uids.txt [--shard i n]
Writes cache/sliceworld/<uid>.npy ([T, 257, 1024] fp16) and <uid>.z.npy (the slices' indices on the 1 mm grid).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from perr.context_wm import ORGANS

OUT = "/raid/home/CAMCA/hj880/perr/cache/sliceworld"
MAX_T = 196
WINDOWS = ((-600, 1500), (40, 400), (400, 1500))
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def slices(V):
    """[T, 3, 224, 224] DINOv2 input of an iso volume, and the sampled z indices."""
    z = np.unique(np.round(np.linspace(0, len(V) - 1, min(len(V), MAX_T))).astype(int))
    x = torch.from_numpy(np.asarray(V[z], np.float32))
    ch = [((x - (wl - ww / 2)) / ww).clamp(0, 1) for wl, ww in WINDOWS]
    x = torch.stack(ch, 1)
    x = (x * 255).round() / 255                                  # the paper's uint8 images
    x = F.interpolate(x, size=(224, 224), mode="area")
    m, s = torch.tensor(MEAN)[None, :, None, None], torch.tensor(STD)[None, :, None, None]
    return (x - m) / s, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uids")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    a = ap.parse_args()
    from transformers import AutoModel
    os.makedirs(OUT, exist_ok=True)
    enc = AutoModel.from_pretrained("facebook/dinov2-large", torch_dtype=torch.bfloat16).cuda().eval()
    uids = open(a.uids).read().split()[a.shard[0]::a.shard[1]]
    iso = ORGANS["lung"]["iso"]
    for i, u in enumerate(uids):
        if os.path.exists(f"{OUT}/{u}.z.npy"):
            continue
        x, z = slices(np.load(f"{iso}/{u}.vol.npy", mmap_mode="r"))
        feats = []
        with torch.no_grad():
            for j in range(0, len(x), 64):
                feats.append(enc(pixel_values=x[j:j + 64].cuda().bfloat16()).last_hidden_state.half().cpu())
        np.save(f"{OUT}/{u}.npy", torch.cat(feats).numpy())
        np.save(f"{OUT}/{u}.z.npy", z)
        print(i, u, len(z), flush=True)


if __name__ == "__main__":
    main()
