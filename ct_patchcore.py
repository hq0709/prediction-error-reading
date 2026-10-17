"""Non-predictive baseline for the lung CT world model: PatchCore (Roth et al., CVPR 2022) on axial slices of the 1 mm
lung volumes (cache/iso).

Features: ImageNet WideResNet-50-2 layer2 + layer3, each 3 x 3 average-pooled, layer3 upsampled to layer2's stride
(8 px), concatenated (1536-d), as in PatchCore; slices are upsampled 2x first, so a feature cell is 4 mm. Memory bank:
a random subsample of the cells that are scored (any overlap with the grown lung domain, so normal lung boundaries
are in the bank) outside the nodule exclusion masks (annotated nodules + 6 mm) of world-model training scans. Anomaly of a cell = distance to its nearest bank cell. The slice maps stack into a 3-D
map on the 1 mm grid, which is smoothed (sigma 2 mm) and read exactly like the world model's error (peaks inside the
grown lung domain, 10 mm suppression, ctx_recall.py's hit rule) on the 200 evaluation scans.

    CUDA_VISIBLE_DEVICES=5 python ct_patchcore.py --bank-scans 60 --bank-size 500000
    CUDA_VISIBLE_DEVICES=1 python ct_patchcore.py --backbone dinov2     # DINOv2-base patch tokens (14 px at 2x = 7 mm)
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy import ndimage as ndi

from ctx_recall import KS, first_hit
from perr.context_wm import load_iso, norm, to_original
from perr.study import load_luna

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
UP, STRIDE = 2, 8                     # input upsampling and feature stride: one cell = STRIDE / UP = 4 mm


class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3

    @torch.no_grad()
    def forward(self, x):
        f2 = self.l2(self.stem(x))
        f3 = self.l3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)                                    # (B, 1536, H/8, W/8)


class DinoExtractor(torch.nn.Module):
    """DINOv2-base patch tokens as a (B, 768, h, w) map (input sides rounded down to multiples of 14)."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.bfloat16)

    @torch.no_grad()
    def forward(self, x):
        h, w = x.shape[-2] // 14 * 14, x.shape[-1] // 14 * 14
        x = x[..., :h, :w]
        t = self.m(pixel_values=x.to(torch.bfloat16)).last_hidden_state[:, 1:]
        return t.transpose(1, 2).reshape(x.shape[0], -1, h // 14, w // 14)


@torch.no_grad()
def slice_features(net, V, zs, bs=16):
    """Features of axial slices zs of a 1 mm volume (HU): (len(zs), 1536, h, w)."""
    out = []
    for i in range(0, len(zs), bs):
        x = torch.from_numpy(norm(np.asarray(V[zs[i:i + bs]]))).cuda()[:, None]
        x = F.interpolate(x, scale_factor=UP, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net((x - MEAN.cuda()) / STD.cuda()).half())
    return torch.cat(out)


def cells(mask, shape):
    """Downsample a 2-D 1 mm mask stack (Z, Y, X) to the feature grid (Z, h, w) by area."""
    t = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[:, None]
    return F.adaptive_avg_pool2d(t, shape)[:, 0].numpy()


@torch.no_grad()
def nn_dist(x, bank, chunk=8192):
    out = []
    for i in range(0, len(x), chunk):
        out.append(torch.cdist(x[i:i + chunk].half(), bank.half()).float().min(1).values)
    return torch.cat(out) if out else torch.zeros(0, device=bank.device)


def peaks(W, dom, k=50, nms_mm=10.0):
    W = W.copy()
    W[np.asarray(dom) == 0] = 0
    r = int(np.ceil(nms_mm))
    out = []
    for _ in range(k):
        i = int(np.argmax(W))
        z, y, x = np.unravel_index(i, W.shape)
        v = float(W[z, y, x])
        if v <= 0:
            break
        out.append((z, y, x, v))
        W[max(0, z - r):z + r + 1, max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-scans", type=int, default=60)
    ap.add_argument("--bank-size", type=int, default=500000)
    ap.add_argument("--backbone", choices=["wrn50", "dinov2"], default="wrn50")
    a = ap.parse_args()
    net = (Extractor() if a.backbone == "wrn50" else DinoExtractor()).cuda().eval()
    rng = np.random.default_rng(0)
    train = json.load(open("runs_wm/ctx_nll/split.json"))["train"]
    per = a.bank_size // a.bank_scans
    bank = []
    for uid in rng.choice(train, a.bank_scans, replace=False):
        V, D, _ = load_iso(uid)
        ex = np.load(f"/raid/home/CAMCA/hj880/perr/cache/iso/{uid}.excl.npy", mmap_mode="r")
        zs = np.nonzero(np.asarray(D).any((1, 2)))[0]
        f = slice_features(net, V, zs)                                   # (n, C, h, w)
        # the same cells as are scored (any overlap with the grown lung domain, boundaries included), minus nodules
        ok = (cells(np.asarray(D)[zs] > 0, f.shape[-2:]) > 0) & (cells(np.asarray(ex)[zs] > 0, f.shape[-2:]) == 0)
        idx = np.argwhere(ok)
        idx = idx[rng.choice(len(idx), min(per, len(idx)), replace=False)]
        bank.append(f[idx[:, 0], :, idx[:, 1], idx[:, 2]])
    bank = torch.cat(bank)
    print(f"memory bank {tuple(bank.shape)} from {a.bank_scans} world-model training scans", flush=True)

    uids = [r["uid"] for r in json.load(open("results/ctx_recall_nll.json"))]
    rows = []
    for uid in uids:
        V, D, meta = load_iso(uid)
        zs = np.nonzero(np.asarray(D).any((1, 2)))[0]
        W = np.zeros(V.shape, np.float32)
        for i in range(0, len(zs), 64):                                 # bounded memory
            z = zs[i:i + 64]
            f = slice_features(net, V, z)
            n, c, h, w = f.shape
            inside = torch.from_numpy(cells(np.asarray(D)[z] > 0, (h, w)) > 0).cuda()
            dmap = torch.zeros(n, h, w, device="cuda")
            q = f.permute(0, 2, 3, 1)[inside]
            dmap[inside] = nn_dist(q, bank)
            up = F.interpolate(dmap[:, None], size=V.shape[1:], mode="bilinear", align_corners=False)[:, 0]
            W[z] = up.cpu().numpy()
        W = ndi.gaussian_filter(W, 2.0)
        s = load_luna(uid)
        pk = [(*to_original(p[:3], meta), p[3]) for p in peaks(W, D)]
        rows.append({"uid": uid, "patchcore": first_hit(s, pk), "peaks": [list(map(float, p)) for p in pk]})
    json.dump(rows, open(f"results/ct_patchcore_{a.backbone}_recall.json", "w"))
    f = [r["patchcore"] for r in rows]
    ref = {r["uid"]: r for r in json.load(open("results/ctx_recall_nll.json"))}
    print(f"{len(rows)} scans, recall of a nodule >= 6 mm among the top K:")
    print(f"PatchCore ({a.backbone})".ljust(24), " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in f]):.2f}" for K in KS))
    for key in ("shape", "ctx2spool"):
        g = [ref[r["uid"]].get(key) for r in rows]
        print(f"{key:24s}", " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in g]):.2f}" for K in KS))


if __name__ == "__main__":
    main()
