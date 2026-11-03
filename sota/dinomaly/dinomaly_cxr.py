"""Dinomaly (Guo et al., CVPR 2025; github.com/guojiajeremy/Dinomaly @ 1f252be, Apache-2.0) trained one-class on the
MIMIC-CXR "No Finding" frontal training images of the 512 world model (79,886 canvases, cache/cxr/<key>.npy), and its
anomaly maps over the whole 512 canvas.

Recipe = the repo's class-separated (one-class) scripts, dinomaly_{mvtec,visa,realiad,mpdd}_sep.py, which agree on every
setting: DINOv2-R ViT-B/14 encoder (official facebookresearch weights, frozen), layers 2-9 fused in 2 groups, noisy MLP
bottleneck (dropout 0.2), 8 linear-attention decoder blocks, StableAdamW (lr 2e-3, betas 0.9/0.999, wd 1e-4, AMSGrad,
eps 1e-8), warm-up 100 iterations then cosine to 2e-4, global hard-mining cosine loss with p rising to 0.9 over 1,000
iterations and factor 0.1, gradient clipping 0.1, batch 16, 5,000 iterations, seed 1, fp32. Map = mean over the two
groups of 1 - cosine(encoder, decoder) per token, bilinear to the input size, then (as in utils.evaluation_batch)
resized to 256 and smoothed with the repo's 5 x 5 Gaussian kernel (sigma 4); here finally bilinear to the 512 canvas.

Adaptation (NOTES.md): input = the whole 512 canvas resized to 448 x 448 without the 392 centre crop (the paper's
"R448" setting, Table A2), so the map covers the whole radiograph; grey canvas replicated to 3 channels, ImageNet
normalisation as in the repo.

    python -I -B sota/dinomaly/dinomaly_cxr.py train --out sota/dinomaly/runs/vitb_r448
    python -I -B sota/dinomaly/dinomaly_cxr.py maps --run sota/dinomaly/runs/vitb_r448 --keys <file> --name dinomaly
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

os.environ.setdefault("XFORMERS_DISABLED", "1")       # the vendored DINOv2 Block needs the plain-attention path
DINOMALY = "/raid/home/CAMCA/hj880/ext_src/dinomaly"
DINOV2_WEIGHTS = "/raid/home/CAMCA/hj880/ext_src/dinov2_weights/dinov2_vitb14_reg4_pretrain.pth"
ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.append(DINOMALY)

from PIL import Image                                                        # noqa: E402
from torch.utils.data import DataLoader, Dataset                            # noqa: E402
from torchvision import transforms                                          # noqa: E402

from dinov1.utils import trunc_normal_                                      # noqa: E402  (Dinomaly repo)
from dinov2.models import vision_transformer as vision_transformer_dinov2   # noqa: E402
from models.uad import ViTill                                               # noqa: E402
from models.vision_transformer import Block as VitBlock, LinearAttention2, bMlp   # noqa: E402

_spec = importlib.util.spec_from_file_location("stableadamw", f"{DINOMALY}/optimizers/StableAdamW.py")
_sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sa)
StableAdamW = _sa.StableAdamW

IMAGE_SIZE = 448
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---- verbatim from the repo's utils.py (importing utils.py itself pulls cv2 / matplotlib / skimage) -------------------
def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1), b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)
    loss = loss / len(a)
    return loss


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels, bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


class WarmCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]
# ---------------------------------------------------------------------------------------------------------------------


def setup_seed(seed):                                       # as in the repo
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model():
    """vit_encoder.load('dinov2reg_vit_base_14') + the bottleneck / decoder / ViTill of the *_sep.py scripts."""
    encoder = vision_transformer_dinov2.vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
                                                 num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1)
    sd = torch.load(DINOV2_WEIGHTS, map_location="cpu", weights_only=True)
    res = encoder.load_state_dict(sd, strict=False)                # strict=False as in vit_encoder.load
    print(f"DINOv2-R ViT-B/14 loaded: missing {res.missing_keys}, unexpected {res.unexpected_keys}", flush=True)
    embed_dim, num_heads = 768, 12
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList([VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0., attn=LinearAttention2)
                             for _ in range(8)])
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    trainable = nn.ModuleList([bottleneck, decoder])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    return model, trainable


TRANSFORM = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
                                transforms.Normalize(mean=MEAN, std=STD)])       # get_data_transforms without the crop


class Canvases(Dataset):
    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        img = Image.fromarray(np.load(f"{ROOT}/cache/cxr/{self.keys[i]}.npy")).convert("RGB")
        return TRANSFORM(img), i


def train_keys():
    import pandas as pd
    d = pd.read_csv(f"{ROOT}/results/cxr_mimic.csv")
    n = d[d["No Finding"] == 1.0]
    return n[n.split == "train"].key.tolist()


def train(a):
    setup_seed(1)
    dev = "cuda"
    keys = train_keys()
    if a.train_limit:
        keys = keys[:a.train_limit]
    print(f"{len(keys)} training images (MIMIC-CXR No Finding, frontal)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    total_iters, batch_size = a.iters, 16
    loader = DataLoader(Canvases(keys), batch_size=batch_size, shuffle=True, num_workers=a.workers, drop_last=True)
    model, trainable = build_model()
    model = model.to(dev)
    optimizer = StableAdamW([{'params': trainable.parameters()}], lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4,
                            amsgrad=True, eps=1e-8)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)
    print(f"trainable parameters {sum(p.numel() for p in trainable.parameters()) / 1e6:.1f} M", flush=True)
    it, t0, history = 0, time.time(), []
    for epoch in range(int(np.ceil(total_iters / len(loader)))):
        model.train()
        loss_list = []
        for img, _ in loader:
            img = img.to(dev)
            en, de = model(img)
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()
            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                history.append({"iter": it, "loss": float(np.mean(loss_list)), "lr": optimizer.param_groups[0]["lr"]})
                print(f"iter [{it}/{total_iters}], loss:{np.mean(loss_list):.4f} lr {optimizer.param_groups[0]['lr']:.2e} "
                      f"({time.time() - t0:.0f}s, {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB)", flush=True)
                loss_list = []
        if it == total_iters:
            break
    torch.save({"trainable": trainable.state_dict(), "iters": it, "args": vars(a)}, f"{a.out}/model.pt")
    json.dump(history, open(f"{a.out}/history.json", "w"), indent=1)
    print(f"done: {it} iterations in {time.time() - t0:.0f}s", flush=True)


def load_trained(run, dev="cuda"):
    model, trainable = build_model()
    ck = torch.load(f"{run}/model.pt", map_location="cpu", weights_only=True)
    trainable.load_state_dict(ck["trainable"], strict=True)
    return model.to(dev).eval()


@torch.no_grad()
def raw_maps(model, keys, bs=16, workers=8):
    """(key, 1 x 448 x 448 map) of the repo's cal_anomaly_maps at the input size, before resizing and smoothing."""
    dl = DataLoader(Canvases(keys), batch_size=bs, shuffle=False, num_workers=workers)
    for img, idx in dl:
        en, de = model(img.cuda())
        amap, _ = cal_anomaly_maps(en, de, img.shape[-1])
        for i, m in zip(idx.tolist(), amap):
            yield keys[i], m


def official_post(m, kernel):
    """utils.evaluation_batch post-processing (resize to 256, 5 x 5 Gaussian sigma 4), then bilinear to the 512 canvas."""
    m = F.interpolate(m[None], size=256, mode='bilinear', align_corners=False)
    m = kernel(m)
    return F.interpolate(m, size=512, mode='bilinear', align_corners=False)[0, 0]


def final_post(m, off, spec):
    """official: the repo's post-processing (official_post); id: no smoothing (bilinear from the input size to the
    canvas); sS: that map smoothed with a Gaussian of sigma S canvas px (scipy, as dev_recall.py)."""
    if spec == "official":
        return off
    up = F.interpolate(m[None], size=512, mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    if spec == "id":
        return up
    from scipy import ndimage as ndi
    return ndi.gaussian_filter(up.astype(np.float16).astype(np.float32), float(spec[1:]))


def save(p, arr):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, arr.astype(np.float16))


def maps(a):
    """--name: official-post maps to cache/cxr_maps/<name>/; --dev-variants: for choosing the smoothing on the held-out
    tenth, the official-post map (cache/uad_dev/dinomaly/official) and the unsmoothed map bilinearly resized from the
    input size to the canvas (cache/uad_dev/dinomaly/id; dev_recall.py smooths it further)."""
    model = load_trained(a.run)
    keys = open(a.keys).read().split()
    kernel = get_gaussian_kernel(kernel_size=5, sigma=4).cuda()
    n = 0
    for key, m in raw_maps(model, keys):
        if a.save_raw:
            save(f"{a.run}/raw448/{key}.npy", m[0].cpu().numpy())
        off = official_post(m, kernel).cpu().numpy()
        if a.name:
            save(f"{ROOT}/cache/cxr_maps/{a.name}/{key}.npy", final_post(m, off, a.post))
        if a.dev_variants:
            save(f"{ROOT}/{a.dev_dir}/official/{key}.npy", off)
            up = F.interpolate(m[None], size=512, mode='bilinear', align_corners=False)[0, 0]
            save(f"{ROOT}/{a.dev_dir}/id/{key}.npy", up.cpu().numpy())
        n += 1
    print(f"{n} maps written (name={a.name or '-'}, dev variants={a.dev_variants}, raw={a.save_raw})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--iters", type=int, default=5000)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--train-limit", type=int, default=0)
    m = sub.add_parser("maps")
    m.add_argument("--run", required=True)
    m.add_argument("--keys", required=True)
    m.add_argument("--name", default="", help="write official-post maps to cache/cxr_maps/<name>/<key>.npy")
    m.add_argument("--post", default="official", help="official | id | sS (smoothing chosen on the held-out tenth)")
    m.add_argument("--save-raw", action="store_true", help="also keep the 448 raw maps in <run>/raw448")
    m.add_argument("--dev-variants", action="store_true", help="write the dev smoothing variants to --dev-dir")
    m.add_argument("--dev-dir", default="cache/uad_dev/dinomaly")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else maps(a)


if __name__ == "__main__":
    main()
