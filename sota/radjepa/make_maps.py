"""RadJEPA (Khan et al., 2026) anomaly maps for the ChestX-Det images: the I-JEPA training checkpoint released as
AIDElab-IITBombay/RadJEPA jepa_encoder.pth.tar (keys encoder, predictor, target_encoder; ViT-B/14 at 224, 12-block
384-wide predictor, epoch 300), run with the official I-JEPA model code (facebookresearch/ijepa
src/models/vision_transformer.py, which the checkpoint's key names and shapes match). Block prediction error as in
sota/chexworld/jepa_maps.py (shared core).

    python -I sota/radjepa/make_maps.py verify             # adapter == model forward; JEPA loss per preprocessing
    python -I sota/radjepa/make_maps.py search             # free settings on the held-out tenth (wmx_search/rj_*)
    python -I sota/radjepa/make_maps.py final R BH BW NORM  # chosen setting, held-out + test -> cache/cxr_maps/radjepa
Run on a GPU through queue/jobs3.tsv.
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr/sota/chexworld")
sys.path.insert(0, "/raid/home/CAMCA/hj880/ext_src/ijepa/repo")       # `src` (namespace package, no __init__ code)
import jepa_maps as jm  # noqa: E402
from src.models import vision_transformer as vit  # noqa: E402
from src.masks.multiblock import MaskCollator  # noqa: E402

CKPT = "/raid/home/CAMCA/hj880/ext_src/radjepa/hf/jepa_encoder.pth.tar"
HELD = "/raid/home/CAMCA/hj880/perr/sota/radjepa/heldout_keys.txt"
TEST = "/raid/home/CAMCA/hj880/perr/sota/radjepa/test_keys.txt"
TILE = (392, (224, 84))       # native 224 crops every 84 px of the canvas resized to 392: crop area 0.33 of the image,
                              # inside the training crop scale 0.3-1.0; positions stay the model's own
SEARCH = ([(224, 7, 6, None), (224, 4, 4, None), (224, 2, 2, None), (224, 1, 1, None), (448, 14, 12, None),
           (448, 8, 8, None), (448, 4, 4, None), (448, 2, 2, None)] + [(TILE[0], bh, bw, TILE[1]) for bh, bw in
                                                                        ((7, 6), (4, 4), (2, 2))])
DEV = "cuda"                                                    # overridable for a CPU dry run
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))      # I-JEPA make_transforms default normalisation


class RadJEPA:
    patch = 14
    native = 224
    budget = 48_000          # tokens per chunk (the I-JEPA attention materialises the attention matrix)

    def __init__(self):
        ck = torch.load(CKPT, map_location="cpu", weights_only=True)
        self.epoch, self.final_loss = ck["epoch"], ck["loss"]
        psd = ck["predictor"]
        depth = len({k.split(".")[1] for k in psd if k.startswith("predictor_blocks.")})
        pdim = psd["mask_token"].shape[-1]
        self.enc = vit.vit_base(img_size=[224], patch_size=14)
        self.tenc = vit.vit_base(img_size=[224], patch_size=14)
        self.pred = vit.vit_predictor(num_patches=self.enc.patch_embed.num_patches, embed_dim=768,
                                      predictor_embed_dim=pdim, depth=depth, num_heads=self.enc.num_heads)
        for m, part in ((self.enc, "encoder"), (self.tenc, "target_encoder"), (self.pred, "predictor")):
            m.load_state_dict({k.replace("module.", "", 1): v for k, v in ck[part].items()}, strict=True)
            m.to(DEV).eval()
        self.depth, self.pdim = depth, pdim
        self.mean, self.std = IMAGENET
        self.device = DEV
        self.set_resolution(224)

    def preprocess(self, img, R):
        return jm.to_input(img, R, Image.BILINEAR, self.mean, self.std)    # RandomResizedCrop default: bilinear

    def set_resolution(self, R):
        n = (R // self.patch) ** 2
        self.enc_pos = jm.interp_pos(self.enc.pos_embed.data, n).to(DEV)
        self.tenc_pos = jm.interp_pos(self.tenc.pos_embed.data, n).to(DEV)
        self.pred_pos = jm.interp_pos(self.pred.predictor_pos_embed.data, n).to(DEV)

    def tokens_ctx(self, x):
        return self.enc.patch_embed(x) + self.enc_pos

    def encode_ctx(self, t):
        for blk in self.enc.blocks:
            t = blk(t)
        return self.enc.norm(t)

    def targets(self, x):
        t = self.tenc.patch_embed(x) + self.tenc_pos
        for blk in self.tenc.blocks:
            t = blk(t)
        return self.tenc.norm(t)

    def predict(self, z, ctx_idx, tgt_idx):
        """VisionTransformerPredictor.forward with one context and one target mask."""
        p = self.pred
        B, Nc = ctx_idx.shape
        pos = self.pred_pos.expand(B, -1, -1)
        x = p.predictor_embed(z) + jm.gather(pos, ctx_idx)
        tok = p.mask_token.expand(B, tgt_idx.shape[1], -1) + jm.gather(pos, tgt_idx)
        x = torch.cat([x, tok], 1)
        for blk in p.predictor_blocks:
            x = blk(x)
        return p.predictor_proj(p.predictor_norm(x)[:, Nc:])


@torch.no_grad()
def verify(ad, keys):
    """1) the adapter reproduces the model's own forward at 224; 2) the model's own JEPA loss (smooth L1 on layer-normed
    targets, I-JEPA training masks) on held-out images under candidate preprocessings, next to the checkpoint's final
    training loss."""
    ad.set_resolution(224)
    x = torch.stack([ad.preprocess(jm.canvas(k), 224) for k in keys[:4]]).to(DEV)
    tgt_idx, ctx_idx = jm.block_indices(16, 7, 6, 1, DEV)
    sel = torch.tensor([0, 30, 60, 99], device=DEV)
    ti, ci = tgt_idx[sel], ctx_idx[sel]
    own = ad.pred(ad.enc(x, [ci]), [ci], [ti])
    mine = ad.predict(ad.encode_ctx(jm.gather(ad.tokens_ctx(x), ci)), ci, ti)
    ht = ad.tenc(x)
    print(f"verify: predictor max |diff| {(own - mine).abs().max().item():.2e} (scale {own.abs().mean().item():.2f}); "
          f"target encoder max |diff| {(ht - ad.targets(x)).abs().max().item():.2e}", flush=True)
    variants = {"imagenet+bilinear (I-JEPA)": (IMAGENET, Image.BILINEAR, 1.0),
                "imagenet+bicubic": (IMAGENET, Image.BICUBIC, 1.0),
                "MIMIC stats (RAD-DINO)": (((0.5307,) * 3, (0.2583,) * 3), Image.BILINEAR, 1.0),
                "xray mean/std (MedMAE)": (((0.5056,) * 3, (0.252,) * 3), Image.BILINEAR, 1.0),
                "no normalisation": (((0, 0, 0), (1, 1, 1)), Image.BILINEAR, 1.0),
                "imagenet, contrast x0.25": (IMAGENET, Image.BILINEAR, 0.25)}
    imgs = [jm.canvas(k) for k in keys[:64]]
    for name, ((mean, std), interp, contrast) in variants.items():
        torch.manual_seed(0)
        col = MaskCollator(input_size=(224, 224), patch_size=14, enc_mask_scale=(0.85, 1.0),
                           pred_mask_scale=(0.15, 0.2), aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
                           allow_overlap=False)
        losses, mses, base = [], [], []
        for rep in range(4):
            for i in range(0, len(imgs), 16):
                xb = torch.stack([jm.to_input(im, 224, interp, mean, std, contrast) for im in imgs[i:i + 16]]).to(DEV)
                _, me, mp = col([torch.zeros(1)] * len(xb))
                me, mp = [m.to(DEV) for m in me], [m.to(DEV) for m in mp]
                hf = F.layer_norm(ad.tenc(xb), (768,))
                hbar = hf.mean(1, keepdim=True)                      # the image's mean target token
                h = vit.repeat_interleave_batch(torch.cat([jm.gather(hf, m) for m in mp]), len(xb), repeat=len(me))
                hb = torch.cat([hbar.expand(-1, m.shape[1], -1) for m in mp])
                z = ad.pred(ad.enc(xb, me), me, mp)
                losses.append(F.smooth_l1_loss(z, h).item())
                mses.append(F.mse_loss(z, h).item())
                base.append(F.mse_loss(hb, h).item())
        print(f"JEPA smooth-L1 loss, {name:28s}: {np.mean(losses):.4f} (checkpoint's final training loss "
              f"{ad.final_loss:.4f}); MSE {np.mean(mses):.4f} / MSE of the image-mean token {np.mean(base):.4f} "
              f"= skill ratio {np.mean(mses) / np.mean(base):.3f}", flush=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "search", "final", "smoke"])
    ap.add_argument("setting", nargs="*", help="search: R BH BW (one configuration); final: R BH BW NORM")
    a = ap.parse_args()
    ad = RadJEPA()
    held = open(HELD).read().split()
    print(f"RadJEPA epoch {ad.epoch}, predictor depth {ad.depth} width {ad.pdim}, "
          f"{sum(p.numel() for p in ad.pred.parameters()) / 1e6:.1f}M predictor parameters", flush=True)
    if a.mode == "verify":
        verify(ad, held)
    elif a.mode == "smoke":                                  # a few images, timing of the heaviest configurations
        for cfg in SEARCH[:1] + [c for c in SEARCH if c[0] == 448][::3] + [c for c in SEARCH if c[3]][:1]:
            jm.run(ad, held[:4], [cfg], lambda c: {n: f"wmx_smoke/rj_{cfg_tag(c)}_{n}" for n in jm.NORMS})
    elif a.mode == "search":
        cfgs = [parse(a.setting)] if a.setting else SEARCH   # one configuration per queue job, or all
        jm.run(ad, held, cfgs, lambda c: {n: f"wmx_search/rj_{cfg_tag(c)}_{n}" for n in jm.NORMS})
    else:
        cfg, norm = parse(a.setting[:3]), a.setting[3]
        assert norm in jm.NORMS
        jm.run(ad, held + open(TEST).read().split(), [cfg], lambda c: {norm: "radjepa"})


def parse(setting):
    """R BH BW (the canvas resized to R) or T<R> BH BW (native crops of the canvas resized to R)."""
    r, bh, bw = setting[0], int(setting[1]), int(setting[2])
    return (int(r[1:]), bh, bw, TILE[1]) if r.startswith("T") else (int(r), bh, bw, None)


def cfg_tag(c):
    return f"{'T' if c[3] else 'R'}{c[0]}_b{c[1]}x{c[2]}"

if __name__ == "__main__":
    main()
