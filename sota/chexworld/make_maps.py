"""CheXWorld (Yue et al., CVPR 2025) anomaly maps for the ChestX-Det images: the world model's own context encoder, target
encoder and conditioned predictor (iwm_dual_easy, ViT-B/16, 224, epoch-229 checkpoint of the public release), block
prediction error as in sota/chexworld/jepa_maps.py. The predictor is conditioned on the context view's augmentation
(brightness, contrast, gamma, blur radius, noise sigma); the context here is the unaugmented image, so the identity
parameters (1, 1, 1, 0, 0) are given, and the mask tokens carry the model's own same-view relative position
(rel_pos_21 = 0), exactly as in its intra-view (masked-block) training branch.

    python -I sota/chexworld/make_maps.py verify            # adapter == model forward; JEPA loss per preprocessing
    python -I sota/chexworld/make_maps.py search            # free settings on the held-out tenth (wmx_search/cxw_*)
    python -I sota/chexworld/make_maps.py final R BH BW NORM # chosen setting, held-out + test -> cache/cxr_maps/chexworld
Run on a GPU through queue/jobs3.tsv. Code: ext_src/chexworld/repo (models/ only, imported without models/__init__.py).
"""
import argparse
import importlib
import sys
import types
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr/sota/chexworld")
import jepa_maps as jm  # noqa: E402

REPO = "/raid/home/CAMCA/hj880/ext_src/chexworld/repo"
CKPT = "/raid/home/CAMCA/hj880/ext_src/chexworld/weights/chexworld_pretrained.tar"
HELD = "/raid/home/CAMCA/hj880/perr/sota/chexworld/heldout_keys.txt"
TEST = "/raid/home/CAMCA/hj880/perr/sota/chexworld/test_keys.txt"
TILE = (384, (224, 80))       # native 224 crops every 80 px of the canvas resized to 384: crop area 0.34 of the image,
                              # inside the training crop scale 0.3-1.0; positions stay the model's own
SEARCH = ([(224, b, b, None) for b in (6, 4, 2, 1)] + [(448, b, b, None) for b in (12, 8, 4, 2)]
          + [(TILE[0], b, b, TILE[1]) for b in (6, 4, 2)])
DEV = "cuda"                                                    # overridable for a CPU dry run
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))      # timm IMAGENET_DEFAULT_MEAN/STD (norm_type default)
IDENTITY_AUG = (1.0, 1.0, 1.0, 0.0, 0.0)                        # brightness, contrast, gamma factors; blur; noise
warnings.filterwarnings("ignore", category=FutureWarning)       # torch.backends.cuda.sdp_kernel() in the model code


def repo_module(name):
    """Import ext_src/chexworld/repo/<name>.py as part of a stand-in package (models/__init__.py is not executed)."""
    pkg_name, sub = name.split("/")
    alias = f"cxw_{pkg_name}"
    if alias not in sys.modules:
        pkg = types.ModuleType(alias)
        pkg.__path__ = [f"{REPO}/{pkg_name}"]
        sys.modules[alias] = pkg
    return importlib.import_module(f"{alias}.{sub}")


def load_checkpoint():
    """weights_only load; the checkpoint's 'args' entry is an argparse.Namespace (inspected: the pickle's only globals
    are OrderedDict, _rebuild_tensor_v2, FloatStorage and Namespace), so that one class is allow-listed."""
    with torch.serialization.safe_globals([argparse.Namespace]):
        ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    return ck["model"], vars(ck["args"]), ck["epoch"]


class CheXWorld:
    patch = 16
    native = 224

    def __init__(self):
        jv, jva = repo_module("models/jepa_vit"), repo_module("models/jepa_vit_add")
        self.pe = repo_module("models/pos_embed")
        sd, a, self.epoch = load_checkpoint()
        assert a["ssl_type"] == "iwm_dual_easy" and a["model"] == "vit_base" and a["input_size"] == 224
        assert a["target_norm_type"] == "avg_ln" and a["target_last_k"] == 1 and a["loss_type"] == "l2"
        self.args = a
        self.enc = jv.vit_base(img_size=224, patch_size=16, drop_path_rate=0.0)
        self.tenc = jv.vit_base(img_size=224, patch_size=16, drop_path_rate=0.0)
        self.pred = jva.vit_predictor_conditioned(
            policy_dim=a["policy_dim"], policy_net_layers=3, num_patches=self.enc.patch_embed.num_patches,
            embed_dim=768, predictor_embed_dim=a["pred_emb_dim"], depth=a["pred_depth"], num_heads=self.enc.num_heads,
            extra=True, ssl_type=a["ssl_type"], unify_embed=False, cond_type="feat")
        for m, pre in ((self.enc, "encoder."), (self.tenc, "target_encoder."), (self.pred, "predictor.")):
            m.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True)
            m.to(DEV).eval()
        self.aug = torch.tensor(IDENTITY_AUG, device=DEV)[None]
        self.mean, self.std = IMAGENET
        self.device = DEV
        self.set_resolution(224)

    def preprocess(self, img, R):
        return jm.to_input(img, R, Image.BICUBIC, self.mean, self.std)     # training crops: bicubic (interpolation=3)

    def set_resolution(self, R):
        n = (R // self.patch) ** 2
        g0 = int(self.pred.num_patches ** 0.5)
        rel0 = self.pe.get_2d_sincos_pos_embed_relative_easy(torch.zeros(1, 4), self.pred.predictor_pos_embed.shape[-1], g0)
        self.enc_pos = jm.interp_pos(self.enc.pos_embed.data, n).to(DEV)
        self.tenc_pos = jm.interp_pos(self.tenc.pos_embed.data, n).to(DEV)
        self.ctx_pos = jm.interp_pos(self.pred.predictor_pos_embed.data, n).to(DEV)   # context tokens (predictor)
        self.tgt_pos = jm.interp_pos(rel0.float(), n).to(DEV)                         # mask tokens, rel_pos 0

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
        """VisionTransformerPredictorConditioned.forward (cond_type feat, is_mmb, one context and one target mask)."""
        p = self.pred
        B, Nc = ctx_idx.shape
        x = p.predictor_embed(z) + jm.gather(self.ctx_pos.expand(B, -1, -1), ctx_idx)
        tok = p.mask_token.expand(B, tgt_idx.shape[1], -1) + jm.gather(self.tgt_pos.expand(B, -1, -1), tgt_idx)
        tok = p.policy_net(torch.cat([tok, self.aug.expand(B, tgt_idx.shape[1], -1)], -1))
        x = torch.cat([x, tok], 1)
        for blk in p.predictor_blocks:
            x = blk(x)
        return p.predictor_proj(p.predictor_norm(x)[:, Nc:])


@torch.no_grad()
def verify(ad, keys):
    """1) the adapter reproduces the model's own forward at 224; 2) the model's own JEPA loss (l2 on layer-normed
    targets, training masks) on held-out images under candidate preprocessings -- the training one should be lowest."""
    ad.set_resolution(224)
    x = torch.stack([ad.preprocess(jm.canvas(k), 224) for k in keys[:4]]).to(DEV)
    tgt_idx, ctx_idx = jm.block_indices(14, 6, 6, 1, DEV)
    sel = torch.tensor([0, 17, 40, 80], device=DEV)
    ti, ci = tgt_idx[sel], ctx_idx[sel]
    own = ad.pred(ad.enc(x, [ci]), ad.aug.expand(4, -1), [ci], [ti], is_mmb=True)
    mine = ad.predict(ad.encode_ctx(jm.gather(ad.tokens_ctx(x), ci)), ci, ti)
    ht = ad.tenc(x)
    print(f"verify: predictor max |diff| {(own - mine).abs().max().item():.2e} (scale {own.abs().mean().item():.2f}); "
          f"target encoder max |diff| {(ht - ad.targets(x)).abs().max().item():.2e}", flush=True)
    MMB = repo_module("masks/multi_multiblock").MaskCollator
    a = ad.args
    variants = {"imagenet+bicubic (training)": (IMAGENET, Image.BICUBIC, 1.0),
                "imagenet+bilinear": (IMAGENET, Image.BILINEAR, 1.0),
                "MIMIC stats (RAD-DINO)": (((0.5307,) * 3, (0.2583,) * 3), Image.BICUBIC, 1.0),
                "xray mean/std (MedMAE)": (((0.5056,) * 3, (0.252,) * 3), Image.BICUBIC, 1.0),
                "no normalisation": (((0, 0, 0), (1, 1, 1)), Image.BICUBIC, 1.0),
                "imagenet, contrast x0.25": (IMAGENET, Image.BICUBIC, 0.25)}
    imgs = [jm.canvas(k) for k in keys[:64]]
    for name, ((mean, std), interp, contrast) in variants.items():
        torch.manual_seed(0)
        col = MMB(input_size=224, patch_size=16, enc_mask_scale=tuple(a["enc_mask_scale"]),
                  pred_mask_scale=tuple(a["pred_mask_scale"]), aspect_ratio=(0.75, 1.5), nenc=a["mask_nenc"],
                  npred=a["mask_npred"], min_keep=a["mask_min_keep"], allow_overlap=False, merge=a["mask_merge"])
        losses, base = [], []
        for rep in range(4):
            for i in range(0, len(imgs), 16):
                xb = torch.stack([jm.to_input(im, 224, interp, mean, std, contrast) for im in imgs[i:i + 16]]).to(DEV)
                _, me, mp = col([torch.zeros(1)] * len(xb))
                me, mp = [m.to(DEV) for m in me], [m.to(DEV) for m in mp]
                hf = F.layer_norm(ad.tenc(xb), (768,))
                hbar = hf.mean(1, keepdim=True)
                h = torch.cat([jm.gather(hf, m) for m in mp])
                hb = torch.cat([hbar.expand(-1, m.shape[1], -1) for m in mp])
                z = ad.pred(ad.enc(xb, me), ad.aug.expand(len(xb), -1), me, mp, is_mmb=True)
                losses.append(F.mse_loss(z, h).item())
                base.append(F.mse_loss(hb, h).item())
        print(f"JEPA l2 loss, {name:28s}: {np.mean(losses):.4f}; MSE of the image-mean token {np.mean(base):.4f} "
              f"= skill ratio {np.mean(losses) / np.mean(base):.3f}", flush=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "search", "final", "smoke"])
    ap.add_argument("setting", nargs="*", help="search: R BH BW (one configuration); final: R BH BW NORM")
    a = ap.parse_args()
    ad = CheXWorld()
    held = open(HELD).read().split()
    print(f"CheXWorld epoch {ad.epoch}, {sum(p.numel() for p in ad.pred.parameters()) / 1e6:.1f}M predictor parameters",
          flush=True)
    if a.mode == "verify":
        verify(ad, held)
    elif a.mode == "smoke":                                  # a few images, timing of the heaviest configurations
        for cfg in SEARCH[:1] + [c for c in SEARCH if c[0] == 448][::3] + [c for c in SEARCH if c[3]][:1]:
            jm.run(ad, held[:4], [cfg], lambda c: {n: f"wmx_smoke/cxw_{cfg_tag(c)}_{n}" for n in jm.NORMS})
    elif a.mode == "search":
        cfgs = [parse(a.setting)] if a.setting else SEARCH   # one configuration per queue job, or all
        jm.run(ad, held, cfgs, lambda c: {n: f"wmx_search/cxw_{cfg_tag(c)}_{n}" for n in jm.NORMS})
    else:
        cfg, norm = parse(a.setting[:3]), a.setting[3]
        assert norm in jm.NORMS
        jm.run(ad, held + open(TEST).read().split(), [cfg], lambda c: {norm: "chexworld"})


def parse(setting):
    """R BH BW (the canvas resized to R) or T<R> BH BW (native crops of the canvas resized to R)."""
    r, bh, bw = setting[0], int(setting[1]), int(setting[2])
    return (int(r[1:]), bh, bw, TILE[1]) if r.startswith("T") else (int(r), bh, bw, None)


def cfg_tag(c):
    return f"{'T' if c[3] else 'R'}{c[0]}_b{c[1]}x{c[2]}"

if __name__ == "__main__":
    main()
