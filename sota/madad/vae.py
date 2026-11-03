"""MAD-AD's VAE (farzadbz/Medical-VAE, KL-f8, 1 channel) rebuilt from the repo's own ldm Encoder/Decoder code and the
tensors extracted by extract_vae.py (no pickled code is executed). Same encode/decode interface as the pickled
AutoencoderKL that train-MAD-AD.py / evaluate-MAD-AD.py use: encode(x) -> DiagonalGaussianDistribution, decode(z).
"""
import json
import os
import sys

import torch

MADAD = "/raid/home/CAMCA/hj880/ext_src/madad"
VAE_DIR = "/raid/home/CAMCA/hj880/ext_src/madad_vae"
if MADAD not in sys.path:
    sys.path.append(MADAD)

from ldm.modules.diffusionmodules.model import Decoder, Encoder          # noqa: E402  (MAD-AD repo code)
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution  # noqa: E402

SCALE = 0.18215          # latent scaling used throughout MAD-AD


class VAE(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dd = {k: cfg[k] for k in ("ch", "out_ch", "ch_mult", "num_res_blocks", "attn_resolutions", "dropout",
                                  "in_channels", "resolution", "z_channels", "double_z", "give_pre_end", "tanh_out")}
        self.encoder = Encoder(**dd)
        self.decoder = Decoder(**dd)
        self.quant_conv = torch.nn.Conv2d(2 * cfg["z_channels"], 2 * cfg["embed_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(cfg["embed_dim"], cfg["z_channels"], 1)

    def encode(self, x):
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))


def load_vae(device="cpu"):
    cfg = json.load(open(f"{VAE_DIR}/klf8_vae_config.json"))
    vae = VAE(cfg)
    sd = torch.load(f"{VAE_DIR}/klf8_vae_state.pt", map_location="cpu", weights_only=True)
    vae.load_state_dict(sd, strict=True)
    return vae.eval().requires_grad_(False).to(device)


if __name__ == "__main__":       # CPU sanity check: reconstruction error on a few cached radiographs
    import numpy as np
    keys = sys.argv[1:] or ["cxdet/36200"]
    vae = load_vae()
    for k in keys:
        a = np.load(f"/raid/home/CAMCA/hj880/perr/cache/cxr/{k}.npy").astype(np.float32) / 255.0
        for size in (256, 512):
            x = torch.from_numpy(a)[None, None]
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="area") if size != 512 else x
            x = x * 2 - 1
            with torch.no_grad():
                z = vae.encode(x).mean
                r = vae.decode(z).clamp(-1, 1)
            err = (r - x).abs()
            print(f"{k} {size}: latent {tuple(z.shape)} std {z.mul(SCALE).std():.3f}  |recon-x| mean {err.mean():.4f} "
                  f"p99 {err.flatten().quantile(0.99):.4f} (in [-1,1] units)", flush=True)
