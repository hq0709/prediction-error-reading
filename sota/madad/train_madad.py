"""MAD-AD (Beizaee et al., IPMI 2025; github.com/farzad-bz/MAD-AD @ df05cd0) trained one-class on the MIMIC-CXR
"No Finding" frontal training images of the 512 world model (79,886 canvases, cache/cxr/<key>.npy).

Follows train-MAD-AD.py (README command: UNet_L, --augmentation True, defaults otherwise): the repo's UNet (models.py),
masked latent diffusion (diffusion_x0, 10 steps, x0 prediction, cosine schedule), random latent masks (ratio U[0, 0.4],
patch sizes {1,2,4,8} drawn independently along x and y), the body mask forcing background latents to stay clean, batch
96, AdamW lr 4e-5 / wd 1e-4, cosine-with-warmup over 1.2 x epochs (warm-up 5% of the epochs), its albumentations augmentation,
and its validation (every ckpt_every epochs; best_mse.pt, last.pt). The repo's random_mask and diffusion code are
imported unchanged. Adaptations (NOTES.md): radiograph canvases instead of brain-MRI PNG slices (the "brain mask" is
the radiograph's non-padding region), 512 x 512 input (latent 64 x 64), the epoch count shortened to the budget (so the
same schedule shape is evaluated per optimiser step rather than per epoch: with 15 instead of 200 epochs the repo's
per-epoch stepping would spend the whole first epoch at lr 0), bf16 autocast (--bf16), host-device syncs removed,
attention computed with torch SDPA (same math), micro-batches with gradient accumulation to the global batch of 96
when a card cannot hold 96, deterministic seeding, a final-epoch checkpoint, and resumable state.

    python -I -B sota/madad/train_madad.py --out sota/madad/runs/unet_l_512 --epochs 15 --warmup-epochs 0.75 \
        --ckpt-every 2 --micro 32 --bf16
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
MADAD = "/raid/home/CAMCA/hj880/ext_src/madad"
ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.append(MADAD)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import albumentations as A                                    # noqa: E402
from torch.utils.data import DataLoader, Dataset             # noqa: E402
from transformers import get_cosine_schedule_with_warmup     # noqa: E402

from diffusion_x0 import create_diffusion                    # noqa: E402  (MAD-AD repo)
from models import UNET_models                               # noqa: E402  (MAD-AD repo)
from vae import SCALE, load_vae                              # noqa: E402

_spec = importlib.util.spec_from_file_location("madad_train", f"{MADAD}/train-MAD-AD.py")
_madad_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_madad_train)                       # defines functions only (main is behind __name__ guard)
random_mask = _madad_train.random_mask                       # the repo's mask sampler, unchanged

torch.backends.cuda.matmul.allow_tf32 = True                 # as in train-MAD-AD.py
torch.backends.cudnn.allow_tf32 = True


def sdpa_legacy_attention(self, qkv):
    """QKVAttentionLegacy.forward of the repo computed with torch SDPA: softmax(q.k / sqrt(ch)) v, identical math and
    head grouping. 4-D contiguous (batch, heads, tokens, ch) inputs so that the fused (flash / memory-efficient) kernels
    are used instead of materialising the 4096 x 4096 attention matrices of the 64 x 64 latent."""
    bs, width, length = qkv.shape
    ch = width // (3 * self.n_heads)
    q, k, v = qkv.reshape(bs, self.n_heads, ch * 3, length).split(ch, dim=2)
    q, k, v = (t.transpose(-1, -2).contiguous() for t in (q, k, v))
    a = F.scaled_dot_product_attention(q, k, v)
    return a.transpose(-1, -2).reshape(bs, -1, length)


def checkpointed_attention_block(self, x):
    """AttentionBlock.forward of the repo (always gradient-checkpointed) with torch's non-reentrant checkpoint, which
    also restores the autocast state when recomputing; values and gradients are those of the repo's version."""
    return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)


def patch_attention():
    import torch.utils.checkpoint                                  # noqa: F401
    from ldm.modules.diffusionmodules import openaimodel_medical as om
    om.QKVAttentionLegacy.forward = sdpa_legacy_attention
    om.AttentionBlock.forward = checkpointed_attention_block
    patch_syncs()


_DEV_CACHE = {}


def _on_device(key, make, device):
    k = (key, str(device))
    if k not in _DEV_CACHE:
        _DEV_CACHE[k] = make().to(device)
    return _DEV_CACHE[k]


def patch_syncs():
    """The repo builds small tensors on the CPU and copies them to the GPU inside every forward pass (diffusion
    constants, the timestep map, the sinusoidal frequencies, the VAE's sampling noise). Each such copy waits for the
    GPU to drain, which on a shared, time-sliced card costs a time slice. Same values, computed once per device (the
    frequencies and constants bit-identically on the CPU as before; the VAE noise drawn directly on the GPU)."""
    from diffusion_x0 import gaussian_diffusion as gd
    from diffusion_x0 import respace
    from ldm.modules.diffusionmodules import openaimodel_medical as om
    from ldm.modules.distributions import distributions as dd

    def extract(arr, timesteps, broadcast_shape):
        arr = np.asarray(arr)
        t = _on_device(("arr", arr.dtype.str, arr.tobytes()), lambda: torch.from_numpy(arr.copy()), timesteps.device)
        res = t[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res + torch.zeros(broadcast_shape, device=timesteps.device)

    def wrapped_call(self, x, ts, **kwargs):
        m = _on_device(("map", tuple(self.timestep_map), str(ts.dtype)),
                       lambda: torch.tensor(self.timestep_map, dtype=ts.dtype), ts.device)
        return self.model(x, m[ts], **kwargs)

    def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
        assert not repeat_only
        half = dim // 2
        freqs = _on_device(("freqs", dim, max_period), lambda: torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half), timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def sample(self):
        return self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)

    gd._extract_into_tensor = extract
    respace._WrappedModel.__call__ = wrapped_call
    om.timestep_embedding = timestep_embedding
    dd.DiagonalGaussianDistribution.sample = sample


def keys_of(split):
    import pandas as pd
    d = pd.read_csv(f"{ROOT}/results/cxr_mimic.csv")
    n = d[d["No Finding"] == 1.0]
    return n[n.split == split].key.tolist()


def body_mask(img):
    """The radiograph's non-padding region (bounding box of non-zero rows / columns): the counterpart of the brain mask."""
    rows, cols = np.nonzero(img.any(1))[0], np.nonzero(img.any(0))[0]
    m = np.zeros(img.shape, np.int32)
    if len(rows) and len(cols):
        m[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = 1
    return m


class CXRDataset(Dataset):
    """MedicalDataset.__getitem__ of the repo for 512 canvases: same augmentation, background zeroed, [-1, 1]."""

    def __init__(self, keys, image_size=512, augment=False):
        self.keys, self.image_size, self.augment = keys, image_size, augment
        if augment:
            self.aug = A.Compose([A.Affine(rotate=5, p=0.5),
                                  A.Affine(translate_px=int(image_size // 32), p=0.5),
                                  A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        img = np.load(f"{ROOT}/cache/cxr/{self.keys[i]}.npy")
        if img.shape[0] != self.image_size:
            from PIL import Image
            img = np.array(Image.fromarray(img).resize((self.image_size, self.image_size)), np.uint8)
        mask = body_mask(img)
        img = img.copy()
        img[mask == 0] = 0
        if self.augment:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
            img[mask == 0] = 0
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None] * 2 - 1      # ToTensor + Normalize(0.5, 0.5)
        return x, torch.from_numpy(mask.astype(np.float32))


def make_mask(x, bmask, ratio_hi, probs):
    """The repo's training-time mask: random_mask (run on the CPU, then one pinned asynchronous copy, instead of one
    synchronising copy per sample), then background latents forced to 1 (never noised)."""
    sx = np.random.choice([1, 2, 4, 8], 1, p=probs).item()
    sy = np.random.choice([1, 2, 4, 8], 1, p=probs).item()
    ratios = np.random.uniform(low=0.0, high=ratio_hi, size=x.shape[0])
    mask = random_mask(torch.empty(x.shape), mask_ratios=ratios, mask_patch_size_x=sx, mask_patch_size_y=sy)
    if x.is_cuda:
        mask = mask.pin_memory().to(x.device, non_blocking=True)
    br = (F.avg_pool2d(bmask.unsqueeze(1), 16, 16) < 0.9).to(torch.int32)
    br = torch.repeat_interleave(br, x.shape[1], 1)
    br = torch.repeat_interleave(torch.repeat_interleave(br, 2, 2), 2, 3)
    return mask.masked_fill_(br == 1, 1.0)            # = mask[br == 1] = 1, without the synchronising nonzero()


def seed_all(s):
    random.seed(s)
    np.random.seed(s % 2 ** 32)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="UNet_L")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--warmup-epochs", type=float, default=10)
    ap.add_argument("--global-batch-size", type=int, default=96)
    ap.add_argument("--micro", type=int, default=96, help="micro-batch; gradients accumulate to the global batch")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--global-seed", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (default: fp32 with TF32 as the repo)")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after this many optimizer steps")
    ap.add_argument("--train-limit", type=int, default=0, help="smoke test: use only this many training images")
    a = ap.parse_args()
    assert a.global_batch_size % a.micro == 0 and a.image_size % 8 == 0
    os.makedirs(f"{a.out}/checkpoints", exist_ok=True)
    patch_attention()
    dev = torch.device("cuda")
    seed_all(a.global_seed)

    train_keys, val_keys = keys_of("train"), keys_of("validate")
    if a.train_limit:
        train_keys = train_keys[:a.train_limit]
    print(f"{len(train_keys)} training images, {len(val_keys)} validation images (MIMIC-CXR No Finding, frontal)", flush=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)

    model = UNET_models[a.model](latent_size=a.image_size // 8).to(dev)
    diffusion = create_diffusion(timestep_respacing="ddim10", predict_xstart=True, sigma_small=False, learn_sigma=False,
                                 diffusion_steps=10)
    vae = load_vae(dev)
    print(f"Number of Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = math.ceil(len(train_keys) / a.global_batch_size)          # optimiser steps per epoch (drop_last=False)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=round(a.warmup_epochs * spe),
                                            num_training_steps=round(a.epochs * 1.2 * spe))   # stepped every optimiser step

    start_epoch, best_mse, history = 0, float("inf"), []
    state_path = f"{a.out}/checkpoints/state.pt"
    if os.path.exists(state_path):                                   # resume
        st = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, best_mse = int(st["epoch"]) + 1, float(st["best_mse"])
        history = json.load(open(f"{a.out}/history.json")) if os.path.exists(f"{a.out}/history.json") else []
        print(f"resumed after epoch {start_epoch - 1} (best val mse {best_mse:.5f})", flush=True)

    ds = CXRDataset(train_keys, a.image_size, augment=True)
    vds = CXRDataset(val_keys, a.image_size, augment=False)
    n_micro = a.global_batch_size // a.micro
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16)
    steps = 0
    for epoch in range(start_epoch, a.epochs):
        seed_all(a.global_seed * 1000 + epoch)
        torch.backends.cudnn.deterministic = False         # set_random_seed() of the repo, called every epoch
        torch.backends.cudnn.benchmark = True
        model.train()
        g = torch.Generator()
        g.manual_seed(a.global_seed * 1000 + epoch)
        loader = DataLoader(ds, batch_size=a.global_batch_size, shuffle=True, generator=g, num_workers=a.workers,
                            drop_last=False, pin_memory=True, persistent_workers=False, prefetch_factor=4)
        t0, run_loss, run_mse, run_mp, n_log = time.time(), 0.0, 0.0, 0.0, 0
        for ii, (x_all, bm_all) in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            parts = list(zip(x_all.split(a.micro), bm_all.split(a.micro)))
            for x, bm in parts:
                x, bm = x.to(dev, non_blocking=True), bm.to(dev, non_blocking=True)
                with torch.no_grad(), amp:
                    z = vae.encode(x).sample().mul_(SCALE).float()
                t = torch.randint(0, diffusion.num_timesteps, (z.shape[0],), device=dev)
                mask = make_mask(z, bm, 0.4, [0.2, 0.3, 0.3, 0.2])
                noise = torch.randn_like(z)
                with amp:
                    ld = diffusion.training_losses(model, z, t, {"mask": mask}, noise=noise)
                loss = ld["loss"].float().mean()
                (loss * (x.shape[0] / x_all.shape[0])).backward()
                w = x.shape[0] / x_all.shape[0]
                run_loss += loss.detach() * w                       # stays on the GPU until the log line
                run_mse += ld["mse"].detach().float().mean() * w
                run_mp += ld["mask_prediction"].detach().float().mean() * w
            opt.step()
            sched.step()
            n_log += 1
            steps += 1
            if steps % a.log_every == 0:
                torch.cuda.synchronize()
                print(f"epoch {epoch} step {ii + 1}/{len(loader)} MSE {float(run_mse) / n_log:.4f} Mask_pred {float(run_mp) / n_log:.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {n_log * a.global_batch_size / (time.time() - t0):.1f} img/s "
                      f"mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
                t0, run_loss, run_mse, run_mp, n_log = time.time(), 0.0, 0.0, 0.0, 0
            if a.max_steps and steps >= a.max_steps:
                break
        last = epoch == a.epochs - 1 or (a.max_steps and steps >= a.max_steps)
        if (epoch % a.ckpt_every == 0 and epoch != 0) or last:
            model.eval()
            seed_all(a.global_seed)                                   # fix_seed(global_seed) as in the repo
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            vl = DataLoader(vds, batch_size=a.global_batch_size, shuffle=False, num_workers=a.workers)
            mse_sum, mp_sum, nb = 0.0, 0.0, 0
            with torch.no_grad():
                for x_all, bm_all in vl:
                    for x, bm in zip(x_all.split(a.micro), bm_all.split(a.micro)):
                        x, bm = x.to(dev), bm.to(dev)
                        with amp:
                            z = vae.encode(x).sample().mul_(SCALE).float()
                        t = torch.randint(0, diffusion.num_timesteps, (z.shape[0],), device=dev)
                        mask = make_mask(z, bm, 0.3, [0.3, 0.3, 0.2, 0.2])
                        with amp:
                            ld = diffusion.training_losses(model, z, t, {"mask": mask}, noise=torch.randn_like(z))
                        w = x.shape[0] / x_all.shape[0]
                        mse_sum += ld["mse"].detach().float().mean() * w
                        mp_sum += ld["mask_prediction"].detach().float().mean() * w
                    nb += 1
            vmse, vmp = float(mse_sum) / nb, float(mp_sum) / nb
            history.append({"epoch": epoch, "val_mse": vmse, "val_mask_pred": vmp, "time": time.time()})
            print(f"{'*' * 30}\nValidation epoch {epoch}: MSE {vmse:.5f} Mask_pred {vmp:.5f}\n{'*' * 30}", flush=True)
            ck = {"model": model.state_dict(), "epoch": epoch, "args": vars(a)}
            if vmse < best_mse:
                best_mse = vmse
                torch.save(ck, f"{a.out}/checkpoints/best_mse.pt")
            torch.save(ck, f"{a.out}/checkpoints/last.pt")
            json.dump(history, open(f"{a.out}/history.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                    "best_mse": best_mse}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        if a.max_steps and steps >= a.max_steps:
            break
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
