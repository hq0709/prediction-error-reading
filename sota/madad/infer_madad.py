"""MAD-AD anomaly maps of 512 radiograph canvases, as evaluate-MAD-AD.py computes them: the VAE posterior mean of the
image (x 0.18215) is taken as z'_T, the masked DDIM reverse process (diffusion_x0 ddim_sample_loop: at every step the
predicted mask decides which latents are replaced by the DDIM update, eta = sigma_t) corrects it, and the map is the
absolute difference between the decoded corrected latent and the decoded input latent, clipped at gamma, divided by
gamma and Gaussian-smoothed (repo: gamma 0.25, sigma 5 px at 256 = 10 px on the 512 canvas).

Free inference settings (chosen on the held-out tenth, dev_recall.py): #DDIM steps N (respacing of the 10-step
process, the paper's Table 4 ablation), the start step s <= N (s < N starts the reverse process at a lower noise
level; the repo uses s = N), eta, gamma and sigma.

    # dev sweep: raw |difference| maps per sampling setting -> cache/uad_dev/madad/<setting>/<key>.npy
    python -I -B sota/madad/infer_madad.py --run sota/madad/runs/unet_l_512 --keys sota/madad/keys_dev.txt --sweep
    # final maps: cache/cxr_maps/madad/<key>.npy
    python -I -B sota/madad/infer_madad.py --run sota/madad/runs/unet_l_512 --keys sota/madad/keys_all.txt \
        --steps 10 --start 10 --eta 0.5 --gamma 0.25 --sigma 10 --name madad
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import train_madad as tm                             # noqa: E402  (repo imports, attention patch, dataset)

ROOT = tm.ROOT
SWEEP = [(10, 10, 0.5), (10, 7, 0.5), (10, 5, 0.5), (10, 3, 0.5), (5, 5, 0.5), (5, 3, 0.5), (2, 2, 0.5), (2, 1, 0.5),
         (10, 10, 0.0), (10, 10, 1.0)]


def setting_name(n, s, eta):
    return f"n{n}_s{s}_eta{eta:g}"


def load_model(run, ckpt, dev):
    args = json.load(open(f"{run}/args.json"))
    model = tm.UNET_models[args["model"]](latent_size=args["image_size"] // 8)
    ck = torch.load(f"{run}/checkpoints/{ckpt}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"], strict=True)
    print(f"{run}/checkpoints/{ckpt}.pt: epoch {ck['epoch']}", flush=True)
    return model.to(dev).eval(), args


@torch.no_grad()
def residuals(model, vae, keys, settings, image_size=512, bs=16, dev="cuda", bf16=False):
    """Yield (key, {setting: |decode(corrected) - decode(z)| as 512 x 512 float32}) for every key."""
    ds = tm.CXRDataset(keys, image_size, augment=False)
    diffs = {st: tm.create_diffusion(timestep_respacing=f"ddim{st[0]}", predict_xstart=True, sigma_small=False,
                                     learn_sigma=False, diffusion_steps=10) for st in settings}
    amp = torch.autocast(dev.type if hasattr(dev, "type") else "cuda", dtype=torch.bfloat16, enabled=bf16)
    for i in range(0, len(keys), bs):
        x = torch.stack([ds[j][0] for j in range(i, min(i + bs, len(keys)))]).to(dev)
        with amp:
            z = vae.encode(x).mean.float() * tm.SCALE
            x0 = vae.decode(z / tm.SCALE).float()
        out = [{} for _ in range(x.shape[0])]
        for st in settings:
            n, s, eta = st
            torch.manual_seed(zlib.crc32(keys[i].encode()) + 1000 * n + s)     # eta > 0 samples noise: fixed per batch
            with amp:
                zc = diffs[st].ddim_sample_loop(model, z.shape, noise=z, clip_denoised=False, t=s,
                                                model_kwargs={"mask": torch.ones_like(z)}, progress=False, device=dev,
                                                eta=eta)
                xs = vae.decode(zc.float() / tm.SCALE).float()
            d = (xs - x0).abs().mean(1)                                        # channel mean (1 channel)
            if d.shape[-1] != 512:
                d = torch.nn.functional.interpolate(d[:, None], size=512, mode="bilinear", align_corners=False)[:, 0]
            for b in range(x.shape[0]):
                out[b][st] = d[b].cpu().numpy()
        for b in range(x.shape[0]):
            yield keys[i + b], out[b]


def post(d, gamma, sigma):
    """evaluate-MAD-AD.py: clip(|diff|, 0, gamma) / gamma, then scipy gaussian_filter(sigma)."""
    a = np.clip(d, 0.0, gamma) / gamma if gamma > 0 else d
    return ndi.gaussian_filter(a, sigma) if sigma > 0 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--keys", required=True)
    ap.add_argument("--sweep", action="store_true", help="dev: raw maps of every SWEEP setting to cache/uad_dev/madad")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--start", type=int, default=0, help="start step s (default: s = steps, as the repo)")
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=10.0, help="px on the 512 canvas (repo: 5 px at 256)")
    ap.add_argument("--name", default="madad")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-dir", default="cache/uad_dev/madad", help="where --sweep writes its raw maps")
    ap.add_argument("--limit-settings", type=int, default=0, help="test: only the first n SWEEP settings")
    ap.add_argument("--chosen", default=None, help="select_madad.py output: take steps/start/eta/gamma/sigma from it")
    a = ap.parse_args()
    if a.chosen:
        c = json.load(open(a.chosen))
        a.steps, a.start, a.eta, a.gamma, a.sigma = c["steps"], c["start"], c["eta"], c["gamma"], c["sigma"]
        print(f"settings from {a.chosen}: {c['variant']}", flush=True)
    dev = torch.device(a.device)
    tm.patch_attention()
    model, args = load_model(a.run, a.ckpt, dev)
    vae = tm.load_vae(dev)
    keys = open(a.keys).read().split()
    settings = (SWEEP[:a.limit_settings] if a.limit_settings else SWEEP) if a.sweep else [(a.steps, a.start or a.steps, a.eta)]
    t0, n = time.time(), 0
    for key, res in residuals(model, vae, keys, settings, args["image_size"], a.bs, dev, a.bf16):
        for st, d in res.items():
            if a.sweep:
                p = f"{ROOT}/{a.dev_dir}/{setting_name(*st)}/{key}.npy"
                arr = d
            else:
                p = f"{ROOT}/cache/cxr_maps/{a.name}/{key}.npy"
                arr = post(d, a.gamma, a.sigma)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.save(p, arr.astype(np.float16))
        n += 1
        if n % 100 == 0:
            print(f"{n}/{len(keys)} images ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {n} images, settings {[setting_name(*s) for s in settings]}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
