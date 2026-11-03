"""GPU timing of the MAD-AD training step components at 512 (UNet_L on 64 x 64 latents, VAE encoder), fp32/TF32 vs
bf16 autocast, NCHW vs channels_last; for sizing the shortened schedule (NOTES.md)."""
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import train_madad as tm      # noqa: E402

tm.patch_attention()
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
model = tm.UNET_models["UNet_L"](latent_size=64).to(dev)
vae = tm.load_vae(dev)
diff = tm.create_diffusion(timestep_respacing="ddim10", predict_xstart=True, sigma_small=False, learn_sigma=False,
                           diffusion_steps=10)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)


def timeit(fn, n=6):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


for micro in (8, 16, 24):
    for bf16 in (True, False):
        for cl in (False,):
            if micro > 8 and not bf16:
                continue
            amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)
            mf = torch.channels_last if cl else torch.contiguous_format
            model.to(memory_format=mf)
            vae.to(memory_format=mf)
            x = torch.randn(micro, 1, 512, 512, device=dev).clamp(-1, 1).contiguous(memory_format=mf)
            z = torch.randn(micro, 4, 64, 64, device=dev)
            bm = torch.ones(micro, 512, 512, device=dev)

            def enc():
                with torch.no_grad(), amp:
                    vae.encode(x).sample()

            def step():
                t = torch.randint(0, 10, (micro,), device=dev)
                mask = tm.make_mask(z, bm, 0.4, [0.2, 0.3, 0.3, 0.2])
                with amp:
                    ld = diff.training_losses(model, z.contiguous(memory_format=mf), t, {"mask": mask},
                                              noise=torch.randn_like(z))
                ld["loss"].float().mean().backward()
                opt.zero_grad(set_to_none=True)

            def maskonly():
                tm.make_mask(z, bm, 0.4, [0.2, 0.3, 0.3, 0.2])

            def full():
                with torch.no_grad(), amp:
                    zz = vae.encode(x).sample().mul_(tm.SCALE).float()
                t = torch.randint(0, 10, (micro,), device=dev)
                mask = tm.make_mask(zz, bm, 0.4, [0.2, 0.3, 0.3, 0.2])
                with amp:
                    ld = diff.training_losses(model, zz, t, {"mask": mask}, noise=torch.randn_like(zz))
                ld["loss"].float().mean().backward()
                opt.zero_grad(set_to_none=True)

            try:
                tf = timeit(full)
                print(f"micro {micro} bf16 {bf16}: full micro-step {tf * 1e3:.0f} ms -> {micro / tf:.1f} img/s", flush=True)
                te, ts, tmk = timeit(enc), timeit(step), timeit(maskonly)
                print(f"micro {micro} bf16 {bf16} channels_last {cl}: VAE encode {te * 1e3:.0f} ms, UNet step {ts * 1e3:.0f} ms, "
                      f"mask {tmk * 1e3:.1f} ms -> {micro / (te + ts):.1f} img/s; peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB",
                      flush=True)
            except torch.cuda.OutOfMemoryError as e:
                print(f"micro {micro} bf16 {bf16} channels_last {cl}: OOM", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
os.system("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader")
