"""IterMask2 (Liang et al., MICCAI 2024; github.com/ZiyunLiang/IterMask2, MIT) on chest radiographs, as a baseline
model of normal appearance for PER. The official UNetModel (models/unet.py of the clone in
/raid/home/CAMCA/hj880/ext_src/IterMask2) is imported unchanged; the training losses, the training-mask generator and
the iterative mask refinement at test time are ported line by line from utils/train_util.py, datasets/brats2021.py,
datasets/brats_preprocess.py and models/test_model.py. What had to change for radiographs is listed in NOTES.md.

Two networks, trained separately as in the official README:
  first_iter          high-frequency image -> image (the first iteration's reconstruction)
  masked_autoencoder  (image with a random mask filled with Gaussian noise, high-frequency image) -> image
The high-frequency image keeps the FFT phase everywhere and the amplitude at radius >= 15 cycles per image; the amplitude
inside the radius is set to 1. Loss: MSE over the whole image; AdamW, lr 1e-4, weight decay 0, batch 32.

Data: the MIMIC-CXR "No Finding" frontal training images of the 512 world model (results/cxr_mimic.csv; 79,886, as in
logs/train_cxr_nll.log), the 512 x 512 letterboxed canvases cache/cxr/<key>.npy. The radiograph rectangle (the canvas
minus its letterbox padding) plays the role of the brain mask.

    python sota/itermask2/itermask2_cxr.py bench --size 512
    python sota/itermask2/itermask2_cxr.py train --model first_iter --size 512 --iters 90000
    python sota/itermask2/itermask2_cxr.py train --model masked_autoencoder --size 512 --iters 90000
    python sota/itermask2/itermask2_cxr.py thres --size 512          # validation threshold (official: 80th percentile)
    python sota/itermask2/itermask2_cxr.py infer --size 512 --keys sota/itermask2/dev_keys.txt
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import multivariate_normal

sys.dont_write_bytecode = True
SRC = "/raid/home/CAMCA/hj880/ext_src/IterMask2"
sys.path.insert(0, SRC)
from models.unet import UNetModel  # noqa: E402  (official, unchanged)

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(1, ROOT)
CACHE = f"{ROOT}/cache/cxr"
RUNS = f"{ROOT}/sota/itermask2/runs"
MAPS = f"{ROOT}/cache/cxr_maps"
G = 240              # the official masks are drawn on the 240 x 240 BraTS grid and resized to the model input
RADIUS = 15          # official high-pass radius (FFT bins = cycles per image)
# official config: channel_mult {64: (1, 2, 3, 4), 128: (1, 1, 2, 3, 4)}, attention at feature maps 32, 16, 8 and a
# bottleneck of 8 x 8; larger inputs get one more full-resolution level per doubling, so attention and bottleneck stay
CHANNEL_MULT = {128: (1, 1, 2, 3, 4), 256: (1, 1, 1, 2, 3, 4), 512: (1, 1, 1, 1, 2, 3, 4)}


def make_model(size, image_level_cond):
    """configs/brats_configs.py + utils/script_util.create_model."""
    return UNetModel(in_channels=1, model_channels=32, out_channels=1, num_res_blocks=2,
                     attention_resolutions=tuple(size // r for r in (32, 16, 8)), dropout=0.1,
                     channel_mult=CHANNEL_MULT[size], dims=2, use_checkpoint=False, use_fp16=False, num_heads=1,
                     num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=True, resblock_updown=True,
                     image_level_cond=image_level_cond)


def foreground(img):
    """The radiograph itself: the canvas minus the letterbox padding (rows and columns that are all zero)."""
    rows, cols = np.nonzero(img.any(1))[0], np.nonzero(img.any(0))[0]
    fg = np.zeros(img.shape, np.float32)
    if len(rows) and len(cols):
        fg[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = 1
    return fg


def normalize(img, fg, upper_per=0.9, lower_per=0.02):
    """datasets/brats_preprocess.py: the [lower_per, upper_per] intensity range of the brain (here the radiograph) to
    about [-3, 3], with the percentiles taken over the whole image as in the official code."""
    img = np.asarray(img, np.float64)
    perc_upper = fg.sum() * (1 - upper_per) / img.size
    perc_lower = fg.sum() * lower_per / img.size
    upper = np.percentile(img, (1 - perc_upper) * 100)
    lower = np.percentile(img, perc_lower * 100)
    rng = max(upper - lower, 1.0)
    return ((img - rng / 2) / rng * 6).astype(np.float32)


def official_mask(brain_mask):
    """datasets/brats2021.py BraTS2021Dataset.__getitem__, random mask generation, unchanged (240 x 240 grid;
    1 = visible, 0 = masked)."""
    num = random.randint(0, 2)
    covar11 = random.uniform(0.3, 10)
    covar22 = random.uniform(0.3, 10)
    covar12 = random.uniform(0, np.sqrt(covar11 * covar22)) * (-1 if random.randint(0, 1) == 0 else 1)
    target_area = np.where(brain_mask > 0)
    rand = np.random.randint(target_area[0].shape)
    if num == 0 or num == 1:
        x, y = np.mgrid[-12:12:.1, -12:12:.1]
        num1 = np.random.randint(np.sqrt(covar11 * covar22) * 100, np.sqrt(covar11 * covar22) * 200)
        num2 = np.random.randint(5, 10)
        num3 = np.random.randint(3, 10)
        mean1 = target_area[0][rand][0] / 240 * 24 - 12
        mean2 = target_area[1][rand][0] / 240 * 24 - 12
    else:
        x, y = np.mgrid[-48:48:.4, -48:48:.4]
        num1 = np.random.randint(np.sqrt(covar11 * covar22) * 5, np.sqrt(covar11 * covar22) * 20)
        num2 = np.random.randint(5, 10)
        num3 = np.random.randint(3, 10)
        mean1 = target_area[0][rand][0] / 240 * 96 - 48
        mean2 = target_area[1][rand][0] / 240 * 96 - 48
    pos = np.dstack((x, y))
    rv = multivariate_normal([mean1, mean2], [[covar11, covar12], [covar12, covar22]], allow_singular=True)
    gau_pdf = rv.pdf(pos)
    gau_pdf = gau_pdf / gau_pdf.max()
    p = gau_pdf[target_area[0], target_area[1]]
    l1, l2, l3, l4 = 4, 8, 16, 32
    d1 = np.random.choice(target_area[0].shape[0], num1 + num2 + num3, p=p.reshape(-1) / p.sum())
    d2 = np.random.choice(target_area[0].shape[0], num1 + num2 + num3, p=p.reshape(-1) / p.sum())
    gauss_mask = np.ones((brain_mask.shape[0], brain_mask.shape[1]))
    for i in range(d1.shape[0]):
        r, c = target_area[0][d1[i]], target_area[1][d2[i]]
        side = l1 if i < num1 else l2 if i < num1 + num2 else l3 if i < num1 + num2 + num3 else l4
        gauss_mask[r:r + side, c:c + side] = 0
    return gauss_mask.astype(np.float32)


def resize(a, size, mode):
    t = torch.from_numpy(np.ascontiguousarray(a, np.float32))[None, None]
    if t.shape[-1] == size and t.shape[-2] == size:
        return t[0]
    return F.interpolate(t, size=(size, size), mode=mode)[0]


def brain(key, img, fg, domain="radiograph"):
    """The brain mask's counterpart. Training: the radiograph. At test time (with the validation threshold computed
    the same way) optionally the thorax, i.e. the search domain of the address search (perr.cxr_anat.search_domain,
    the mask the candidates are drawn from), a free inference setting chosen on the held-out tenth."""
    if domain == "radiograph":
        return fg
    from perr.cxr_anat import search_domain
    from perr.cxr_wm import valid_region
    return search_domain(key, valid_region(img)).astype(np.float32)


def load(key, size, domain="radiograph"):
    """(normalised image, brain mask) at the model size, (1, size, size) each."""
    img = np.load(f"{CACHE}/{key}.npy")
    fg = foreground(img)
    return resize(normalize(img, fg), size, "area"), resize(brain(key, img, fg, domain), size, "area")


class Slices(torch.utils.data.Dataset):
    """The training images with the official random mask (drawn on the 240 grid, resized with area interpolation as
    monai Resized does in the official loader)."""

    def __init__(self, keys, size, domain="radiograph"):
        self.keys, self.size, self.domain = keys, size, domain

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        img = np.load(f"{CACHE}/{self.keys[i]}.npy")
        fg = foreground(img)
        bm = brain(self.keys[i], img, fg, self.domain)
        x = resize(normalize(img, fg), self.size, "area")
        bm240 = resize(bm, G, "nearest")[0].numpy()
        gm = resize(official_mask(bm240), self.size, "area")
        return {"input": x, "brainmask": resize(bm, self.size, "area"), "gauss_mask": gm}


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 31
    np.random.seed(s)
    random.seed(s + 1)


_HP = {}


def highfreq(x):
    """utils/train_util.py training_losses: high-frequency condition (amplitude kept at radius >= 15, set to 1 inside;
    phase kept everywhere), in float32."""
    h, w = x.shape[-2:]
    if (h, w, x.device) not in _HP:
        X, Y = np.ogrid[:h, :w]
        dist = np.sqrt((X - h // 2) ** 2 + (Y - w // 2) ** 2)
        _HP[(h, w, x.device)] = torch.from_numpy(dist >= RADIUS).to(x.device)
    mask = _HP[(h, w, x.device)]
    y = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
    amp = torch.abs(mask * y) * mask + torch.ones_like(y.real) * ~mask
    img = torch.fft.ifft2(torch.fft.ifftshift(amp * torch.exp(1j * torch.angle(y)), dim=(-2, -1)))
    return torch.real(img)


def training_loss(model, name, x, gauss_mask, brain_mask):
    x_cond = highfreq(x)
    if name == "masked_autoencoder":
        noise = torch.randn_like(x)
        m = (1 - gauss_mask) * brain_mask
        inp = torch.cat(((1 - m) * x + m * noise, x_cond), 1)
    else:
        inp = x_cond
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(inp.float())
    return ((x - pred.float()) ** 2).mean()


def train_keys():
    import pandas as pd
    d = pd.read_csv(f"{ROOT}/results/cxr_mimic.csv")
    normal = d[d["No Finding"] == 1.0]
    return normal[normal.split == "train"].key.tolist(), normal[normal.split == "validate"].key.tolist()


def train(a):
    torch.manual_seed(1)                                   # config.seed = 1
    keys, _ = train_keys()
    out = f"{RUNS}/{a.model}_{a.size}{a.tag}"
    os.makedirs(out, exist_ok=True)
    print(f"{len(keys)} training images, model {a.model}, input {a.size}", flush=True)
    model = make_model(a.size, a.model == "masked_autoencoder").cuda()
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    step = 0
    ck = sorted(f for f in os.listdir(out) if f.startswith("model") and f.endswith(".pt"))
    if ck and not a.bench:                                  # resume from the last saved checkpoint
        st = torch.load(f"{out}/{ck[-1]}", map_location="cuda")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"resumed from {ck[-1]} (step {step})", flush=True)
    dl = torch.utils.data.DataLoader(Slices(keys, a.size), batch_size=32, shuffle=True, drop_last=False,
                                     num_workers=a.workers, worker_init_fn=seed_worker, persistent_workers=True,
                                     prefetch_factor=4, pin_memory=True)
    model.train()
    t0, run = time.time(), []
    while step < a.iters:
        for b in dl:
            x, bm, gm = (b[k].cuda(non_blocking=True) for k in ("input", "brainmask", "gauss_mask"))
            loss = training_loss(model, a.model, x, gm, bm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            run.append(loss.item())
            if step % 100 == 0 or (a.bench and step % 10 == 0):
                print(f"step {step} loss {np.mean(run):.5f} {(time.time() - t0) / len(run):.3f} s/it "
                      f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
                run, t0 = [], time.time()
            if not a.bench and (step % 5000 == 0 or step == a.iters):
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                           f"{out}/model{step:06d}.pt")
                for f in sorted(f for f in os.listdir(out) if f.startswith("model") and f.endswith(".pt"))[:-1]:
                    if int(f[5:11]) % 30000:                  # keep every 30k and the last
                        os.remove(f"{out}/{f}")
            if step >= a.iters:
                break
    print("done", flush=True)


def load_model(name, size, step=None, tag=""):
    out = f"{RUNS}/{name}_{size}{tag}"
    f = f"model{step:06d}.pt" if step else sorted(f for f in os.listdir(out) if f.startswith("model"))[-1]
    m = make_model(size, name == "masked_autoencoder").cuda()
    m.load_state_dict(torch.load(f"{out}/{f}", map_location="cuda")["model"])
    print(f"loaded {name} {f}", flush=True)
    return m.eval()


@torch.no_grad()
def thres(a):
    """models/test_model.validation_thres + the threshold of iter_mask_refinement: masked autoencoder on the
    validation images with the training masks; the threshold is the k-th smallest masked error with
    k = N - 0.2 * (masked pixels), i.e. the 80th percentile of the error under the masks."""
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    _, keys = train_keys()
    mae = load_model("masked_autoencoder", a.size, a.step, a.tag)
    if a.max_val:
        keys = keys[:a.max_val]
    ds = Slices(keys, a.size, a.domain)
    losses, masks = [], []
    for i in range(0, len(keys), 32):
        items = [ds[j] for j in range(i, min(i + 32, len(keys)))]
        x = torch.stack([t["input"] for t in items]).cuda()
        gm = torch.stack([t["gauss_mask"] for t in items]).cuda()
        bm = torch.stack([t["brainmask"] for t in items]).cuda()
        x_cond = highfreq(x)
        m = (1 - gm) * bm
        inp = torch.cat(((1 - m) * x + m * torch.randn_like(x), x_cond), 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = mae(inp.float()).float()
        losses.append(((x - pred) ** 2).cpu())
        masks.append(m.cpu())
    loss, mask = torch.cat(losses), torch.cat(masks)
    lm = (loss * mask).flatten()
    k = int(mask.shape[0] * mask.shape[2] * mask.shape[3] - mask.sum().item() * 0.20)
    t = torch.kthvalue(lm.cuda(), k).values.item()
    json.dump({"thres": t, "n_val": len(keys), "size": a.size, "domain": a.domain, "masked_fraction": mask.mean().item()},
              open(a.thres_file or f"{RUNS}/thres_{a.size}{a.tag}.json", "w"))
    print(f"validation threshold {t:.5f} from {len(keys)} images (masked fraction {mask.mean().item():.3f})", flush=True)


@torch.no_grad()
def refine(mae, first, x, brain_mask, thres_validation, max_iter=300):
    """models/test_model.iter_mask_refinement, unchanged except the iteration cap (never reached in our runs; the
    count is returned)."""
    x_cond = highfreq(x)

    def run(model, inp):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(inp.float()).float()

    n = x.shape[0]
    flag = torch.ones(n, device=x.device)
    final_reconstruction, final_mask = torch.zeros_like(x), torch.zeros_like(x)
    iters = torch.zeros(n, dtype=torch.long)
    mask_inpaint_input = brain_mask
    i = 0
    while flag.sum() != 0:
        if i == 0:
            error_map = ((x - run(first, x_cond)) ** 2) * brain_mask
            thres = torch.zeros(n, device=x.device)
            for b in range(n):
                kthnum = brain_mask.shape[2] * brain_mask.shape[3] - brain_mask[b].sum() * 0.6
                thres[b] = torch.kthvalue(error_map[b, 0].flatten(), int(kthnum)).values
            thres = thres.view(n, 1, 1, 1)
        else:
            thres = thres_validation
        mask_new = torch.where(thres < error_map, 1.0, 0.0) * brain_mask
        ratio = (mask_inpaint_input.sum(dim=(1, 2, 3)) - mask_new.sum(dim=(1, 2, 3))) / mask_inpaint_input.sum(dim=(1, 2, 3))
        ratio = torch.where(torch.isnan(ratio), -1, ratio)
        update_flag = (ratio < 0.01) * (flag == 1)
        if i > max_iter:                                    # safety cap: finish whatever is left
            update_flag = flag == 1
        flag = flag * (~update_flag).int()
        if i > 0:
            idx = torch.where(update_flag)
            final_reconstruction[idx] = x0_pred[idx]
            final_mask[idx] = mask_inpaint_input[idx]
            iters[idx[0].cpu()] = i
        if flag.sum() == 0:
            break
        mask_inpaint_input = mask_new
        x_masked = (1 - mask_inpaint_input) * x + mask_inpaint_input * torch.randn_like(x)
        x0_pred = run(mae, torch.cat((x_masked, x_cond), 1))
        error_map = (x - (mask_inpaint_input * x0_pred + (1 - mask_inpaint_input) * x)) ** 2
        i += 1
    return final_mask, final_reconstruction, iters


def infer(a):
    """Anomaly maps (test.py: squared error between the image and its final reconstruction, inside the brain mask),
    upsampled to the 512 canvas, float16, cache/cxr_maps/<name>/<key>.npy."""
    torch.manual_seed(0)
    keys = [k for k in open(a.keys).read().split()]
    mae = load_model("masked_autoencoder", a.size, a.step, a.tag)
    first = load_model("first_iter", a.size, a.step, a.tag)
    th = json.load(open(a.thres_file or f"{RUNS}/thres_{a.size}{a.tag}.json"))
    assert th.get("domain", "radiograph") == a.domain, "validation threshold computed with another brain mask"
    t = th["thres"]
    keys = [k for k in keys if a.overwrite or not os.path.exists(f"{MAPS}/{a.name}/{k}.npy")]
    print(f"{len(keys)} images, validation threshold {t:.5f}", flush=True)
    stats, t0 = [], time.time()
    for i in range(0, len(keys), a.bs):
        ks = keys[i:i + a.bs]
        xs, fgs = zip(*[load(k, a.size, a.domain) for k in ks])
        x, fg = torch.stack(xs).cuda(), torch.stack(fgs).cuda()
        fm, rec, it = refine(mae, first, x, fg, t)
        err = ((x - rec) ** 2) * fg
        if a.size != 512:
            err = F.interpolate(err, size=(512, 512), mode="bilinear", align_corners=False)
        for k, e, m, f, n in zip(ks, err[:, 0].cpu().numpy(), fm.cpu(), fg.cpu(), it.tolist()):
            p = f"{MAPS}/{a.name}/{k}.npy"
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.save(p, e.astype(np.float16))
            stats.append({"key": k, "iters": n, "final_mask_frac": float(m.sum() / max(f.sum(), 1))})
        print(f"{i + len(ks)}/{len(keys)} ({time.time() - t0:.0f}s)", flush=True)
    if stats:
        with open(f"{RUNS}/infer_{a.name}_{os.path.basename(a.keys)}.jsonl", "a") as f:
            for s in stats:
                f.write(json.dumps(s) + "\n")
        print(f"iterations: mean {np.mean([s['iters'] for s in stats]):.1f}, max {max(s['iters'] for s in stats)}; "
              f"final mask fraction mean {np.mean([s['final_mask_frac'] for s in stats]):.3f}", flush=True)


@torch.no_grad()
def show(a):
    """Diagnostic panel for a few ChestX-Det images (public data only; never MIMIC): image, high-frequency input,
    first-iteration reconstruction, final reconstruction, final mask, anomaly map (rows), as one PNG at --png."""
    from PIL import Image
    keys = [k for k in open(a.keys).read().split() if k.startswith("cxdet/")][:a.bs]
    mae = load_model("masked_autoencoder", a.size, a.step, a.tag)
    first = load_model("first_iter", a.size, a.step, a.tag)
    t = json.load(open(a.thres_file or f"{RUNS}/thres_{a.size}{a.tag}.json"))["thres"]
    torch.manual_seed(0)
    xs, fgs = zip(*[load(k, a.size, a.domain) for k in keys])
    x, fg = torch.stack(xs).cuda(), torch.stack(fgs).cuda()
    fm, rec, it = refine(mae, first, x, fg, t)
    xc = highfreq(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        r0 = first(xc.float()).float()
    err = ((x - rec) ** 2) * fg
    g = lambda v, lo, hi: (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    rows = []
    for i in range(len(keys)):
        e = err[i, 0].cpu().numpy()
        tiles = [g(x[i, 0].cpu().numpy(), -3, 3.5), g(xc[i, 0].cpu().numpy(), -3, 3.5), g(r0[i, 0].cpu().numpy(), -3, 3.5),
                 g(rec[i, 0].cpu().numpy(), -3, 3.5), g(fm[i, 0].cpu().numpy(), 0, 1),
                 g(e, 0, max(float(np.percentile(e[e > 0], 99.5)) if (e > 0).any() else 1, 1e-6))]
        rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(a.png)
    print(a.png, "iterations", it.tolist(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "bench", "thres", "infer", "show"])
    ap.add_argument("--model", choices=["first_iter", "masked_autoencoder"], default="masked_autoencoder")
    ap.add_argument("--size", type=int, default=512, choices=sorted(CHANNEL_MULT))
    ap.add_argument("--iters", type=int, default=90000, help="official: the 90k checkpoint is the one tested")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--step", type=int, default=None, help="checkpoint to load (default: the last)")
    ap.add_argument("--keys", help="file of image keys (infer)")
    ap.add_argument("--name", default="itermask2_raw", help="output folder under cache/cxr_maps (infer)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tag", default="", help="run-name suffix (smoke tests)")
    ap.add_argument("--domain", default="radiograph", choices=["radiograph", "thorax"],
                    help="brain mask at test time and for the validation threshold (training: radiograph)")
    ap.add_argument("--thres-file", default=None, help="validation threshold file (default runs/thres_<size><tag>.json)")
    ap.add_argument("--max-val", type=int, default=0, help="only the first N validation images (smoke tests)")
    ap.add_argument("--png", default=None, help="output of show")
    a = ap.parse_args()
    a.bench = a.cmd == "bench"
    if a.cmd in ("train", "bench"):
        if a.bench:
            a.iters = 60
        train(a)
    elif a.cmd == "thres":
        thres(a)
    elif a.cmd == "show":
        show(a)
    else:
        infer(a)


if __name__ == "__main__":
    main()
