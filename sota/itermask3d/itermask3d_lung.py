"""IterMask3D (Liang et al., 2025; github.com/ZiyunLiang/IterMask3D, MIT) on lung CT, as a baseline model of normal
appearance for PER. The official network (Nets/unet3d.py UNET3D with Nets/updated_res_units.py, clone in
/raid/home/CAMCA/hj880/ext_src/IterMask3D) is imported unchanged; the training-mask generator (dataloader.py), the
training step (train.py) and the test-time iterative mask refinement (test.py) are ported line by line. What had to
change for 1 mm lung CT is listed in NOTES.md.

Data: the 568 LUNA16 training scans of the lung world model in use (runs_wm/ctx_free/split.json "train"; the world
model "nllfree" was trained on them without lesion annotations, and so is this model: crops are placed anywhere in
the lung, nodules included), the 1 mm volumes cache/iso/<uid>.vol.npy (HU) and their grown lung domains
cache/iso/<uid>.dom.npy (domain + 5 mm rim), which play the role of the brain mask. Intensities: the lung window
(-1000, 400) HU mapped linearly to [-3, 3] and clipped (the official normalisation script is not in the repository;
IterMask2 maps the brain's intensity range to [-3, 3], which keeps the N(0, 1) noise fill and the threshold-fit limit
on the scale they were designed for). Training input: random 192^3 crops (the official volume size) centred on the
label-free random domain points cache/iso/<uid>.centres_all.npy.

    python sota/itermask3d/itermask3d_lung.py bench
    python sota/itermask3d/itermask3d_lung.py train --epochs 500
    python sota/itermask3d/itermask3d_lung.py infer --uids <file> --gammas 0.01 0.05 [--save-maps]
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
from scipy.ndimage import gaussian_filter, zoom
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal, ortho_group

sys.dont_write_bytecode = True
sys.path.insert(0, "/raid/home/CAMCA/hj880/ext_src/IterMask3D")
from Nets.unet3d import UNET3D  # noqa: E402  (official, unchanged)

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(1, ROOT)
from perr.context_wm import ISO, load_iso, to_original  # noqa: E402

RUNS = f"{ROOT}/sota/itermask3d/runs"
MAPS = f"{ROOT}/cache/itermask3d"
CROP = 192                 # official input size
RADIUS = 15                # official high-pass radius at 192 voxels (= 15 / 192 cycles per mm at 1 mm)
HU_LO, HU_HI = -1000.0, 400.0


def norm(hu):
    return (np.clip((np.asarray(hu, np.float32) - HU_LO) / (HU_HI - HU_LO), 0, 1) * 6 - 3).astype(np.float32)


_HP = {}


def highfreq(x):
    """dataloader.py high-frequency input: FFT amplitude kept at radius >= 15, set to 1 inside; phase kept. For a
    volume of any shape the radius is the official one in cycles per mm (15 / 192 along every axis)."""
    shape = tuple(x.shape[-3:])
    key = (shape, x.device)
    if key not in _HP:
        g = np.ogrid[:shape[0], :shape[1], :shape[2]]
        d2 = sum(((g[i] - shape[i] // 2) * (CROP / shape[i])) ** 2 for i in range(3))
        _HP[key] = torch.from_numpy(np.sqrt(d2) >= RADIUS).to(x.device)
    mask = _HP[key]
    y = torch.fft.fftshift(torch.fft.fftn(x.float(), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    amp = torch.abs(mask * y) * mask + torch.ones_like(y.real) * ~mask
    img = torch.fft.ifftn(torch.fft.ifftshift(amp * torch.exp(1j * torch.angle(y)), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    return torch.real(img)


def official_mask(brainmask, shape):
    """dataloader.py Dataset3D.__getitem__, training mask generation, unchanged (1 = visible, 0 = masked)."""
    num = random.randint(0, 3)
    orth_mat = ortho_group.rvs(3)
    lambda1 = random.uniform(0.3, 10)
    lambda2 = random.uniform(0.3, 10)
    lambda3 = random.uniform(0.3, 10)
    diag = [lambda1, lambda2, lambda3]
    cov_mat = np.dot(np.dot(orth_mat, np.diag(diag)), orth_mat.T)
    if num == 0 or num == 1:
        resize_brain = zoom(brainmask, (0.25, 0.25, 0.25), order=0)
    elif num == 2:
        resize_brain = zoom(brainmask, (0.5, 0.5, 0.5), order=0)
    else:
        resize_brain = zoom(brainmask, (0.125, 0.125, 0.125), order=0)
    target_area = np.where(resize_brain > 0)
    rand = np.random.randint(target_area[0].shape)
    mean1, mean2, mean3 = target_area[0][rand][0], target_area[1][rand][0], target_area[2][rand][0]
    rv = multivariate_normal([0, 0, 0], cov_mat)
    mask_small = np.ones(resize_brain.shape)
    point = rv.rvs(100000)
    p = point * 1 + [mean1, mean2, mean3]
    x = np.where(p[:, 0] >= resize_brain.shape[0], resize_brain.shape[0] - 1, p[:, 0]).astype(int)
    y = np.where(p[:, 1] >= resize_brain.shape[1], resize_brain.shape[1] - 1, p[:, 1]).astype(int)
    z = np.where(p[:, 2] >= resize_brain.shape[2], resize_brain.shape[2] - 1, p[:, 2]).astype(int)
    mask_small[x, y, z] = 0
    f = 4 if num in (0, 1) else 2 if num == 2 else 8
    mask = mask_small.repeat(f, axis=0).repeat(f, axis=1).repeat(f, axis=2)
    if mask.shape[0] < shape[0] or mask.shape[1] < shape[1] or mask.shape[2] < shape[2]:
        pad = [(abs(shape[i] - mask.shape[i]) // 2, abs(shape[i] - mask.shape[i]) - abs(shape[i] - mask.shape[i]) // 2)
               for i in range(3)]
        mask = np.pad(mask, pad, "constant")
    if mask.shape[0] > shape[0] or mask.shape[1] > shape[1] or mask.shape[2] > shape[2]:
        mask = mask[:shape[0], :shape[1], :shape[2]]
    return mask.astype(np.float32)


class Crops(torch.utils.data.Dataset):
    """One random 192^3 crop of each training scan per epoch, with the official training mask and RandRotate90
    (prob 0.1, spatial axes (0, 2)). Returns (image, brain mask, mask); the high-frequency channel is computed on
    the GPU from the (rotated) crop."""

    def __init__(self, uids):
        self.uids = uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        u = self.uids[i]
        V = np.load(f"{ISO}/{u}.vol.npy", mmap_mode="r")
        D = np.load(f"{ISO}/{u}.dom.npy", mmap_mode="r")
        C = np.load(f"{ISO}/{u}.centres_all.npy", mmap_mode="r")
        c = np.asarray(C[np.random.randint(len(C))], int)
        lo = [int(np.clip(c[k] - CROP // 2, 0, max(0, V.shape[k] - CROP))) for k in range(3)]
        sl = tuple(slice(lo[k], lo[k] + CROP) for k in range(3))
        img = norm(V[sl])
        bm = (np.asarray(D[sl]) > 0).astype(np.float32)
        if img.shape != (CROP,) * 3:                       # never for the LUNA16 volumes (all >= 228 mm per axis)
            pad = [(0, CROP - s) for s in img.shape]
            img, bm = np.pad(img, pad, constant_values=-3.0), np.pad(bm, pad)
        m = official_mask(bm, img.shape)
        st = np.stack([img, bm, m])
        if random.random() < 0.1:                          # monai RandRotate90(prob=0.1, spatial_axes=(0, 2))
            st = np.rot90(st, k=random.randint(1, 3), axes=(1, 3))   # monai: k = randint(max_k=3) + 1
        return torch.from_numpy(np.ascontiguousarray(st))


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 31
    np.random.seed(s)
    random.seed(s + 1)


def split():
    return json.load(open(f"{ROOT}/runs_wm/ctx_free/split.json"))


def train(a):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    uids = split()["train"]
    runs = f"{RUNS}{a.tag}"
    os.makedirs(runs, exist_ok=True)
    model = UNET3D(in_channels=2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    start = 0
    ck = f"{runs}/last.pt"
    if os.path.exists(ck) and not a.bench:
        st = torch.load(ck, map_location="cuda")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start = st["epoch"] + 1
        print(f"resumed after epoch {start}", flush=True)
    dl = torch.utils.data.DataLoader(Crops(uids), batch_size=2, shuffle=True, drop_last=False, num_workers=a.workers,
                                     worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=4,
                                     pin_memory=True)
    print(f"{len(uids)} training scans, {len(dl)} steps per epoch, epochs {a.epochs}, lr drop at {a.drop}, "
          f"bf16 autocast {a.amp}", flush=True)
    for epoch in range(start, a.epochs):
        if epoch >= a.drop:
            for g in opt.param_groups:
                g["lr"] = 1e-4
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in dl:
            b = b.cuda(non_blocking=True)
            gt, bm, md = b[:, 0:1], b[:, 1:2], b[:, 2:3]
            hf = highfreq(gt)
            mask = (1 - md) * bm
            inp = torch.cat(((1 - mask) * gt + mask * torch.randn_like(gt), hf), 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                out = model(inp.float())
            loss = ((gt - out.float()) ** 2).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n += 1
            if a.bench and n % 10 == 0:
                print(f"{n} steps {(time.time() - t0) / n:.3f} s/step loss {tot / n:.4f} "
                      f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
            if a.bench and n >= 60:
                return
        print(f"epoch {epoch + 1}/{a.epochs} loss {tot / n:.5f} ({time.time() - t0:.0f}s, lr {opt.param_groups[0]['lr']:g})",
              flush=True)
        st = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch}
        torch.save(st, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        if (epoch + 1) % 20 == 0:                           # official: save every 20 epochs
            torch.save(model.state_dict(), f"{runs}/model_epoch{epoch + 1}.pth")
    torch.save(model.state_dict(), f"{runs}/model_final.pth")
    print("done", flush=True)


def tangent_model_shifted(t, a, b):
    return a * np.tan(b * t)


def tangent_model_derivative(t, a, b):
    return a * b * (1 / np.cos(b * t)) ** 2


def stop_threshold(thres_all, gamma, y_limit):
    """test.py at i == 100: fit a * tan(b t) to the threshold curve (points below the y limit), take the first
    iteration where its derivative exceeds gamma; if the fit is poor (R^2 < 0.85, or it fails) use the smoothed
    finite differences instead. Batch size 1 (the official default)."""
    y = np.asarray(thres_all, np.float64)
    x_data = np.arange(1, 101, dtype=np.float64)
    idx = np.nonzero(y > y_limit)[0]
    index = int(idx[0]) if len(idx) else 100
    try:
        (pa, pb), _ = curve_fit(tangent_model_shifted, np.arange(1, index + 1, dtype=np.float64), y[:index],
                                p0=[1.0, 0.01], maxfev=500000)
        df = tangent_model_derivative(x_data, pa, pb)
        y_pred = tangent_model_shifted(np.arange(1, 81, dtype=np.float64), pa, pb)
        r2 = float(1 - np.sum((y[:80] - y_pred) ** 2) / np.sum((y[:80] - np.mean(y[:80])) ** 2))
        fallback = r2 < 0.85                                # as in test.py (a NaN R^2 keeps the fit)
    except Exception:                                       # the official script would stop here; fall back instead
        r2, fallback = float("nan"), True
    if fallback:
        d = np.concatenate([[y[0]], y[1:] - y[:-1]])
        df = np.array([d[max(0, i - 2):min(100, i + 3)].mean() for i in range(100)])
    stop_k = int(np.argmax(df > gamma))
    return float(y[stop_k]), stop_k, float(r2)


class Volume:
    """A whole 1 mm volume, normalised, padded at the far end of each axis to a multiple of 16 (the network's
    stride), with its brain mask (grown lung domain) and high-frequency channel."""

    def __init__(self, uid):
        V, D, self.meta = load_iso(uid)
        self.shape = V.shape
        P = [(-s) % 16 for s in V.shape]
        x = np.pad(norm(V), [(0, p) for p in P], constant_values=-3.0)
        bm = np.pad((np.asarray(D) > 0).astype(np.float32), [(0, p) for p in P])
        self.D = np.asarray(D)
        self.gt = torch.from_numpy(x)[None, None].cuda()
        self.bm = torch.from_numpy(bm)[None, None].cuda()
        self.hf = highfreq(self.gt)

    def crop(self, t):
        z, y, x = self.shape
        return t[0, 0, :z, :y, :x]


@torch.no_grad()
def predict(model, v, mask, amp):
    masked = (1 - mask) * v.gt + mask * torch.randn_like(v.gt)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        return model(torch.cat((masked, v.hf), 1).float()).float()


@torch.no_grad()
def find_thresholds(model, v, amp):
    """test.py, i = 0..100 of the iterative process: the whole brain masked first, then the mask set to the top
    (1 - 0.01 i) of the brain by the last error, 100 times; returns the 100 thresholds."""
    total, fg = v.bm.numel(), v.bm.sum()
    out = predict(model, v, v.bm, amp)
    err = ((v.gt - out) ** 2) * v.bm
    thres_all = []
    for i in range(1, 101):
        k = int(torch.clamp(total - fg * (1 - 0.01 * i), min=1).item())
        t = torch.kthvalue(err.flatten(), k).values.item()
        thres_all.append(t)
        if i < 100:
            mask = torch.where(err > t, 1.0, 0.0)
            out = predict(model, v, mask, amp)
            err = ((v.gt - out) ** 2) * v.bm
    return thres_all


@torch.no_grad()
def shrink(model, v, thres_final, amp, max_iter=300):
    """test.py after the threshold is fixed: restart from the whole brain mask ('brain_mask', the README's choice for
    3D), mask = error > threshold, until the mask shrinks by less than 1%; the anomaly map is the squared error of
    the last reconstruction inside the brain mask."""
    mask = v.bm
    out = predict(model, v, mask, amp)
    n = 1
    while True:
        err = ((v.gt - out) ** 2) * v.bm
        new = torch.where(err > thres_final, 1.0, 0.0)
        ratio = ((mask.sum() - new.sum()) / mask.sum()).item()
        if np.isnan(ratio):
            ratio = -1
        if ratio < 0.01 or n >= max_iter:
            return err, n, float(mask.sum().item() / v.bm.sum().item())
        mask = new
        out = predict(model, v, mask, amp)
        n += 1


def peaks(W, dom, k=50, nms_mm=10.0):
    """ct_patchcore.peaks, verbatim."""
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


def map_peaks(E, v):
    """ct_patchcore.anomaly_peaks read-out: 2 mm Gaussian smoothing, peaks inside the grown lung domain, 10 mm
    suppression, original voxel coordinates."""
    W = gaussian_filter(np.asarray(E, np.float32), 2.0)
    return [(*to_original(p[:3], v.meta), p[3]) for p in peaks(W, v.D, nms_mm=10.0)]


def infer(a):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    model = UNET3D(in_channels=2).cuda()
    model.load_state_dict(torch.load(a.model, map_location="cuda"))
    model.eval()
    uids = open(a.uids).read().split()
    if a.max_uids:
        uids = uids[:a.max_uids]
    if a.shard:
        i, n = a.shard
        uids = uids[i::n]
    out = f"{RUNS}/{a.out}"
    os.makedirs(out, exist_ok=True)
    os.makedirs(MAPS, exist_ok=True)
    for u in uids:
        if all(os.path.exists(f"{out}/{u}.g{g:g}.json") for g in a.gammas):
            continue
        t0 = time.time()
        v = Volume(u)
        ts = find_thresholds(model, v, a.amp)
        for g in a.gammas:
            t, k, r2 = stop_threshold(ts, g, a.y_limit)
            E, n, frac = shrink(model, v, t, a.amp)
            E = v.crop(E).cpu().numpy()
            if a.save_maps:            # several gammas (dev): one map each until choose_gamma.py keeps the chosen one
                np.save(f"{MAPS}/{u}.g{g:g}.npy" if len(a.gammas) > 1 else f"{MAPS}/{u}.npy", E.astype(np.float16))
            pk = map_peaks(E, v)
            json.dump({"uid": u, "gamma": g, "y_limit": a.y_limit, "thres_all": ts, "thres_final": t, "stop_k": k,
                       "r2": r2, "shrink_iters": n, "final_mask_frac": frac,
                       "peaks": [list(map(float, p)) for p in pk]}, open(f"{out}/{u}.g{g:g}.json", "w"))
            print(f"{u[-12:]} gamma {g:g}: stop {k} thres {t:.4f} R2 {r2:.3f} shrink {n} mask {frac:.3f}", flush=True)
        print(f"{u[-12:]} {v.shape} done in {time.time() - t0:.0f}s", flush=True)
        del v
        torch.cuda.empty_cache()


@torch.no_grad()
def bench_infer(a):
    model = UNET3D(in_channels=2).cuda().eval()
    for amp in (False, True):
        x = torch.randn(1, 2, 352, 336, 400, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        for i in range(4):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                model(x)
            torch.cuda.synchronize()
        print(f"inference forward 352x336x400 bf16 {amp}: {time.time() - t0:.2f}s, peak "
              f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "bench", "infer"])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--drop", type=int, default=300, help="epoch at which lr drops from 1e-3 to 1e-4 (official 300)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (official: float32 with a GradScaler)")
    ap.add_argument("--model", default=f"{RUNS}/model_final.pth")
    ap.add_argument("--uids")
    ap.add_argument("--shard", type=int, nargs=2)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.01])
    ap.add_argument("--y-limit", type=float, default=10.0, help="official fit_funtion_y_limit default")
    ap.add_argument("--out", default="infer")
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--tag", default="", help="run-folder suffix (smoke tests)")
    ap.add_argument("--max-uids", type=int, default=0)
    a = ap.parse_args()
    a.bench = a.cmd == "bench"
    if a.bench:
        bench_infer(a)
        for amp in (False, True):
            a.amp = amp
            train(a)
    elif a.cmd == "train":
        train(a)
    else:
        infer(a)


if __name__ == "__main__":
    main()
