"""Learned anatomy world model: predict the lung from its surroundings, and read the prediction error.

The model sees a 64 mm cube of CT with its central 24 mm hidden and predicts the hidden part. It is trained on
normal anatomy only: cubes whose hidden part touches an annotated nodule are never drawn. Vessels, airways,
fissures and the pleura continue into the hidden part from the sides and are predicted; a compact structure
that nothing outside announces is not. The loss is the mean squared error, so the prediction is a conditional
mean and an uncertain vessel is predicted as a blur of the right mass rather than as air.

Dense prediction: the study is resampled to 1 mm, tiled into 8 mm blocks, and each block is predicted from a
cube centred on it, so every voxel is predicted from tissue at least 8 mm away. The expected volume E is the
stitched prediction, the error R = observed - E, and candidate addresses are peaks of R smoothed at sigma mm.

No torch import at module level: the cache builder runs in the harness environment.
"""

from __future__ import annotations

import os
import zlib

import numpy as np
from scipy import ndimage as ndi

LUNA_CACHE = "/raid/home/CAMCA/hj880/perr/cache/luna"
ISO = "/raid/home/CAMCA/hj880/perr/cache/iso"
CTX, HOLE, BLOCK = 64, 24, 8          # mm (= voxels at 1 mm)
MARGIN = 36                           # mm of tissue kept around the organ domain
RIM = 5                               # mm: the domain is grown by this much so that juxtapleural nodules are inside
# Per organ: source cache, domain mask, 1 mm cache, intensity window, and the sign of a finding (nodules are denser
# than expected, liver tumours darker).
ORGANS = {"lung": {"src": LUNA_CACHE, "dom": "domain", "iso": ISO, "window": (-1000.0, 400.0), "sign": 1.0},
          "liver": {"src": "/raid/home/CAMCA/hj880/perr/cache/liver", "dom": "liver",
                    "iso": "/raid/home/CAMCA/hj880/perr/cache/iso_liver", "window": (-200.0, 300.0), "sign": -1.0},
          # extra liver training data (TCIA CRLM), cached next to the MSD cases so one model trains on both
          "liver_crlm": {"src": "/raid/home/CAMCA/hj880/perr/cache/liver_crlm", "dom": "liver",
                         "iso": "/raid/home/CAMCA/hj880/perr/cache/iso_liver", "window": (-200.0, 300.0), "sign": -1.0},
          # extra liver training data (MSD Task08 Hepatic Vessel, liver from TotalSegmentator), cached likewise
          "liver_t08": {"src": "/raid/home/CAMCA/hj880/perr/cache/liver_t08", "dom": "liver",
                        "iso": "/raid/home/CAMCA/hj880/perr/cache/iso_liver", "window": (-200.0, 300.0), "sign": -1.0},
          # external liver test set (HCC-TACE-Seg): read with the liver model, never trained on
          "liver_ext": {"src": "/raid/home/CAMCA/hj880/perr/cache/liver_ext", "dom": "liver",
                        "iso": "/raid/home/CAMCA/hj880/perr/cache/iso_liver_ext", "window": (-200.0, 300.0), "sign": -1.0}}
HU_LO, HU_HI = ORGANS["lung"]["window"]


def norm(hu, organ: str = "lung"):
    lo, hi = ORGANS[organ]["window"]
    return np.clip((np.asarray(hu, np.float32) - lo) / (hi - lo), 0, 1)


def build_iso(uid: str, organ: str = "lung") -> None:
    """Resample a cached study to 1 mm around its organ domain; also the grown domain and the lesion exclusion mask
    (lung: every annotated nodule, radius + 6 mm; liver: every tumour voxel grown by 6 mm), used only to keep
    lesions out of training."""
    cfg = ORGANS[organ]
    stem = f"{cfg['iso']}/{uid}"
    if os.path.exists(stem + ".meta.npy"):
        return
    os.makedirs(cfg["iso"], exist_ok=True)
    vol = np.load(f"{cfg['src']}/{uid}.vol.npy", mmap_mode="r")
    dom = np.load(f"{cfg['src']}/{uid}.{cfg['dom']}.npy", mmap_mode="r")
    meta = np.load(f"{cfg['src']}/{uid}.meta.npy", allow_pickle=True).item()
    sp = np.array(meta["spacing"], float)
    idx = [np.nonzero(np.asarray(dom).any(axis=ax))[0] for ax in ((1, 2), (0, 2), (0, 1))]
    lo = np.maximum(np.array([a.min() for a in idx]) * sp - MARGIN, 0)
    hi = np.minimum(np.array([a.max() for a in idx]) * sp + MARGIN, (np.array(vol.shape) - 1) * sp)
    shape = tuple(int(v) for v in np.floor(hi - lo) + 1)
    V = ndi.affine_transform(np.asarray(vol, np.float32), 1 / sp, offset=lo / sp, output_shape=shape, order=1, mode="nearest")
    D = ndi.affine_transform(np.asarray(dom, np.float32), 1 / sp, offset=lo / sp, output_shape=shape, order=1) > 0.5
    Dg = ndi.distance_transform_edt(~D) <= RIM
    ex = np.zeros(shape, bool)
    if organ.startswith("liver"):       # tumour masks grown by 3 mm are cached; grow them by 3 mm more
        tum = np.maximum(np.load(f"{cfg['src']}/{uid}.big.npy"), np.load(f"{cfg['src']}/{uid}.small.npy")).astype(np.float32)
        tum = ndi.affine_transform(tum, 1 / sp, offset=lo / sp, output_shape=shape, order=1) > 0.5
        ex = ndi.distance_transform_edt(~tum) <= 3 if tum.any() else ex
    for zyx, d in (meta["nodules"] if organ == "lung" else []):          # lung: spheres around annotated nodules
        c = np.array(zyx, float) * sp - lo
        r = d / 2 + 6
        sl = tuple(slice(max(0, int(c[i] - r)), max(0, min(shape[i], int(c[i] + r) + 2))) for i in range(3))
        g = np.ogrid[sl]
        ex[sl] |= sum((g[i] - c[i]) ** 2 for i in range(3)) <= r * r
    np.save(stem + ".vol.npy", np.round(V).astype(np.int16))
    np.save(stem + ".dom.npy", (D.astype(np.uint8) + Dg.astype(np.uint8)))      # 2 = domain, 1 = grown rim
    np.save(stem + ".excl.npy", ex.astype(np.uint8))
    # training centres: inside the grown domain, the hidden cube clear of nodules, the context inside the volume
    h = CTX // 2
    clear = ndi.maximum_filter(ex, size=HOLE + 1) == 0
    inner = np.zeros(shape, bool)
    inner[h:shape[0] - h, h:shape[1] - h, h:shape[2] - h] = True
    ok = Dg & clear & inner
    rng = np.random.default_rng(zlib.crc32(uid.encode()))
    allc = np.argwhere(ok)
    tis = np.argwhere(ok & (V > (-500 if organ == "lung" else 0)))
    pick = lambda a, n: a[rng.choice(len(a), size=min(n, len(a)), replace=False)] if len(a) else a
    np.save(stem + ".centres.npy", np.concatenate([pick(allc, 20000), pick(tis, 20000)]).astype(np.int16))
    np.save(stem + ".meta.npy", {"lo_mm": lo, "spacing": sp, "shape": shape, "n_all": len(allc), "n_tissue": len(tis)})


def all_centres(uid: str, organ: str = "lung"):
    """Training centres without the lesion exclusion, for a world model trained without any lesion annotation:
    the same draw as build_iso (inside the grown domain, the context inside the volume, half of them on tissue) with
    annotated lesions left in. Cached as <uid>.centres_all.npy."""
    stem = f"{ORGANS[organ]['iso']}/{uid}"
    if not os.path.exists(stem + ".centres_all.npy"):
        V = np.load(stem + ".vol.npy", mmap_mode="r")
        Dg = np.load(stem + ".dom.npy") > 0
        h, shape = CTX // 2, Dg.shape
        inner = np.zeros(shape, bool)
        inner[h:shape[0] - h, h:shape[1] - h, h:shape[2] - h] = True
        ok = Dg & inner
        rng = np.random.default_rng(zlib.crc32(uid.encode()))
        allc = np.argwhere(ok)
        tis = np.argwhere(ok & (np.asarray(V) > (-500 if organ == "lung" else 0)))
        pick = lambda a, n: a[rng.choice(len(a), size=min(n, len(a)), replace=False)] if len(a) else a  # noqa: E731
        np.save(stem + ".centres_all.npy", np.concatenate([pick(allc, 20000), pick(tis, 20000)]).astype(np.int16))
    return np.load(stem + ".centres_all.npy")


def load_iso(uid: str, organ: str = "lung"):
    stem = f"{ORGANS[organ]['iso']}/{uid}"
    meta = np.load(stem + ".meta.npy", allow_pickle=True).item()
    return np.load(stem + ".vol.npy", mmap_mode="r"), np.load(stem + ".dom.npy", mmap_mode="r"), meta


def to_original(zyx_iso, meta):
    """1 mm grid index -> continuous voxel index of the original study."""
    return (np.asarray(zyx_iso, float) + meta["lo_mm"]) / meta["spacing"]


def make_unet(ch=(32, 64, 128, 256, 320), out_ch=1, in_ch=2):
    """out_ch 1: predicted CT; out_ch 2: predicted CT and its log-variance (the model's own uncertainty).
    in_ch 2: (hidden-zeroed CT, hidden mask); 3 adds a co-registered prior CT (the disease-course world)."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                                     nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())

        def forward(self, x):
            return self.net(x)

    class UNet3D(nn.Module):
        """Input: (hidden-zeroed CT, hidden mask); output: the predicted CT on the whole cube."""

        def __init__(self):
            super().__init__()
            self.enc, cin = nn.ModuleList(), in_ch
            for c in ch:
                self.enc.append(Block(cin, c))
                cin = c
            self.up, self.dec = nn.ModuleList(), nn.ModuleList()
            for c in reversed(ch[:-1]):
                self.up.append(nn.ConvTranspose3d(cin, c, 2, stride=2))
                self.dec.append(Block(2 * c, c))
                cin = c
            self.out = nn.Conv3d(cin, out_ch, 1)

        def forward(self, x):
            skips = []
            for i, b in enumerate(self.enc):
                x = b(x)
                if i < len(self.enc) - 1:
                    skips.append(x)
                    x = F.max_pool3d(x, 2)
            for up, dec in zip(self.up, self.dec):
                x = dec(torch.cat([up(x), skips.pop()], 1))
            y = self.out(x)
            return torch.sigmoid(y) if out_ch == 1 else torch.cat([torch.sigmoid(y[:, :1]), y[:, 1:]], 1)

    return UNet3D()


def hole_mask():
    m = np.zeros((CTX, CTX, CTX), np.float32)
    a, b = (CTX - HOLE) // 2, (CTX + HOLE) // 2
    m[a:b, a:b, a:b] = 1
    return m


def error_map(R, S=None, sigma=2.0, signed=True, weight="none"):
    """Smoothed prediction error. Signed smoothing lets the halo of a blurred (uncertain) vessel cancel its core,
    so what remains is mass the context did not predict. With the model's log-variance S the error can be
    weighted by the model's confidence: weight "z" divides by the predicted s.d. before smoothing, "pool" divides
    the smoothed error by the smoothed s.d. (both: an error is only surprising where the model was sure)."""
    R = np.asarray(R, np.float32)
    if S is not None and weight == "z":
        R = R * np.exp(-0.5 * np.asarray(S, np.float32))
    W = ndi.gaussian_filter(R if signed else np.clip(R, 0, None), sigma)
    if S is not None and weight == "pool":
        W = W / np.sqrt(ndi.gaussian_filter(np.exp(np.asarray(S, np.float32)), sigma))
    return W


def peaks_from_error(R, dom, k=50, sigma=2.0, nms_mm=10.0, signed=True, S=None, weight="none"):
    """Top-k peaks of the smoothed error inside the grown domain (1 mm grid)."""
    W = error_map(R, S, sigma, signed, weight)
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


class ContextPrior:
    """Candidate addresses from the learned anatomy world model's cached error (run infer_context_wm.py first).
    Defaults are the final configuration, fixed on the world-model dev split: the model with a variance head
    (tag "nll"), signed smoothing at 2 mm, and the smoothed error divided by the smoothed predicted s.d."""

    name = "context"

    def __init__(self, tag: str = "nll", sigma: float = 2.0, signed: bool = True, weight: str = "pool", organ: str = "lung"):
        self.tag, self.sigma, self.signed, self.weight, self.organ = tag, sigma, signed, weight, organ

    def top(self, study, k: int = 10, nms_mm: float = 10.0):
        cfg = ORGANS[self.organ]
        R = np.load(f"{cfg['iso']}/{study.case_id}.{self.tag}R.npy", mmap_mode="r")
        if cfg["sign"] < 0:
            R = -np.asarray(R, np.float32)
        S = np.load(f"{cfg['iso']}/{study.case_id}.{self.tag}S.npy", mmap_mode="r") if self.weight != "none" else None
        _, D, meta = load_iso(study.case_id, self.organ)
        return [(*to_original(p[:3], meta), p[3]) for p in
                peaks_from_error(R, D, k=k, sigma=self.sigma, nms_mm=nms_mm, signed=self.signed, S=S, weight=self.weight)]


def interleave(a, b, spacing, nms_mm: float = 10.0):
    """Alternate two ranked address lists, dropping an address within nms_mm of one already taken."""
    out, sp = [], np.array(spacing, float)
    for pair in zip(a, b):
        for p in pair:
            if all(np.linalg.norm((np.array(p[:3]) - np.array(q[:3])) * sp) > nms_mm for q in out):
                out.append(p)
    return out


def fuse_rrf(lists, spacing, c: float = 5.0, nms_mm: float = 10.0):
    """Reciprocal-rank fusion of ranked address lists: addresses within nms_mm of each other are one address,
    scored sum over lists of 1 / (c + rank in that list); an address both world models rank high comes first."""
    sp = np.array(spacing, float)
    cands = sorted(((r, li, p) for li, lst in enumerate(lists) for r, p in enumerate(lst)), key=lambda t: t[0])
    groups = []                                        # [address, {list: best rank}]
    for r, li, p in cands:
        for g in groups:
            if np.linalg.norm((np.array(p[:3]) - np.array(g[0][:3])) * sp) <= nms_mm:
                g[1].setdefault(li, r)
                break
        else:
            groups.append([p, {li: r}])
    scored = [(sum(1 / (c + r + 1) for r in g[1].values()), g[0]) for g in groups]
    return [(*p[:3], sc) for sc, p in sorted(scored, key=lambda t: -t[0])]


class MixedPrior:
    """The shape world model and the learned one, rank-interleaved (shape first)."""

    name = "mixed"

    def __init__(self, tag: str = "ctx", sigma: float = 2.0, signed: bool = True, weight: str = "none"):
        self.ctx = ContextPrior(tag, sigma, signed, weight)

    def top(self, study, k: int = 10):
        from .worldmodels import ShapePrior
        return interleave(ShapePrior().top(study, k), self.ctx.top(study, k), study.spacing)[:k]


class FusedPrior:
    """The shape world model and the learned one, fused by reciprocal rank (c = 10, fixed on the dev split)."""

    name = "fused"

    def __init__(self, tag: str = "ctx", sigma: float = 2.0, weight: str = "none", c: float = 10.0, depth: int = 50):
        self.ctx, self.c, self.depth = ContextPrior(tag, sigma, True, weight), c, depth

    def top(self, study, k: int = 10):
        from .worldmodels import ShapePrior
        lists = [ShapePrior().top(study, self.depth), self.ctx.top(study, self.depth)]
        return fuse_rrf(lists, study.spacing, c=self.c)[:k]
