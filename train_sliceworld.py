"""SliceWorld's world-state model (Tian & Song, "SliceWorld: A Predictive and Controllable World-State Model for CT
Report Generation", NeurIPS 2026, arXiv 2605.24371), re-implemented from the paper's appendix because no code or weights
are released, and trained on the same lung training scans and nodule annotations as our reader
(results/lung_train_uids_100.txt). The report-generation stage is not used: the comparison is between world models,
on what SliceWorld's state can localise (a slice).

Model, as in the appendix: frozen DINOv2 ViT-L/14 slice tokens (sliceworld_feats.py) pooled by a learnable query
(cross-attention, 16 heads), e_t = W_e LN(.) of width 512, a causal 12-layer Mamba (d_state 16, d_conv 4, expand 2) with
learned slice positions, state factors anatomy / lesion / uncertainty (256 / 192 / 64) projected to a 512-d world token,
multi-step future-slice feature prediction K = 5 from h_t and from the world token, the factor-aware losses (anatomy
smoothness, lesion sparsity, uncertainty regressing the stop-gradient prediction error, lesion occupancy BCE) and the
counterfactual losses (lesion factor zeroed: invariance on lesion-free slices, hinge with margin 0.1 on lesion slices),
every weight 1; AdamW 1e-4, weight decay 0.01, batch 8, dropout 0.1. Implementation choices the paper leaves open:
the prediction targets are the future slices' DINOv2 CLS features (z-scored on the training scans; a fixed target cannot
collapse), squared errors and norms are averaged over dimensions, the lesion occupancy label of a slice is 1 inside an
annotated nodule's z extent, and the epoch is chosen on the reader's 62 validation scans by occupancy AUPRC (the paper
trains 3 epochs on the 10x larger DeepLesion).

Per evaluation scan it writes, for every sampled slice, the lesion-occupancy probability, the one-step prediction error
of that slice given the slices before it, and that error over the predicted uncertainty.

    python train_sliceworld.py --out runs_sw/lung
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from perr.context_wm import ORGANS
from perr.study import pilot_cases

FE = "/raid/home/CAMCA/hj880/perr/cache/sliceworld"
ISO, SRC = ORGANS["lung"]["iso"], ORGANS["lung"]["src"]
K, D_VIS, D = 5, 1024, 512


def nodules_iso(u):
    """[(centre zyx on the 1 mm grid, diameter mm)] of a scan's annotated nodules."""
    meta = np.load(f"{ISO}/{u}.meta.npy", allow_pickle=True).item()
    src = np.load(f"{SRC}/{u}.meta.npy", allow_pickle=True).item()
    sp = np.array(src["spacing"], float)
    return [(np.array(z, float) * sp - meta["lo_mm"], float(d)) for z, d in src["nodules"]]


def load(u):
    f = torch.from_numpy(np.load(f"{FE}/{u}.npy"))
    z = np.load(f"{FE}/{u}.z.npy")
    m = np.zeros(len(z), np.float32)
    for c, d in nodules_iso(u):
        m[np.abs(z - c[0]) <= max(d / 2, 1.0)] = 1
    return f, z, torch.from_numpy(m)


class SliceWorld(nn.Module):
    def __init__(self, n_layers=12, dropout=0.1, max_len=256):
        super().__init__()
        from transformers import MambaConfig, MambaModel
        self.q = nn.Parameter(torch.randn(1, 1, D_VIS) * 0.02)
        self.pool = nn.MultiheadAttention(D_VIS, 16, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(D_VIS)
        self.we = nn.Linear(D_VIS, D)
        self.pos = nn.Embedding(max_len, D)
        self.drop = nn.Dropout(dropout)
        self.seq = MambaModel(MambaConfig(vocab_size=8, hidden_size=D, num_hidden_layers=n_layers, state_size=16,
                                          conv_kernel=4, expand=2, use_cache=False))
        self.f_anat, self.f_les, self.f_unc = nn.Linear(D, 256), nn.Linear(D, 192), nn.Linear(D, 64)
        self.f_occ = nn.Linear(192, 1)
        self.world = nn.Linear(512, D)
        self.g_h, self.g_w = nn.Linear(D, K * D_VIS), nn.Linear(D, K * D_VIS)
        self.g_unc = nn.Linear(64, 1)

    def forward(self, P):
        """P: [B, T, 257, 1024] DINOv2 tokens (CLS first)."""
        B, T = P.shape[:2]
        x = P[:, :, 1:].reshape(B * T, -1, D_VIS)
        e, _ = self.pool(self.q.expand(B * T, 1, D_VIS), x, x, need_weights=False)
        e = self.we(self.ln(e[:, 0])).view(B, T, D)
        h = self.seq(inputs_embeds=self.drop(e + self.pos(torch.arange(T, device=P.device)))).last_hidden_state
        a, l, u = self.f_anat(h), self.f_les(h), self.f_unc(h)
        w = self.world(torch.cat([a, l, u], -1))
        w_cf = self.world(torch.cat([a, torch.zeros_like(l), u], -1))
        return {"a": a, "l": l, "occ": self.f_occ(l)[..., 0], "unc": self.g_unc(u)[..., 0],
                "ph": self.g_h(self.drop(h)).view(B, T, K, D_VIS), "pw": self.g_w(self.drop(w)).view(B, T, K, D_VIS),
                "pcf": self.g_w(self.drop(w_cf)).view(B, T, K, D_VIS)}


def future(tgt, lens):
    """Targets e_{t+k} [B, T, K, D] and their validity mask [B, T, K]."""
    B, T, _ = tgt.shape
    out = torch.zeros(B, T, K, tgt.shape[-1], device=tgt.device, dtype=tgt.dtype)
    ok = torch.zeros(B, T, K, dtype=torch.bool, device=tgt.device)
    for k in range(1, K + 1):
        out[:, :T - k, k - 1] = tgt[:, k:]
        for b in range(B):
            ok[b, :max(0, lens[b] - k), k - 1] = True
    return out, ok


def losses(o, tgt, m, lens):
    B, T = m.shape
    valid = torch.zeros(B, T, dtype=torch.bool, device=m.device)
    for b in range(B):
        valid[b, :lens[b]] = True
    fut, ok = future(tgt, lens)
    okf = ok.float()
    se_h = ((o["ph"] - fut) ** 2).mean(-1)                       # [B, T, K]
    se_w = ((o["pw"] - fut) ** 2).mean(-1)
    se_cf = ((o["pcf"] - fut) ** 2).mean(-1)
    mfp = (se_h * okf).sum() / okf.sum() + (se_w * okf).sum() / okf.sum()
    smooth = ((o["a"][:, 1:] - o["a"][:, :-1]) ** 2).mean(-1)[valid[:, 1:]].mean()
    sparse = o["l"].abs().mean(-1)[valid].mean()
    has = ok.any(-1)
    err = ((se_h.detach() * okf).sum(-1) / okf.sum(-1).clamp(min=1))
    unc = ((o["unc"] - err) ** 2)[has].mean()
    occ = F.binary_cross_entropy_with_logits(o["occ"][valid], m[valid])
    lf = (se_w * okf).sum(-1) / okf.sum(-1).clamp(min=1)
    lcf = (se_cf * okf).sum(-1) / okf.sum(-1).clamp(min=1)
    neg, pos = has & (m == 0), has & (m == 1)
    inv = (((o["pw"] - o["pcf"]) ** 2).mean(-1) * okf).sum(-1)[neg].sum() / okf.sum(-1)[neg].sum().clamp(min=1)
    eff = F.relu(0.1 + lf - lcf)[pos].mean() if pos.any() else lf.sum() * 0
    parts = {"mfp": mfp, "smooth": smooth, "sparse": sparse, "unc": unc, "occ": occ, "inv": inv, "eff": eff}
    return sum(parts.values()), {k: float(v) for k, v in parts.items()}


def batch(items, mu, sd, dev):
    lens = [len(f) for f, _, _ in items]
    T = max(lens)
    P = torch.zeros(len(items), T, 257, D_VIS, dtype=torch.float16)
    m = torch.zeros(len(items), T)
    for i, (f, _, mm) in enumerate(items):
        P[i, :len(f)] = f
        m[i, :len(f)] = mm
    P = P.to(dev, non_blocking=True)
    tgt = (P[:, :, 0].float() - mu) / sd
    return P, tgt, m.to(dev), lens


def auprc(score, y):
    o = np.argsort(-score)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


@torch.no_grad()
def read(model, uids, mu, sd, dev):
    """Per scan: sampled z, occupancy probability, one-step prediction error and error / predicted uncertainty."""
    model.eval()
    out = {}
    for u in uids:
        f, z, m = load(u)
        P, tgt, _, lens = batch([(f, z, m)], mu, sd, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(P)
        err = torch.full((len(z),), float("nan"))
        err[1:] = ((o["ph"][0, :-1, 0].float() - tgt[0, 1:]) ** 2).mean(-1).cpu()   # slice t given slices < t
        unc = torch.full((len(z),), float("nan"))
        unc[1:] = o["unc"][0, :-1].float().clamp(min=1e-3).cpu()
        out[u] = {"z": z.tolist(), "occ": torch.sigmoid(o["occ"][0].float()).cpu().tolist(), "err": err.tolist(),
                  "surprise": (err / unc).tolist(), "m": m.tolist()}
    model.train()
    return out


def slice_hit(u, z):
    """A slice on the 1 mm grid within radius + 3 mm (along z) of an annotated nodule >= 6 mm."""
    return any(d >= 6 and abs(z - c[0]) <= d / 2 + 3 for c, d in nodules_iso(u))


def top_slice(r, key):
    s = np.nan_to_num(np.array(r[key], float), nan=-np.inf)
    return r["z"][int(np.argmax(s))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = torch.device("cuda")
    train = open("results/lung_train_uids_100.txt").read().split()
    val = sorted({json.loads(l)["uid"] for l in open("data/reader_ctx/val.jsonl")})
    data = {u: load(u) for u in train}
    cls = torch.cat([data[u][0][:, 0].float() for u in train[::4]])
    mu, sd = cls.mean(0).to(dev), cls.std(0).clamp(min=1e-3).to(dev)
    print(f"{len(train)} training scans ({sum(int(data[u][2].sum() > 0) for u in train)} with nodule slices), "
          f"{len(val)} validation scans", flush=True)
    model = SliceWorld().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    best, log, t0 = (-1.0, -1), [], time.time()
    for ep in range(a.epochs):
        order = rng.permutation(len(train))
        run = []
        for i in range(0, len(order) - a.bs + 1, a.bs):
            P, tgt, m, lens = batch([data[train[j]] for j in order[i:i + a.bs]], mu, sd, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(P)
            o = {k: v.float() for k, v in o.items()}
            loss, parts = losses(o, tgt, m, lens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run.append(parts)
        vr = read(model, val, mu, sd, dev)
        y = np.concatenate([vr[u]["m"] for u in val])
        ap_ = auprc(np.concatenate([vr[u]["occ"] for u in val]), y)
        hit = {k: float(np.mean([slice_hit(u, top_slice(vr[u], k)) for u in val])) for k in ("occ", "err", "surprise")}
        mean = {k: float(np.mean([p[k] for p in run])) for k in run[0]}
        log.append({"epoch": ep + 1, "train": mean, "val_auprc": ap_, "val_slice_hit": hit})
        print(f"epoch {ep + 1} " + " ".join(f"{k} {v:.4f}" for k, v in mean.items()) +
              f" | val occupancy AUPRC {ap_:.3f} slice hit occ {hit['occ']:.3f} err {hit['err']:.3f} "
              f"surprise {hit['surprise']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if ap_ > best[0]:
            best = (ap_, ep + 1)
            torch.save(model.state_dict(), f"{a.out}/model.pt")
        json.dump({"log": log, "best": best}, open(f"{a.out}/log.json", "w"), indent=1)
    print(f"best validation occupancy AUPRC {best[0]:.3f} at epoch {best[1]}", flush=True)
    model.load_state_dict(torch.load(f"{a.out}/model.pt", map_location=dev))
    torch.save({"mu": mu.cpu(), "sd": sd.cpu()}, f"{a.out}/norm.pt")
    for lo, hi in ((60, 160), (160, 260)):
        uids = pilot_cases(hi)[lo:hi]
        r = read(model, uids, mu, sd, dev)
        json.dump(r, open(f"{a.out}/eval_{lo}_{hi}.json", "w"))
        hit = {k: float(np.mean([slice_hit(u, top_slice(r[u], k)) for u in uids])) for k in ("occ", "err", "surprise")}
        print(f"eval {lo}-{hi}: slice hit, lesion occupancy {hit['occ']:.3f}, prediction error {hit['err']:.3f}, "
              f"error / uncertainty {hit['surprise']:.3f}", flush=True)


if __name__ == "__main__":
    main()
