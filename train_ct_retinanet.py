"""Supervised 3-D nodule detector for the lung label-efficiency comparison: MONAI's RetinaNet (ResNet-50 FPN, the
configuration of MONAI's LUNA16 detection tutorial: ATSS matching, hard-negative sampling, anchor shapes, SGD with
Nesterov momentum at 1e-2, NMS 0.22), trained from scratch on the LUNA16 boxes of the same training scans as the reader
(results/lung_train_uids_<b>.txt) and read on the same 1 mm grid as the world model (cache/iso, the lung domain plus a
36 mm margin). The tutorial's anchor shapes and patch size, given in voxels of 0.70 x 0.70 x 1.25 mm, are converted to
1 mm. Every annotated nodule is a training box (LUNA16 annotations, >= 3 mm).

The reported location is the centre of the highest-scoring box, or of the highest-scoring box at least 6 mm across
(the detector's own size estimate); the checkpoint and that rule are chosen on the reader's 62 validation scans by the
same hit rate (perr/score.py: within radius + 3 mm of a nodule >= 6 mm). The evaluation sets are scored once, at the end.

    CUDA_VISIBLE_DEVICES=0 venvs/clipvenv/bin/python train_ct_retinanet.py --train-uids results/lung_train_uids_10.txt --out runs_det/ct_retina_b10
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from perr.context_wm import ORGANS, to_original
from perr.score import MIN_DIAM, TOL_MM
from perr.study import Nodule, pilot_cases

ISO = ORGANS["lung"]["iso"]
SRC = ORGANS["lung"]["src"]
PATCH = (96, 128, 128)                               # z, y, x at 1 mm (the tutorial's 192 x 192 x 80 at 0.70 / 1.25 mm)
ANCHORS = [[5, 6, 4], [6, 4, 6], [8, 7, 7]]          # the tutorial's [6,8,4], [8,6,5], [10,10,6] (x, y, z voxels) in z, y, x mm
ROI = (160, 224, 224)
HU = (-1024.0, 300.0)


class Scan:
    def __init__(self, uid):
        self.uid = uid
        self.V = np.load(f"{ISO}/{uid}.vol.npy", mmap_mode="r")
        self.meta = np.load(f"{ISO}/{uid}.meta.npy", allow_pickle=True).item()
        src = np.load(f"{SRC}/{uid}.meta.npy", allow_pickle=True).item()
        self.spacing = np.array(src["spacing"], float)
        self.nodules = [Nodule(tuple(z), d) for z, d in src["nodules"]]
        c = [np.array(z, float) * self.spacing - self.meta["lo_mm"] for z, _ in src["nodules"]]
        d = [dd for _, dd in src["nodules"]]
        self.boxes = np.array([[*(ci - di / 2), *(ci + di / 2)] for ci, di in zip(c, d)], np.float32).reshape(-1, 6)
        self.centres = np.load(f"{ISO}/{uid}.centres.npy", mmap_mode="r")      # places in the grown lung domain


def norm(a):
    return np.clip((np.asarray(a, np.float32) - HU[0]) / (HU[1] - HU[0]), 0, 1)


def crop(V, lo):
    out = np.full(PATCH, HU[0], np.float32)
    src = [slice(max(0, l), min(s, l + p)) for l, s, p in zip(lo, V.shape, PATCH)]
    dst = [slice(a.start - l, a.stop - l) for a, l in zip(src, lo)]
    out[tuple(dst)] = V[tuple(src)]
    return out


class Patches(IterableDataset):
    """Patches of 96 x 128 x 128 mm, as the tutorial's positive / negative crops (pos = neg = 1): half centred near a
    random annotated nodule of a random scan with nodules, half on a random place of the lung domain of a random scan;
    random flips on every axis, a random quarter turn in the axial plane, intensity scale and shift."""

    def __init__(self, uids, seed):
        self.uids, self.seed = uids, seed
        self.pos_uids = [u for u in uids if len(Scan(u).boxes)]

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        scans = {}
        while True:
            positive = bool(self.pos_uids) and rng.random() < 0.5
            pool = self.pos_uids if positive else self.uids
            u = pool[rng.integers(len(pool))]
            if u not in scans:
                if len(scans) > 64:
                    scans.pop(next(iter(scans)))
                scans[u] = Scan(u)
            s = scans[u]
            half = np.array(PATCH) / 2
            if positive:
                b = s.boxes[rng.integers(len(s.boxes))]
                c = (b[:3] + b[3:]) / 2
                slack = np.maximum(half - (b[3:] - b[:3]) / 2 - 4, 0)
                c = c + rng.uniform(-slack, slack)
            else:
                c = np.array(s.centres[rng.integers(len(s.centres))], float)
            lo = np.round(c - half).astype(int)
            img = norm(crop(s.V, lo))
            bx = s.boxes - np.concatenate([lo, lo]).astype(np.float32)
            ctr = (bx[:, :3] + bx[:, 3:]) / 2
            keep = np.all((ctr >= 0) & (ctr < np.array(PATCH)), 1)
            bx = np.clip(bx[keep], 0, np.concatenate([PATCH, PATCH]).astype(np.float32))
            for ax in range(3):
                if rng.random() < 0.5:
                    img = img[(slice(None),) * ax + (slice(None, None, -1),)]
                    n = PATCH[ax]
                    bx[:, [ax, ax + 3]] = n - bx[:, [ax + 3, ax]]
            k = int(rng.integers(4))
            for _ in range(k):                        # quarter turn in the (y, x) plane: (y, x) -> (X - 1 - x, y)
                img = np.rot90(img, 1, axes=(1, 2))
                y0, x0, y1, x1 = bx[:, 1].copy(), bx[:, 2].copy(), bx[:, 4].copy(), bx[:, 5].copy()
                X = img.shape[1]                      # PATCH is square in y, x
                bx[:, 1], bx[:, 4] = X - x1, X - x0
                bx[:, 2], bx[:, 5] = y0, y1
            img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
            yield (torch.from_numpy(np.ascontiguousarray(img))[None], torch.from_numpy(bx.astype(np.float32)),
                   torch.zeros(len(bx), dtype=torch.long))


def make_detector(device, pretrained=False):
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
    from monai.networks.nets import resnet
    returned = [1, 2]
    anchor = AnchorGeneratorWithAnchorShape(feature_map_scales=[2 ** l for l in range(len(returned) + 1)],
                                            base_anchor_shapes=ANCHORS)
    stride = [2, 2, 2]                               # isotropic grid (the tutorial's [2, 2, 1] is for 1.25 mm slices)
    backbone = resnet.ResNet(block=resnet.ResNetBottleneck, layers=[3, 4, 6, 3], block_inplanes=resnet.get_inplanes(),
                             n_input_channels=1, conv1_t_stride=stride, conv1_t_size=[7, 7, 7],
                             bias_downsample=not pretrained)
    if pretrained:                                   # MedicalNet ResNet-50 (Med3D, 23 datasets): same conv1 and blocks
        sd = resnet.get_pretrained_resnet_medicalnet(50, device="cpu", datasets23=True)
        missing, unexpected = backbone.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        print(f"MedicalNet weights loaded; missing {missing}, unexpected {unexpected}", flush=True)
    fe = resnet_fpn_feature_extractor(backbone=backbone, spatial_dims=3, pretrained_backbone=False,
                                      trainable_backbone_layers=None, returned_layers=returned)
    size_div = [s * 2 * 2 ** max(returned) for s in fe.body.conv1.stride]
    net = RetinaNet(spatial_dims=3, num_classes=1, num_anchors=anchor.num_anchors_per_location()[0],
                    feature_extractor=fe, size_divisible=size_div)
    det = RetinaNetDetector(network=net, anchor_generator=anchor, debug=False).to(device)
    det.set_atss_matcher(num_candidates=4, center_in_gt=False)
    det.set_hard_negative_sampler(batch_size_per_image=64, positive_fraction=0.3, pool_size=20, min_neg=16)
    det.set_target_keys(box_key="box", label_key="label")
    det.set_box_selector_parameters(score_thresh=0.001, topk_candidates_per_level=1000, nms_thresh=0.22,
                                    detections_per_img=100)
    det.set_sliding_window_inferer(roi_size=list(ROI), overlap=0.25, sw_batch_size=1, mode="constant", device=device)
    return det


@torch.no_grad()
def detect(det, s, device):
    """Boxes of one scan: [(score, centre zyx in original voxels, size mm)], best first."""
    det.eval()
    x = torch.from_numpy(norm(s.V))[None].to(device)
    out = det([x], use_inferer=True)[0]
    b, sc = out["box"].float().cpu().numpy(), out["label_scores"].float().cpu().numpy()
    det.train()
    rows = []
    for i in np.argsort(-sc):
        c = (b[i, :3] + b[i, 3:]) / 2
        rows.append({"score": float(sc[i]), "zyx": [float(v) for v in to_original(c, s.meta)],
                     "size_mm": float(np.mean(b[i, 3:] - b[i, :3]))})
    return rows


def hit6(s, zyx):
    """perr/score.py's hit: within radius + 3 mm of an annotated nodule >= 6 mm."""
    p = np.array(zyx, float) * s.spacing
    return any(n.diameter_mm >= MIN_DIAM and np.linalg.norm(p - np.array(n.zyx, float) * s.spacing) <= n.diameter_mm / 2 + TOL_MM
               for n in s.nodules)


def hits(scans, dets):
    """Hit rate (>= 6 mm) of the two reporting rules."""
    out = {}
    for rule in ("top", "top6"):
        h = []
        for s in scans:
            cand = [r for r in dets[s.uid] if rule == "top" or r["size_mm"] >= 6.0] or dets[s.uid][:1]
            h.append(bool(cand) and hit6(s, cand[0]["zyx"]))
        out[rule] = float(np.mean(h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-uids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--val-every", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=4, help="patches per step (the tutorial's 4)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=5000, help="linear warm-up steps (the tutorial warms up for 10 epochs)")
    ap.add_argument("--clip", type=float, default=10.0, help="gradient-norm clipping, a guard against early collapse")
    ap.add_argument("--pretrained", action="store_true", help="MedicalNet ResNet-50 backbone weights")
    ap.add_argument("--smoke", type=int, default=0, help="test run: this many validation scans, no evaluation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    train = open(a.train_uids).read().split()
    val = sorted({json.loads(l)["uid"] for l in open("data/reader_ctx/val.jsonl")})
    assert not set(train) & set(val)
    print(f"{len(train)} training scans ({sum(len(Scan(u).boxes) for u in train)} boxes), {len(val)} validation scans",
          flush=True)
    det = make_detector(dev, a.pretrained)
    opt = torch.optim.SGD(det.network.parameters(), lr=a.lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda g: min(1.0, (g + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(g, a.iters) / a.iters)))
    dl = DataLoader(Patches(train, 0), batch_size=a.bs, num_workers=8, collate_fn=lambda b: b, persistent_workers=True)
    vscans = [Scan(u) for u in (val[:a.smoke] if a.smoke else val)]
    best, t0, run, log = (-1.0, None, 0), time.time(), [], []
    rc, rr = [], []
    for g, batch in enumerate(dl):
        imgs = [b[0].to(dev, non_blocking=True) for b in batch]
        tg = [{"box": b[1].to(dev), "label": b[2].to(dev)} for b in batch]
        out = det(imgs, tg)                          # fp32 with TF32 matmuls: faster than bf16 autocast here
        loss = out[det.cls_key] + out[det.box_reg_key]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(det.network.parameters(), a.clip)
        opt.step()
        sched.step()
        run.append(loss.item())
        rc.append(out[det.cls_key].item())
        rr.append(out[det.box_reg_key].item())
        if (g + 1) % 250 == 0:
            print(f"iter {g + 1} cls {np.mean(rc):.4f} reg {np.mean(rr):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            rc, rr = [], []
        if (g + 1) % a.val_every == 0 or g + 1 in (2000, a.iters):
            dets = {s.uid: detect(det, s, dev) for s in vscans}
            h = hits(vscans, dets)
            log.append({"iter": g + 1, "loss": float(np.mean(run)), **h})
            print(f"iter {g + 1} loss {np.mean(run):.4f} val hit top {h['top']:.3f} top6 {h['top6']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            run = []
            for rule in ("top", "top6"):
                if h[rule] > best[0]:
                    best = (h[rule], rule, g + 1)
                    torch.save(det.network.state_dict(), f"{a.out}/model.pt")
            json.dump({"log": log, "best": best}, open(f"{a.out}/log.json", "w"), indent=1)
        if g + 1 == a.iters:
            break
    print(f"best validation hit {best[0]:.3f} (rule {best[1]}, iter {best[2]})", flush=True)
    if a.smoke:
        return
    det.network.load_state_dict(torch.load(f"{a.out}/model.pt", map_location=dev))
    for lo, hi in ((60, 160), (160, 260)):
        scans = [Scan(u) for u in pilot_cases(hi)[lo:hi]]
        dets = {s.uid: detect(det, s, dev) for s in scans}
        h = hits(scans, dets)
        json.dump({u: d[:20] for u, d in dets.items()}, open(f"{a.out}/eval_{lo}_{hi}.json", "w"))
        print(f"eval {lo}-{hi}: hit (rule chosen on validation, {best[1]}) {h[best[1]]:.3f}; top {h['top']:.3f} "
              f"top6 {h['top6']:.3f}", flush=True)


if __name__ == "__main__":
    main()
