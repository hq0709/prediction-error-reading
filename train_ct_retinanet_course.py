"""Supervised 3-D detector of progressing lesions for the longitudinal-CT column of Table 2 (autoPET IV Longitudinal-CT):
the lung detector's recipe (train_ct_retinanet.py: MONAI's RetinaNet, ResNet-50 FPN, the MONAI LUNA16 tutorial
configuration: ATSS matching, hard-negative sampling, SGD with Nesterov momentum at 1e-2, NMS 0.22, from scratch),
reading the follow-up CT with the registered baseline as a second input channel, on PER's longitudinal cache
(cache/iso_course, perr/course_wm.py: the follow-up and the baseline registered onto it, 1.5 mm isotropic around the
follow-up body), i.e. the input PER's longitudinal world model reads.

Training pairs: those of PER's longitudinal verifier (runs_reader/q8_course_vis, data/course_reader/train.jsonl): 245
pairs of 216 training patients. Development pairs: its 27 validation pairs (24 training patients, held out of the
verifier; PER chose its verifier and its selection rule on them). Boxes: the progressing lesions only, i.e. PER's targets
(perr/course_candidates.selected_targets: new, or grown to >= x1.728 baseline volume, and >= 10 mm), each as the tight
bounding box of its follow-up label (fu_lab) moved to the 1.5 mm grid (--boxes builds sota/retinanet_ct/course_boxes.json
from the training labels only). Other lesions (stable, shrinking, smaller or less grown) are background.

Changes from the lung recipe, and why:
  * grid 1.5 mm (PER's longitudinal cache; autoPET slices are mostly 3 mm): patch 64 x 80 x 80 voxels = 96 x 120 x 120 mm
    and sliding window 112 x 144 x 144 voxels = 168 x 216 x 216 mm, the lung script's 96 x 128 x 128 and 160 x 224 x 224 mm
    rounded to the network's size divisor (16 voxels);
  * two input channels, follow-up and registered baseline (--channels 1: follow-up only, the alternative); the same
    spatial and intensity augmentation for both;
  * anchors x3.05: the tutorial's shapes fit LUNA16 nodules (median training box 5.7 mm); progressing-lesion boxes are
    larger (median 17.5 mm, mean side; 90th percentile 45 mm), so the three shapes are scaled by the ratio of the median
    box sizes and given in 1.5 mm voxels (levels x1, x2, x4 then span about 12-96 mm instead of 4-32 mm);
  * intensity window (-1000, 1000) HU, the window PER's longitudinal world model reads (lesions in lung, liver, bone and
    nodes; the tutorial's (-1024, 300) clips bone); outside the scan, and baseline voxels without data, are -1024 HU;
  * negatives: random places of the follow-up domain, the cached world-model training centres (as the lung script uses
    the lung world model's); one pair whose volume is thinner than the world-model cube has none and uses its domain.
Everything else (positive / negative sampling 1:1, flips, quarter turns, intensity scale and shift, optimiser, schedule,
matching, sampler, NMS) is the lung script's.

The reported location is the centre of the highest-scoring box, or of the highest-scoring box at least 10 mm across
(the detector's own size estimate); the pair score is that box's score. The checkpoint and the rule are chosen on the
development pairs by PER's pair hit (the reported point, rounded to the nearest follow-up voxel, inside a target grown
by 3 mm: perr/course_candidates.label_maps, as PER's candidates are labelled), over pairs with a target; ties go to the
progression AUROC over all development pairs, then to the earlier checkpoint. The 64 evaluation pairs
(sota/eval_course.txt) are read once, at the end, and written as results/sota/<method>_course.jsonl (zyx: voxel index on
the native follow-up grid), scored by sota/lesionlocator/score_course.py and sota/retinanet_ct/score_course.py (PER's
summarize_course.summary); no evaluation label is read here.

    venvs/clipvenv/bin/python train_ct_retinanet_course.py --boxes
    CUDA_VISIBLE_DEVICES=0 venvs/clipvenv/bin/python train_ct_retinanet_course.py --out runs_det/rc_scratch --method retinanet_ct_scratch
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from perr.course_wm import H, ISO, WINDOW, to_original

TRAIN_ROWS = "data/course_reader/train.jsonl"      # PER's longitudinal verifier's training rows
DEV_ROWS = "data/course_reader/val.jsonl"          # its validation rows (pairs of 24 held-out training patients)
EVAL = "sota/eval_course.txt"
BOXES = "sota/retinanet_ct/course_boxes.json"
AUTOPET_CACHE = "/raid/home/CAMCA/hj880/perr/cache/autopet"
PATCH = (64, 80, 80)                                 # z, y, x voxels of 1.5 mm = 96 x 120 x 120 mm
ROI = (112, 144, 144)                                # 168 x 216 x 216 mm
LUNG_ANCHORS = [[5, 6, 4], [6, 4, 6], [8, 7, 7]]     # the lung script's (z, y, x mm)
ANCHOR_SCALE = 3.05                                  # median progressing-lesion box 17.5 mm / median LUNA16 box 5.7 mm
ANCHORS = [[round(v * ANCHOR_SCALE / H) for v in a] for a in LUNG_ANCHORS]  # 1.5 mm voxels: [[10, 12, 8], [12, 8, 12], [16, 14, 14]]
HU = WINDOW                                          # (-1000, 1000): PER's longitudinal window
AIR = -1024.0
MIN_DIAM = 10.0
CHANNELS = 2


def pair_keys(path):
    return sorted({json.loads(l)["uid"] for l in open(path)})


# ------------------------------------------------------------------------------------------------ training boxes

def pair_boxes(key):
    """Every target's tight box on the 1.5 mm grid (iso voxel index [z0, y0, x0, z1, y1, x1], a voxel spans its index
    +- 0.5), clipped to the cached grid; a target whose centre lies outside the grid is dropped (counted)."""
    from perr.autopet import load_pair
    from perr.course_candidates import selected_targets
    P = load_pair(key)
    meta = np.load(f"{ISO}/{key}.meta.npy", allow_pickle=True).item()
    sp, lo = np.array(P.spacing, float), np.array(meta["lo_mm"], float)
    shape = np.array(meta["shape"], float)
    t = selected_targets(P.targets, MIN_DIAM)
    lab = np.asarray(P.fu_lab)
    boxes, diam, dropped = [], [], []
    for r in t.itertuples():
        idx = np.argwhere(lab == int(r.fu_label))
        if not len(idx):
            dropped.append({"fu_label": int(r.fu_label), "why": "no voxels"})
            continue
        b = np.concatenate([((idx.min(0) - 0.5) * sp - lo) / H, ((idx.max(0) + 0.5) * sp - lo) / H])
        c = (b[:3] + b[3:]) / 2
        if np.any(c < -0.5) or np.any(c > shape - 0.5):
            dropped.append({"fu_label": int(r.fu_label), "why": "outside the cached grid", "diam_mm": float(r.diam_mm)})
            continue
        b = np.concatenate([np.maximum(b[:3], -0.5), np.minimum(b[3:], shape - 0.5)])
        boxes.append([round(float(v), 3) for v in b])
        diam.append(round(float(r.diam_mm), 3))
    return key, {"boxes": boxes, "diam": diam, "dropped": dropped}


def build_boxes():
    from multiprocessing import Pool
    keys = pair_keys(TRAIN_ROWS)
    with Pool(16) as p:
        out = dict(p.map(pair_boxes, keys, chunksize=1))
    os.makedirs(os.path.dirname(BOXES), exist_ok=True)
    json.dump(out, open(BOXES, "w"))
    b = np.array([x for r in out.values() for x in r["boxes"]], float).reshape(-1, 6)
    d = np.array([x for r in out.values() for x in r["diam"]], float)
    size = (b[:, 3:] - b[:, :3]).mean(1) * H
    lung = []
    for u in open("results/lung_train_uids_100.txt").read().split():           # the lung detector's 100% training set
        from perr.context_wm import ORGANS
        lung += [dd for _, dd in np.load(f"{ORGANS['lung']['src']}/{u}.meta.npy", allow_pickle=True).item()["nodules"]]
    drop = [x for r in out.values() for x in r["dropped"]]
    print(f"{len(keys)} training pairs ({sum(1 for r in out.values() if r['boxes'])} with a target box), {len(b)} boxes; "
          f"dropped {len(drop)}: {drop}", flush=True)
    for name, v in (("equivalent diameter", d), ("box size mm (mean side)", size), ("LUNA16 box (diameter)", np.array(lung))):
        print(f"  {name:24s} percentiles 0/10/25/50/75/90/99/100: {np.percentile(v, [0, 10, 25, 50, 75, 90, 99, 100]).round(1)}")
    print(f"  ratio of median box sizes progressing lesions / LUNA16: {np.median(size) / np.median(lung):.2f}")


_BOX_CACHE = None


def train_boxes():
    global _BOX_CACHE
    if _BOX_CACHE is None:
        _BOX_CACHE = json.load(open(BOXES))
    return _BOX_CACHE


# ------------------------------------------------------------------------------------------------ the lung recipe

class Scan:
    """One pair on the 1.5 mm grid: follow-up and registered baseline (memory-mapped), boxes, negative centres."""

    def __init__(self, uid, with_boxes=False):
        self.uid = uid
        self.FU = np.load(f"{ISO}/{uid}.fu.npy", mmap_mode="r")
        self.BL = np.load(f"{ISO}/{uid}.bl.npy", mmap_mode="r")
        self.meta = np.load(f"{ISO}/{uid}.meta.npy", allow_pickle=True).item()
        self.native_shape = np.load(f"{AUTOPET_CACHE}/{uid}.fu.npy", mmap_mode="r").shape
        b = train_boxes()[uid]["boxes"] if with_boxes else []
        self.boxes = np.array(b, np.float32).reshape(-1, 6)
        self.centres = np.load(f"{ISO}/{uid}.centres.npy", mmap_mode="r")      # places in the follow-up domain
        if with_boxes and not len(self.centres):
            dom = np.argwhere(np.load(f"{ISO}/{uid}.dom.npy") > 0)
            self.centres = dom[np.random.default_rng(0).choice(len(dom), size=min(len(dom), 40000), replace=False)]

    def channels(self):
        return [self.FU, self.BL][:CHANNELS]


def norm(a):
    return np.clip((np.asarray(a, np.float32) - HU[0]) / (HU[1] - HU[0]), 0, 1)


def crop(V, lo):
    out = np.full(PATCH, AIR, np.float32)
    src = [slice(max(0, l), min(s, l + p)) for l, s, p in zip(lo, V.shape, PATCH)]
    dst = [slice(a.start - l, a.stop - l) for a, l in zip(src, lo)]
    out[tuple(dst)] = V[tuple(src)]
    return out


class Patches(IterableDataset):
    """Patches of 64 x 80 x 80 voxels (96 x 120 x 120 mm), as the tutorial's positive / negative crops (pos = neg = 1):
    half centred near a random target box of a random pair with targets, half on a random place of the follow-up
    domain of a random pair; random flips on every axis, a random quarter turn in the axial plane, intensity scale and
    shift, the same for both channels."""

    def __init__(self, uids, seed):
        self.uids, self.seed = uids, seed
        self.pos_uids = [u for u in uids if len(Scan(u, True).boxes)]

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
                scans[u] = Scan(u, True)
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
            img = np.stack([norm(crop(V, lo)) for V in s.channels()])
            bx = s.boxes - np.concatenate([lo, lo]).astype(np.float32)
            ctr = (bx[:, :3] + bx[:, 3:]) / 2
            keep = np.all((ctr >= 0) & (ctr < np.array(PATCH)), 1)
            bx = np.clip(bx[keep], 0, np.concatenate([PATCH, PATCH]).astype(np.float32))
            for ax in range(3):
                if rng.random() < 0.5:
                    img = img[(slice(None),) * (ax + 1) + (slice(None, None, -1),)]
                    n = PATCH[ax]
                    bx[:, [ax, ax + 3]] = n - bx[:, [ax + 3, ax]]
            k = int(rng.integers(4))
            for _ in range(k):                        # quarter turn in the (y, x) plane: (y, x) -> (X - 1 - x, y)
                img = np.rot90(img, 1, axes=(2, 3))
                y0, x0, y1, x1 = bx[:, 1].copy(), bx[:, 2].copy(), bx[:, 4].copy(), bx[:, 5].copy()
                X = img.shape[2]                      # PATCH is square in y, x
                bx[:, 1], bx[:, 4] = X - x1, X - x0
                bx[:, 2], bx[:, 5] = y0, y1
            img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
            yield (torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(bx.astype(np.float32)),
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
                             n_input_channels=CHANNELS, conv1_t_stride=stride, conv1_t_size=[7, 7, 7],
                             bias_downsample=not pretrained)
    if pretrained:                                   # MedicalNet ResNet-50 (Med3D, 23 datasets): same conv1 and blocks
        sd = {k.replace("module.", ""): v for k, v in
              resnet.get_pretrained_resnet_medicalnet(50, device="cpu", datasets23=True).items()}
        if CHANNELS > 1:                             # one-channel conv1 kernel shared by both channels, halved
            sd["conv1.weight"] = sd["conv1.weight"].repeat(1, CHANNELS, 1, 1, 1) / CHANNELS
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
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
    """Boxes of one pair: [(score, centre zyx in native follow-up voxels, size mm)], best first; the centre is kept
    inside the native grid."""
    det.eval()
    x = torch.from_numpy(np.stack([norm(V) for V in s.channels()])).to(device)
    out = det([x], use_inferer=True)[0]
    b, sc = out["box"].float().cpu().numpy(), out["label_scores"].float().cpu().numpy()
    det.train()
    rows = []
    for i in np.argsort(-sc):
        c = (b[i, :3] + b[i, 3:]) / 2
        zyx = np.clip(to_original(c, s.meta), 0, np.array(s.native_shape) - 1)
        rows.append({"score": float(sc[i]), "zyx": [float(v) for v in zyx],
                     "size_mm": float(np.mean(b[i, 3:] - b[i, :3]) * H)})
    return rows


def answer(dets, rule):
    """The reported box: the top box, or (rule top10) the top box at least 10 mm across (else the top box)."""
    cand = [r for r in dets if rule == "top" or r["size_mm"] >= MIN_DIAM] or dets[:1]
    return cand[0] if cand else None


def target_map(key):
    """PER's target map of a development pair (perr/course_candidates.label_maps: targets grown by 3 mm on the native
    follow-up grid), kept as its bounding box: (offset, mask) or None for a pair without a target."""
    from perr.autopet import load_pair
    from perr.course_candidates import label_maps
    T, _ = label_maps(load_pair(key), MIN_DIAM)
    if not T.any():
        return None
    idx = [np.nonzero(T.any(axis=ax))[0] for ax in ((1, 2), (0, 2), (0, 1))]
    sl = tuple(slice(i.min(), i.max() + 1) for i in idx)
    return np.array([s.start for s in sl]), T[sl].copy()


def on_target(tm, zyx):
    """PER's candidate label: the point rounded to the nearest voxel lies in the grown target map."""
    off, m = tm
    i = np.array([int(round(c)) for c in zyx]) - off
    return bool(np.all(i >= 0) and np.all(i < m.shape) and m[tuple(i)])


def metrics(scans, dets, tmaps):
    """Per rule: pair hit (pairs with a target) and progression AUROC (all pairs; score = the reported box's score)."""
    out = {}
    for rule in ("top", "top10"):
        y, s, h = [], [], []
        for p in scans:
            a = answer(dets[p.uid], rule)
            tm = tmaps[p.uid]
            y.append(tm is not None)
            s.append(a["score"] if a else 0.0)
            if tm is not None:
                h.append(a is not None and on_target(tm, a["zyx"]))
        y, s = np.array(y), np.array(s)
        pos, neg = s[y], s[~y]
        auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else float("nan")
        out[rule] = (float(np.mean(h)) if h else float("nan"), auc)
    return out


def write_answers(path, scans, dets, rule):
    """One line per pair (sota/lesionlocator/score_course.py format): the chosen rule's box first, its score as the
    progression score, the top 10 boxes as candidates."""
    with open(path, "w") as f:
        for s in scans:
            a = answer(dets[s.uid], rule)
            f.write(json.dumps({"case": s.uid, "present": a is not None, "zyx": a["zyx"] if a else None,
                                "score": a["score"] if a else 0.0, "diameter_mm": a["size_mm"] if a else None,
                                "rule": rule, "candidates": [{"zyx": r["zyx"], "diameter_mm": r["size_mm"], "score": r["score"]}
                                                             for r in dets[s.uid][:10]]}) + "\n")


def main():
    global CHANNELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", action="store_true", help="build the training-box cache and exit (CPU)")
    ap.add_argument("--out")
    ap.add_argument("--method", help="name of the results/sota/<method>_course.jsonl answer file")
    ap.add_argument("--channels", type=int, default=2, choices=[1, 2], help="2: follow-up + registered baseline; 1: follow-up")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--val-every", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=4, help="patches per step (the tutorial's 4)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=5000, help="linear warm-up steps (the tutorial warms up for 10 epochs)")
    ap.add_argument("--clip", type=float, default=10.0, help="gradient-norm clipping, a guard against early collapse")
    ap.add_argument("--pretrained", action="store_true", help="MedicalNet ResNet-50 backbone weights")
    ap.add_argument("--smoke", type=int, default=0, help="test run: this many development pairs, no evaluation")
    ap.add_argument("--max-train", type=int, default=0, help="test run: only N training pairs")
    a = ap.parse_args()
    if a.boxes:
        build_boxes()
        return
    CHANNELS = a.channels
    assert a.out and (a.method or a.smoke)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    train = pair_keys(TRAIN_ROWS)
    val = pair_keys(DEV_ROWS)
    ev = open(EVAL).read().split()
    pid = lambda k: k.rsplit("_", 1)[0]                                    # noqa: E731
    assert not {pid(k) for k in train} & {pid(k) for k in val + ev} and set(train) <= set(train_boxes())
    if a.max_train:
        pos = [k for k in train if train_boxes()[k]["boxes"]]
        neg = [k for k in train if not train_boxes()[k]["boxes"]]
        train = sorted(pos[: a.max_train // 2] + neg[: a.max_train - a.max_train // 2])
    open(f"{a.out}/train_pairs.txt", "w").write("\n".join(train) + "\n")
    print(f"{len(train)} training pairs ({sum(len(Scan(u, True).boxes) for u in train)} boxes), {len(val)} development "
          f"pairs; {CHANNELS} channel(s), anchors {ANCHORS} voxels of {H} mm, window {HU} HU", flush=True)
    det = make_detector(dev, a.pretrained)
    opt = torch.optim.SGD(det.network.parameters(), lr=a.lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda g: min(1.0, (g + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(g, a.iters) / a.iters)))
    dl = DataLoader(Patches(train, 0), batch_size=a.bs, num_workers=8, collate_fn=lambda b: b, persistent_workers=True)
    if a.smoke:                                       # a few development pairs, with and without a target
        tgt = {json.loads(l)["uid"] for l in open(DEV_ROWS) if json.loads(l)["label"]}
        val = sorted([k for k in val if k in tgt][: (a.smoke + 1) // 2] + [k for k in val if k not in tgt][: a.smoke // 2])
    vscans = [Scan(u) for u in val]
    tmaps = {s.uid: target_map(s.uid) for s in vscans}
    print(f"development pairs with a target: {sum(t is not None for t in tmaps.values())} of {len(vscans)}", flush=True)
    best, t0, run, log = ((-2.0, -2.0), None, 0), time.time(), [], []
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
            m = metrics(vscans, dets, tmaps)
            log.append({"iter": g + 1, "loss": float(np.mean(run)), **{r: list(v) for r, v in m.items()}})
            print(f"iter {g + 1} loss {np.mean(run):.4f} dev pair hit / AUROC: top {m['top'][0]:.3f} / {m['top'][1]:.3f}, "
                  f"top10 {m['top10'][0]:.3f} / {m['top10'][1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
            run = []
            for rule in ("top", "top10"):
                key = tuple(-1.0 if np.isnan(v) else v for v in m[rule])
                if key > best[0]:                     # pair hit, then AUROC; ties keep the earlier checkpoint
                    best = (key, rule, g + 1)
                    torch.save(det.network.state_dict(), f"{a.out}/model.pt")
            json.dump({"log": log, "best": best}, open(f"{a.out}/log.json", "w"), indent=1)
        if g + 1 == a.iters:
            break
    print(f"best development pair hit {best[0][0]:.3f}, AUROC {best[0][1]:.3f} (rule {best[1]}, iter {best[2]})", flush=True)
    det.network.load_state_dict(torch.load(f"{a.out}/model.pt", map_location=dev))
    if a.smoke:                                       # the answer path, on the smoke development pairs only
        dets = {s.uid: detect(det, s, dev) for s in vscans}
        write_answers(f"{a.out}/smoke_answers_dev.jsonl", vscans, dets, best[1])
        print(f"smoke: chosen checkpoint re-read on the development pairs, {metrics(vscans, dets, tmaps)[best[1]]}; "
              f"answers in {a.out}/smoke_answers_dev.jsonl", flush=True)
        return
    scans = [Scan(u) for u in ev]                     # read once, with the checkpoint and rule fixed above
    dets = {s.uid: detect(det, s, dev) for s in scans}
    json.dump({u: d[:20] for u, d in dets.items()}, open(f"{a.out}/eval_course.json", "w"))
    write_answers(f"results/sota/{a.method}_course.jsonl", scans, dets, best[1])
    print(f"eval: {len(scans)} pairs answered with rule {best[1]} -> results/sota/{a.method}_course.jsonl "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
