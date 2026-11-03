"""Supervised 3-D liver-tumour detector for the liver columns of Table 2: the lung detector's recipe
(train_ct_retinanet.py: MONAI's RetinaNet, ResNet-50 FPN, the MONAI LUNA16 tutorial configuration converted to 1 mm:
ATSS matching, hard-negative sampling, SGD with Nesterov momentum at 1e-2, NMS 0.22; from scratch, or from MedicalNet
weights exactly as the lung script loads them), trained on the tumour boxes of exactly the scans PER's liver verifier
trains on, and read on the same 1 mm grid as PER's liver world model.

Training scans: the cases of PER's final liver verifier (runs_reader/q8_liver_t08h3, data/liver_reader_all3c_t08h/
train.jsonl): 71 MSD Task03 training-pool cases (the evaluation half and the 7 verifier-validation cases left out),
194 TCIA CRLM cases and 203 MSD Task08 cases (the 100 Task08 development cases left out); 468 scans on cache/iso_liver.
Boxes: every annotated tumour, i.e. every 26-connected component of the tumour label exactly as the caches define them
(perr/liver.py, liver_crlm.py, liver_t08.py), as its tight bounding box in the original grid moved to the 1 mm grid
(--boxes builds sota/retinanet_ct/liver_boxes.json from the training labels only).

Changes from the lung recipe, and why:
  * anchors x2.65: the tutorial's anchor shapes fit LUNA16 nodules (median training box 5.7 mm across); liver tumour
    boxes are larger (median 15.2 mm, mean side; 90th percentile 52 mm), so the three shapes are scaled by the ratio of
    the median box sizes (levels x1, x2, x4 then span about 11-84 mm instead of 4-32 mm);
  * intensity window (-200, 300) HU, the window PER's liver world model reads (the tutorial's (-1024, 300) is a lung
    window that leaves liver contrast in a small part of the range); outside the scan is air (-1024 HU), as before.
Everything else (patch 96 x 128 x 128 mm, positive / negative sampling, augmentation, optimiser, schedule, matching,
sampler, NMS, sliding-window inference) is the lung script's.

The reported location is the centre of the highest-scoring box, or of the highest-scoring box at least 10 mm across
(the detector's own size estimate); the checkpoint and that rule are chosen on the 100 held-out Task08 development
cases (results/t08_dev_cases.txt) by PER's liver hit (the point inside a target tumour >= 10 mm grown by 3 mm:
perr.liver_t08.label_at == "big"). The evaluation sets (MSD evaluation half, sota/eval_liver.txt; HCC-TACE-Seg,
sota/eval_liver_ext.txt) are read once, at the end, and written as results/sota/<method>_liver.jsonl and
results/sota/<method>_liver_ext.jsonl (sota/README.md format) for score_sota.py; no evaluation label is read here.

    venvs/clipvenv/bin/python train_ct_retinanet_liver.py --boxes
    CUDA_VISIBLE_DEVICES=0 venvs/clipvenv/bin/python train_ct_retinanet_liver.py --out runs_det/rl_scratch --method retinanet_ct_scratch
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

TRAIN_ROWS = "data/liver_reader_all3c_t08h/train.jsonl"     # PER's final liver verifier's training rows
DEV = "results/t08_dev_cases.txt"                            # 100 Task08 cases held out of the verifier
EVAL = {"liver": "sota/eval_liver.txt", "liver_ext": "sota/eval_liver_ext.txt"}
BOXES = "sota/retinanet_ct/liver_boxes.json"
PATCH = (96, 128, 128)                               # z, y, x at 1 mm (the lung script's)
LUNG_ANCHORS = [[5, 6, 4], [6, 4, 6], [8, 7, 7]]     # the lung script's (z, y, x mm)
ANCHOR_SCALE = 2.65                                  # median liver tumour box 15.2 mm / median LUNA16 box 5.7 mm (--boxes)
ANCHORS = [[round(v * ANCHOR_SCALE) for v in a] for a in LUNG_ANCHORS]      # [[13, 16, 11], [16, 11, 16], [21, 19, 19]]
ROI = (160, 224, 224)
HU = (-200.0, 300.0)                                 # PER's liver window (ORGANS["liver"]["window"])
AIR = -1024.0
MIN_DIAM = 10.0                                      # target tumours (perr/liver.py)
ORGAN = {"liver": "liver", "CRLM": "liver_crlm", "T08": "liver_t08", "HCC": "liver_ext"}


def organ_of(case):
    return ORGAN[case.split("_")[0]]


def label_fn(case):
    """PER's liver label of a point (original voxel index): "big" inside a target tumour >= 10 mm grown by 3 mm."""
    from perr import liver, liver_crlm, liver_ext, liver_t08
    return {"liver": liver, "liver_crlm": liver_crlm, "liver_t08": liver_t08, "liver_ext": liver_ext}[organ_of(case)].label_at


def train_cases():
    return sorted({json.loads(l)["uid"] for l in open(TRAIN_ROWS)})


# ------------------------------------------------------------------------------------------------ training boxes

def _tumour_mask(case):
    """The tumour label of a training case in LPS [z, y, x], read exactly as its cache builder reads it."""
    import SimpleITK as sitk
    from perr import liver, liver_crlm, liver_t08
    org = organ_of(case)
    if org == "liver":
        path, f = f"{liver.ROOT}/labelsTr/{case}.nii.gz", lambda a: a == 2
    elif org == "liver_crlm":
        path, f = f"{liver_crlm.NIFTI}/labelsTr/{case}.nii.gz", lambda a: (a & liver_crlm.TUMOUR) > 0
    elif org == "liver_t08":
        path, f = f"{liver_t08.ROOT}/labelsTr/{liver_t08.to_name(case)}.nii.gz", lambda a: a == 2
    else:
        raise ValueError(f"{case}: not a training case")
    img = sitk.DICOMOrient(sitk.ReadImage(path), "LPS")
    sx, sy, sz = img.GetSpacing()
    return f(sitk.GetArrayFromImage(img).astype(np.uint8)), (sz, sy, sx)


def case_boxes(case):
    """Every tumour component's tight box on the 1 mm grid (iso voxel index, [z0, y0, x0, z1, y1, x1], a voxel spans
    its index +- 0.5), with its equivalent diameter; checked against the cache's component list."""
    from scipy import ndimage as ndi
    org = organ_of(case)
    src = np.load(f"{ORGANS[org]['src']}/{case}.meta.npy", allow_pickle=True).item()
    iso = np.load(f"{ORGANS[org]['iso']}/{case}.meta.npy", allow_pickle=True).item()
    tum, sp = _tumour_mask(case)
    sp = np.array(sp, float)
    assert np.allclose(sp, src["spacing"]) and np.allclose(sp, iso["spacing"]), case
    comp, n = ndi.label(tum, structure=np.ones((3, 3, 3)))
    boxes, diam = [], []
    for i, sl in enumerate(ndi.find_objects(comp), 1):
        v = float((comp[sl] == i).sum()) * float(np.prod(sp))
        diam.append((6 * v / np.pi) ** (1 / 3))
        lo = np.array([s.start for s in sl], float) - 0.5
        hi = np.array([s.stop for s in sl], float) - 0.5
        boxes.append([*(lo * sp - iso["lo_mm"]), *(hi * sp - iso["lo_mm"])])
    ref = [d for _, d in src["nodules"]]
    assert len(ref) == len(diam) and np.allclose(ref, diam, rtol=1e-6), (case, len(ref), len(diam))
    # sanity check of the box -> original-index mapping: where does each box centre fall in PER's label masks?
    lab = label_fn(case)
    centre_label = [lab(case, to_original((np.array(b[:3]) + np.array(b[3:])) / 2, iso)) for b in boxes]
    return case, {"boxes": [[round(float(v), 3) for v in b] for b in boxes], "diam": [round(d, 3) for d in diam],
                  "centre_label": centre_label}


def build_boxes():
    from multiprocessing import Pool
    cases = train_cases()
    with Pool(16) as p:
        out = dict(p.map(case_boxes, cases, chunksize=1))
    os.makedirs(os.path.dirname(BOXES), exist_ok=True)
    json.dump(out, open(BOXES, "w"))
    b = np.array([x for r in out.values() for x in r["boxes"]], float).reshape(-1, 6)
    d = np.array([x for r in out.values() for x in r["diam"]], float)
    ext = b[:, 3:] - b[:, :3]
    size = ext.mean(1)
    lab = [l for r in out.values() for l in r["centre_label"]]
    big = d >= MIN_DIAM
    lung = []
    for u in open("results/lung_train_uids_100.txt").read().split():           # the lung detector's 100% training set
        lung += [dd for _, dd in np.load(f"{ORGANS['lung']['src']}/{u}.meta.npy", allow_pickle=True).item()["nodules"]]
    print(f"{len(cases)} training scans ({sum(1 for r in out.values() if r['boxes'])} with a tumour), {len(b)} boxes "
          f"({int(big.sum())} >= 10 mm)", flush=True)
    for name, v in (("equivalent diameter", d), ("box size (mean side)", size), ("box side z", ext[:, 0]),
                    ("box side y", ext[:, 1]), ("box side x", ext[:, 2]), ("LUNA16 box (diameter)", np.array(lung))):
        print(f"  {name:22s} percentiles 0/10/25/50/75/90/99/100: {np.percentile(v, [0, 10, 25, 50, 75, 90, 99, 100]).round(1)}")
    print(f"  ratio of median box sizes liver / LUNA16: {np.median(size) / np.median(lung):.2f}")
    print(f"  box centres of tumours >= 10 mm in PER's masks: big {sum(l == 'big' for l, g in zip(lab, big) if g)}, "
          f"small {sum(l == 'small' for l, g in zip(lab, big) if g)}, none {sum(l == 'none' for l, g in zip(lab, big) if g)}")


_BOX_CACHE = None


def train_boxes():
    global _BOX_CACHE
    if _BOX_CACHE is None:
        _BOX_CACHE = json.load(open(BOXES))
    return _BOX_CACHE


# ------------------------------------------------------------------------------------------------ the lung recipe

class Scan:
    def __init__(self, uid, with_boxes=False):
        self.uid = uid
        org = organ_of(uid)
        self.V = np.load(f"{ORGANS[org]['iso']}/{uid}.vol.npy", mmap_mode="r")
        self.meta = np.load(f"{ORGANS[org]['iso']}/{uid}.meta.npy", allow_pickle=True).item()
        self.src_shape = np.load(f"{ORGANS[org]['src']}/{uid}.vol.npy", mmap_mode="r").shape
        b = train_boxes()[uid]["boxes"] if with_boxes else []
        self.boxes = np.array(b, np.float32).reshape(-1, 6)
        self.centres = np.load(f"{ORGANS[org]['iso']}/{uid}.centres.npy", mmap_mode="r")    # places in the grown liver


def norm(a):
    return np.clip((np.asarray(a, np.float32) - HU[0]) / (HU[1] - HU[0]), 0, 1)


def crop(V, lo):
    out = np.full(PATCH, AIR, np.float32)
    src = [slice(max(0, l), min(s, l + p)) for l, s, p in zip(lo, V.shape, PATCH)]
    dst = [slice(a.start - l, a.stop - l) for a, l in zip(src, lo)]
    out[tuple(dst)] = V[tuple(src)]
    return out


class Patches(IterableDataset):
    """Patches of 96 x 128 x 128 mm, as the tutorial's positive / negative crops (pos = neg = 1): half centred near a
    random annotated tumour of a random scan with tumours, half on a random place of the liver domain of a random scan;
    random flips on every axis, a random quarter turn in the axial plane, intensity scale and shift."""

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
    """Boxes of one scan: [(score, centre zyx in original voxels, size mm)], best first. The centre is kept inside the
    original volume (a box may reach half a voxel past the last 1 mm sample)."""
    det.eval()
    x = torch.from_numpy(norm(s.V))[None].to(device)
    out = det([x], use_inferer=True)[0]
    b, sc = out["box"].float().cpu().numpy(), out["label_scores"].float().cpu().numpy()
    det.train()
    rows = []
    for i in np.argsort(-sc):
        c = (b[i, :3] + b[i, 3:]) / 2
        zyx = np.clip(to_original(c, s.meta), 0, np.array(s.src_shape) - 1)
        rows.append({"score": float(sc[i]), "zyx": [float(v) for v in zyx], "size_mm": float(np.mean(b[i, 3:] - b[i, :3]))})
    return rows


def answer(dets, rule):
    """The reported box: the top box, or (rule top10) the top box at least 10 mm across (else the top box)."""
    cand = [r for r in dets if rule == "top" or r["size_mm"] >= MIN_DIAM] or dets[:1]
    return cand[0] if cand else None


def hits(scans, dets):
    """PER's liver hit rate (the point inside a target tumour >= 10 mm grown by 3 mm) of the two reporting rules."""
    out = {}
    for rule in ("top", "top10"):
        h = []
        for s in scans:
            a = answer(dets[s.uid], rule)
            h.append(a is not None and label_fn(s.uid)(s.uid, a["zyx"]) == "big")
        out[rule] = float(np.mean(h))
    return out


def write_answers(path, scans, dets, rule):
    """sota/README.md format: one line per case, the chosen rule's box first, the top 10 boxes as candidates."""
    with open(path, "w") as f:
        for s in scans:
            a = answer(dets[s.uid], rule)
            f.write(json.dumps({"case": s.uid, "present": a is not None, "zyx": a["zyx"] if a else None,
                                "diameter_mm": a["size_mm"] if a else None, "score": a["score"] if a else None,
                                "rule": rule, "candidates": [{"zyx": r["zyx"], "diameter_mm": r["size_mm"], "score": r["score"]}
                                                             for r in dets[s.uid][:10]]}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", action="store_true", help="build the training-box cache and exit (CPU)")
    ap.add_argument("--out")
    ap.add_argument("--method", help="name of the results/sota/<method>_liver*.jsonl answer files")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--val-every", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=4, help="patches per step (the tutorial's 4)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=5000, help="linear warm-up steps (the tutorial warms up for 10 epochs)")
    ap.add_argument("--clip", type=float, default=10.0, help="gradient-norm clipping, a guard against early collapse")
    ap.add_argument("--pretrained", action="store_true", help="MedicalNet ResNet-50 backbone weights")
    ap.add_argument("--smoke", type=int, default=0, help="test run: this many development cases, no evaluation")
    ap.add_argument("--max-train", type=int, default=0, help="test run: only the first N training scans")
    a = ap.parse_args()
    if a.boxes:
        build_boxes()
        return
    assert a.out and (a.method or a.smoke)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    train = train_cases()
    val = open(DEV).read().split()
    held = set(val) | {c for p in EVAL.values() for c in open(p).read().split()}
    assert not set(train) & held and set(train) <= set(train_boxes())
    if a.max_train:
        train = train[:: max(1, len(train) // a.max_train)][: a.max_train]
    open(f"{a.out}/train_cases.txt", "w").write("\n".join(train) + "\n")
    print(f"{len(train)} training scans ({sum(len(Scan(u, True).boxes) for u in train)} boxes), {len(val)} development "
          f"cases; anchors {ANCHORS} mm, window {HU} HU", flush=True)
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
            print(f"iter {g + 1} loss {np.mean(run):.4f} dev hit top {h['top']:.3f} top10 {h['top10']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            run = []
            for rule in ("top", "top10"):
                if h[rule] > best[0]:
                    best = (h[rule], rule, g + 1)
                    torch.save(det.network.state_dict(), f"{a.out}/model.pt")
            json.dump({"log": log, "best": best}, open(f"{a.out}/log.json", "w"), indent=1)
        if g + 1 == a.iters:
            break
    print(f"best development hit {best[0]:.3f} (rule {best[1]}, iter {best[2]})", flush=True)
    det.network.load_state_dict(torch.load(f"{a.out}/model.pt", map_location=dev))
    if a.smoke:                                       # the answer path, on the smoke development cases only
        dets = {s.uid: detect(det, s, dev) for s in vscans}
        write_answers(f"{a.out}/smoke_answers_dev.jsonl", vscans, dets, best[1])
        print(f"smoke: chosen checkpoint re-read on the development cases, hit {hits(vscans, dets)[best[1]]:.3f}; "
              f"answers in {a.out}/smoke_answers_dev.jsonl", flush=True)
        return
    for name, path in EVAL.items():                   # read once, with the checkpoint and rule fixed above
        scans = [Scan(u) for u in open(path).read().split()]
        dets = {s.uid: detect(det, s, dev) for s in scans}
        json.dump({u: d[:20] for u, d in dets.items()}, open(f"{a.out}/eval_{name}.json", "w"))
        write_answers(f"results/sota/{a.method}_{name}.jsonl", scans, dets, best[1])
        print(f"eval {name}: {len(scans)} cases answered with rule {best[1]} -> results/sota/{a.method}_{name}.jsonl "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
