"""Can a MIMIC-trained radiograph reader show where its finding is, without ever seeing a box? Readers trained on the
60k MIMIC images with report labels only, evaluated on external radiologist boxes (NIH ChestX-ray14: nodule, mass;
ChestX-Det test: nodule, mass, fracture):

  two-stream, world-model views   train_cxr_evidence.py --crops focal: the pointed place is the view with the largest
                                  attention for the finding (lung lesion -> nodule / mass boxes, fracture -> fracture)
  two-stream, random views        --crops focal_random, the same readout (control)
  image only + Grad-CAM           train_cxr_cls.py image mode: the peak of the finding's Grad-CAM inside the thorax

hit = the pointed place lies in a box of the finding (among images with such a box); on ChestX-Det also presence AUROC
(images without the finding's boxes are negatives) and LROC (score_cxr_focal.lroc).

    CUDA_VISIBLE_DEVICES=1 python eval_cxr_mil_pointing.py
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr/sota_cxr")
sys.path.insert(0, "/raid/home/CAMCA/hj880/perr/sota_cxr/densenet121")
from common import FINDINGS  # noqa: E402
from train import build_model  # noqa: E402

from cxr_recall import boxes  # noqa: E402
from perr.cxr import CACHE  # noqa: E402
from perr.cxr_anat import search_domain  # noqa: E402
from perr.cxr_wm import valid_region  # noqa: E402
from score_cxr_focal import lroc  # noqa: E402
from train_cxr_evidence import TwoStream, norm, prep  # noqa: E402

ROOT = "/raid/home/CAMCA/hj880/perr"
TARGETS = {"Lung Lesion": ("Nodule", "Mass"), "Fracture": ("Fracture",)}


@torch.no_grad()
def two_stream(name, design, keys, dev):
    model = TwoStream().to(dev).eval()
    model.load_state_dict(torch.load(f"{ROOT}/runs_cls/{name}/best.pt", map_location=dev))
    out = {}
    for i in range(0, len(keys), 32):
        ks = keys[i:i + 32]
        x = torch.from_numpy(np.stack([np.load(f"{CACHE}/{k}.npy") for k in ks])).to(dev)
        v = torch.from_numpy(np.stack([np.load(f"{ROOT}/cache/cxr_crops_{design}/{k}.npy") for k in ks])).to(dev)
        pts = [np.load(f"{ROOT}/cache/cxr_crops_{design}/{k}.pts.npy") for k in ks]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xs, vs = prep(x, v, 320, False)
            fg = model.pool(model.g(xs))
            fl = model.pool(model.l(vs)).view(len(ks), v.shape[1], -1)
            a = model.att(fl).softmax(1).float()
            logit = model(xs, vs, torch.ones(len(ks), device=dev)).float()
        for b, k in enumerate(ks):
            out[k] = {}
            for f in TARGETS:
                j = FINDINGS.index(f)
                r, c = pts[b][int(a[b, :len(pts[b]), j].argmax())]
                out[k][f] = (float(logit[b, j]), float(r), float(c))
    return out


def gradcam(name, keys, dev):
    model = build_model().to(dev).eval()
    model.load_state_dict(torch.load(f"{ROOT}/runs_cls/{name}/best.pt", map_location=dev))
    out = {}
    for k in keys:
        img = np.load(f"{CACHE}/{k}.npy")
        x = F.interpolate(torch.from_numpy(img).float().div(255)[None, None].to(dev), size=(320, 320), mode="bilinear",
                          antialias=True, align_corners=False)
        x = norm(x)
        feats = model.features(x)
        feats.retain_grad()
        logit = model.classifier(F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1))
        D = search_domain(k, valid_region(img))
        out[k] = {}
        for f in TARGETS:
            j = FINDINGS.index(f)
            model.zero_grad()
            if feats.grad is not None:
                feats.grad = None
            logit[0, j].backward(retain_graph=True)
            w = feats.grad.mean((2, 3), keepdim=True)
            cam = F.relu((w * feats).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
            cam[~D] = -1
            r, c = np.unravel_index(int(np.argmax(cam)), cam.shape)
            out[k][f] = (float(logit[0, j]), float(r), float(c))
    return out


def evaluate(pred, b, keys, exhaustive):
    rows = []
    for f, types in TARGETS.items():
        bt = b[b.finding.isin(types)]
        if not len(bt):
            continue
        pos = np.array([k in set(bt.key) for k in keys])
        hit = np.array([any(r.x <= pred[k][f][2] <= r.x + r.w and r.y <= pred[k][f][1] <= r.y + r.h
                            for r in bt[bt.key == k].itertuples()) for k in keys])
        score = np.array([pred[k][f][0] for k in keys])
        row = {"finding": f, "boxed images": int(pos.sum()), "hit": hit[pos].mean()}
        if exhaustive:
            row.update({"AUROC": lroc(score, pos, np.ones(len(keys), bool)), "LROC": lroc(score, pos, hit)})
        rows.append(row)
    return rows


def main():
    dev = torch.device("cuda", 0)
    res = []
    for which, split in (("nih", None), ("cxdet", "test")):
        b = boxes(which)
        g = pd.read_csv(f"{ROOT}/results/cxr_{which}.csv")
        if split:
            b, g = b[b.split == split], g[g.split == split]
        keys = sorted(g.key)
        if which == "nih":
            keys = sorted(set(b[b.finding.isin(("Nodule", "Mass"))].key))   # NIH boxes are not exhaustive: positives only
        for label, pred in (("two-stream, world-model views", two_stream("ev60k_focal", "focal", keys, dev)),
                            ("two-stream, random views", two_stream("ev60k_focal_random", "focal_random", keys, dev)),
                            ("image only + Grad-CAM", gradcam("dn60k_image", keys, dev))):
            for r in evaluate(pred, b, keys, which == "cxdet"):
                res.append({"set": which, "reader": label, **r})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(res).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
