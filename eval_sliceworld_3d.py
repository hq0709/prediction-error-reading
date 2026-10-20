"""A 3-D location from SliceWorld (train_sliceworld.py), so its state can be scored like ours (perr/score.py hit >= 6
mm): the slice with the highest lesion-occupancy probability, and in it the place where the occupancy logit's
gradient x input over the slice's DINOv2 patch tokens peaks (the 16 x 16 map upsampled bilinearly to the slice), the
usual way to localise with a classifier that has no location output.

    python eval_sliceworld_3d.py runs_sw/lung
Writes <run>/eval3d_<lo>_<hi>.json.
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from perr.context_wm import ORGANS, to_original
from perr.score import score_case
from perr.study import load_luna, pilot_cases
from train_sliceworld import SliceWorld, load

ISO = ORGANS["lung"]["iso"]


def locate(model, u, mu, sd, dev):
    f, z, _ = load(u)
    P = f[None].to(dev).float().requires_grad_(True)
    o = model(P)
    t = int(torch.argmax(o["occ"][0]))
    o["occ"][0, t].backward()
    s = (P.grad[0, t, 1:] * P[0, t, 1:]).sum(-1).relu().view(1, 1, 16, 16)
    meta = np.load(f"{ISO}/{u}.meta.npy", allow_pickle=True).item()
    Y, X = meta["shape"][1], meta["shape"][2]
    s = F.interpolate(s, size=(Y, X), mode="bilinear", align_corners=False)[0, 0]
    y, x = np.unravel_index(int(torch.argmax(s)), (Y, X))
    return to_original([float(z[t]), float(y), float(x)], meta), meta


def main():
    run = sys.argv[1]
    dev = torch.device("cuda")
    model = SliceWorld().to(dev)
    model.load_state_dict(torch.load(f"{run}/model.pt", map_location=dev))
    model.eval()
    nm = torch.load(f"{run}/norm.pt")
    for lo, hi in ((60, 160), (160, 260)):
        out, hits = {}, []
        for u in pilot_cases(hi)[lo:hi]:
            zyx, _ = locate(model, u, nm["mu"].to(dev), nm["sd"].to(dev), dev)
            h = score_case(load_luna(u), {"present": True, "zyx": [float(v) for v in zyx]})["hit6"]
            out[u] = {"zyx": [float(v) for v in zyx], "hit6": bool(h)}
            hits.append(h)
        json.dump(out, open(f"{run}/eval3d_{lo}_{hi}.json", "w"))
        print(f"eval {lo}-{hi}: 3-D hit (occupancy slice + gradient x input peak) {np.mean(hits):.3f}", flush=True)


if __name__ == "__main__":
    main()
