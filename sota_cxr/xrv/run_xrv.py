"""torchxrayvision pretrained DenseNet-121s (official weights) on the official MIMIC-CXR-JPG test split, frontal images.

Preprocessing as in the torchxrayvision README / training script: the original 8-bit JPEG -> xrv.datasets.normalize
(maxval 255, i.e. [-1024, 1024]) -> single channel -> XRayCenterCrop (square crop of the longer side) ->
XRayResizer(224) (skimage).  Output = sigmoid of the logits (the library's op_norm calibration is monotonic per class
and is switched off so the CSV holds plain probabilities; AUROC is unaffected).  Findings the model does not predict
(Pleural Other, Support Devices) are left empty.

    python run_xrv.py --weights densenet121-res224-mimic_ch --name xrv_mimic_ch
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import FINDINGS, OUT, split_table  # noqa: E402

XRV_NAME = {"Pleural Effusion": "Effusion"}            # our finding name -> xrv pathology name


class Test(Dataset):
    def __init__(self, df):
        self.paths = df.path.tolist()
        self.tf = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.asarray(Image.open(self.paths[i]))      # same decoder as skimage.io.imread (PIL plugin)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3:
            img = img.mean(2)
        return torch.from_numpy(self.tf(img[None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    t0 = time.time()
    df = split_table(a.split)
    model = xrv.models.DenseNet(weights=a.weights, cache_dir="/raid/home/CAMCA/hj880/hf_home/xrv").cuda().eval()
    model.op_threshs = None
    targets = list(model.targets)
    dl = DataLoader(Test(df), batch_size=64, num_workers=16, worker_init_fn=lambda _: torch.set_num_threads(1))
    out = []
    with torch.no_grad():
        for x in dl:
            out.append(torch.sigmoid(model(x.cuda())).cpu().numpy())
    prob = np.concatenate(out)
    res = pd.DataFrame({"key": df.key})
    for f in FINDINGS:
        n = XRV_NAME.get(f, f)
        res[f] = prob[:, targets.index(n)] if n in targets else np.nan
    suffix = "" if a.split == "test" else f"_{a.split}"
    res.to_csv(f"{OUT}/{a.name}{suffix}.csv", index=False)
    print(a.weights, "targets:", targets)
    print(f"wrote {OUT}/{a.name}{suffix}.csv ({len(res)} rows) in {time.time() - t0:.0f} s; predicted:",
          [f for f in FINDINGS if XRV_NAME.get(f, f) in targets])


if __name__ == "__main__":
    main()
