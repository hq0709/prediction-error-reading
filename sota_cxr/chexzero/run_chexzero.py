"""CheXzero (Tiu et al., Nat. Biomed. Eng. 2022), zero-shot, on the official MIMIC-CXR-JPG test split (frontal images).

Official code (rajpurkarlab/CheXzero @ 5c341db, ext_src/CheXzero) and the official ensemble of 10 checkpoints
(Google Drive folder in the README, hf_home/chexzero_weights).  The recipe of notebooks/zero_shot.ipynb:
  images  data_process.img_to_hdf5 + preprocess: cv2.imread -> RGB -> PIL -> longer side to 320 (ANTIALIAS = LANCZOS)
          -> pasted centred on a black 320 x 320 'L' canvas -> float; CXRTestDataset repeats it to 3 channels;
          zero_shot.make: Normalize(101.48761, 83.43944) then Resize(224, bicubic) (pretrained=True, ViT-B/32).
  model   zero_shot.load_clip(path, pretrained=True): CLIP ViT-B/32 skeleton, then the CheXzero state dict.
  prompts ("{}", "no {}") with the CheXpert label names (the CLIP tokenizer lowercases);
          p = exp(pos) / (exp(pos) + exp(neg)) on the raw cosine similarities (zero_shot.run_softmax_eval);
          ensemble = mean of the 10 checkpoints' probabilities (zero_shot.ensemble_models).
Only change: images are preprocessed in memory instead of through an .h5 file, and image / text embeddings are computed
in batches on the GPU (float32, TF32 off) instead of one image at a time on the CPU; the arithmetic is the same
(Resize with antialias=False, as torchvision 0.11 did for tensors).

    python run_chexzero.py
"""
import glob
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize

REPO = "/raid/home/CAMCA/hj880/ext_src/CheXzero"
WEIGHTS = "/raid/home/CAMCA/hj880/hf_home/chexzero_weights"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Image.ANTIALIAS = Image.LANCZOS                       # removed in Pillow 10; same filter
import clip  # noqa: E402  (CheXzero's copy of OpenAI CLIP)
clip._download.__defaults__ = ("/raid/home/CAMCA/hj880/hf_home/clip",)   # keep the ViT-B/32 skeleton off the home disk
import zero_shot  # noqa: E402
from data_process import preprocess  # noqa: E402
from common import FINDINGS, OUT, split_table  # noqa: E402

TEMPLATE = ("{}", "no {}")


def load_img(path):
    """data_process.img_to_hdf5, one image: cv2 read, BGR -> RGB, PIL, CheXzero's preprocess(desired_size=320)."""
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return np.asarray(preprocess(Image.fromarray(img), desired_size=320), np.float32)


def text_weights(model, template, dev):
    """zero_shot.zeroshot_classifier with one template: normalised text embedding per finding, (dim, n_findings)."""
    w = []
    with torch.no_grad():
        for f in FINDINGS:
            e = model.encode_text(clip.tokenize([template.format(f)], context_length=77).to(dev))
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(dim=0)
            w.append(e / e.norm())
    return torch.stack(w, dim=1)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    df = split_table("test")
    with Pool(32) as p:
        imgs = np.stack(p.map(load_img, df.path.tolist(), chunksize=16))          # (n, 320, 320) float32
    print(f"preprocessed {len(imgs)} images in {time.time() - t0:.0f} s", flush=True)
    # torchvision 0.11 (the repo's pin) resized tensors without antialiasing; newer versions default to antialias=True
    tf = Compose([Normalize((101.48761,) * 3, (83.43944,) * 3),
                  Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=False)])
    dev = torch.device("cuda")
    paths = sorted(glob.glob(f"{WEIGHTS}/*.pt"))
    preds = []
    for path in paths:
        model = zero_shot.load_clip(model_path=path, pretrained=True, context_length=77).to(dev).float().eval()
        wp, wn = text_weights(model, TEMPLATE[0], dev), text_weights(model, TEMPLATE[1], dev)
        lp, ln = [], []
        with torch.no_grad():
            for i in range(0, len(imgs), 128):
                x = tf(torch.from_numpy(imgs[i:i + 128]).to(dev).unsqueeze(1).repeat(1, 3, 1, 1))
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                lp.append((f @ wp).cpu().numpy()); ln.append((f @ wn).cpu().numpy())
        lp, ln = np.concatenate(lp).astype(np.float64), np.concatenate(ln).astype(np.float64)
        preds.append(np.exp(lp) / (np.exp(lp) + np.exp(ln)))
        os.makedirs(f"{OUT}/chexzero_parts", exist_ok=True)
        np.save(f"{OUT}/chexzero_parts/{os.path.basename(path)[:-3]}.npy", preds[-1])
        print(f"{os.path.basename(path)} done ({time.time() - t0:.0f} s)", flush=True)
        del model
        torch.cuda.empty_cache()
    avg = np.mean(preds, axis=0)
    res = pd.DataFrame(avg, columns=list(FINDINGS))
    res.insert(0, "key", df.key.values)
    res.to_csv(f"{OUT}/chexzero.csv", index=False)
    # single best checkpoint as well (the README's single-model option), for reference
    single = [p for p in paths if os.path.basename(p) == "best_64_5e-05_original_22000_0.864.pt"]
    if single:
        r1 = pd.DataFrame(preds[paths.index(single[0])], columns=list(FINDINGS))
        r1.insert(0, "key", df.key.values)
        r1.to_csv(f"{OUT}/chexzero_single.csv", index=False)
    print(f"wrote {OUT}/chexzero.csv (ensemble of {len(paths)}) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
