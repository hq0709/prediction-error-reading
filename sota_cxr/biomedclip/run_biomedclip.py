"""BiomedCLIP (Zhang et al. 2023; microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), zero-shot, on the
official MIMIC-CXR-JPG test split (frontal images).

Model card usage: open_clip create_model_from_pretrained / get_tokenizer from the hf-hub id, the model's own preprocess
(RGB, resize the shorter side to 224 bicubic, centre crop 224, CLIP mean / std), context_length 256,
logits = logit_scale * image_features @ text_features.T, softmax.  Prompting as for CheXzero: per finding the
positive / negative pair ("{label}", "no {label}") and p = softmax over the pair (positive entry).  The original JPEG
is the input.  Nothing was tuned; the prompt pair was fixed before the first run.

    python run_biomedclip.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import FINDINGS, OUT, split_table  # noqa: E402

HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
TEMPLATE = ("{}", "no {}")


class Test(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.paths[i]))      # open_clip's transform converts to RGB


def main():
    t0 = time.time()
    df = split_table("test")
    model, preprocess = create_model_from_pretrained(HUB)
    tokenizer = get_tokenizer(HUB)
    dev = torch.device("cuda")
    model = model.to(dev).eval()
    print(preprocess, flush=True)
    texts = [TEMPLATE[0].format(f) for f in FINDINGS] + [TEMPLATE[1].format(f) for f in FINDINGS]
    print("prompts:", texts, flush=True)
    tok = tokenizer(texts, context_length=256).to(dev)
    dl = DataLoader(Test(df.path.tolist(), preprocess), batch_size=64, num_workers=16,
                    worker_init_fn=lambda _: torch.set_num_threads(1))
    probs = []
    with torch.no_grad():
        tf = model.encode_text(tok, normalize=True)
        n = len(FINDINGS)
        for x in dl:
            imf = model.encode_image(x.to(dev), normalize=True)
            logits = model.logit_scale.exp() * imf @ tf.t()               # (b, 2n): positives then negatives
            pair = torch.stack([logits[:, :n], logits[:, n:]], dim=-1)    # (b, n, 2)
            probs.append(pair.softmax(dim=-1)[..., 0].float().cpu().numpy())
    res = pd.DataFrame(np.concatenate(probs), columns=list(FINDINGS))
    res.insert(0, "key", df.key.values)
    res.to_csv(f"{OUT}/biomedclip.csv", index=False)
    print(f"wrote {OUT}/biomedclip.csv ({len(res)} rows) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
