"""The trained one-shot radiograph reader (train_cxr_oneshot.py) on the CheXlocalize radiographs (build_chexloc.py),
unchanged: P(yes) per trained finding and the pointed place, in the format of results/cxr_focal/<run>/cxdet_test.jsonl.

    CUDA_VISIBLE_DEVICES=4 python score_oneshot_chexloc.py runs_reader/cxr_oneshott
Writes results/cxr_focal/<run>/chexloc.jsonl.
"""
import json
import os
import sys

from peft import PeftModel
from transformers import AutoProcessor

from train_cxr_focal_reader import TYPES, load_base
from train_cxr_oneshot import Codec, backbone, locate, presence


def main():
    out = sys.argv[1].rstrip("/")
    ckpt, four_bit = backbone(out)
    proc = AutoProcessor.from_pretrained(ckpt)
    proc.tokenizer.padding_side = "left"
    codec = Codec(proc)
    model = PeftModel.from_pretrained(load_base(ckpt, four_bit), f"{out}/lora_best").eval()
    keys = open("results/chexloc_keys.txt").read().split()
    p = presence(model, codec, keys)
    loc = locate(model, codec, keys)
    run = os.path.basename(out)
    os.makedirs(f"results/cxr_focal/{run}", exist_ok=True)
    with open(f"results/cxr_focal/{run}/chexloc.jsonl", "w") as f:
        for k, q in zip(keys, p):
            for j, t in enumerate(TYPES):
                x, y = loc[(k, t)]
                f.write(json.dumps({"key": k, "i": j, "r": y, "c": x, "wm": 1.0, "src": "wm", "rank": 0,
                                    **{f"p_{u}": (float(q[j]) if u == t else 0.0) for u in TYPES}}) + "\n")
    print("scored chexloc:", len(keys), "images")


if __name__ == "__main__":
    main()
