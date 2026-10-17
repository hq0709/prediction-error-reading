"""MedGemma 1.5 4B (unsloth/medgemma-1.5-4b-it, an ungated mirror of google/medgemma-1.5-4b-it), yes / no prompting per
finding, on the official MIMIC-CXR-JPG test split (frontal images).

Message format of the official quick-start notebook (non-thinking mode): system "You are an expert radiologist.",
user = [question text, image]; apply_chat_template(add_generation_prompt=True), bf16, default Gemma3 processor
(RGB, 896 x 896, mean / std 0.5).  Question:
    "Does this chest X-ray show {phrase}? Answer with only one word: yes or no."
Score = P(yes) at the first answer token, renormalised over the yes / no tokens:
    (p("Yes") + p("yes")) / (p("Yes") + p("yes") + p("No") + p("no")).
The format was checked on validate images only (with "Answer yes or no." the model began half of the answers with
"Based on ..."; with the one-word instruction the yes / no tokens carry > 98 % of the first-token mass).  The finding
phrases and the scoring rule were fixed before any run; nothing was tuned on test images.

Speed-up (default): the 13 prompts of one image share one pass of the vision tower.  The image features are scattered
into the token embeddings exactly as Gemma3Model.forward does (masked_scatter at the image-token positions), and
token_type_ids keep the bidirectional attention inside the image block.  --standard runs the plain
model(**processor_outputs) path instead; both paths agree on validate images (see NOTES.md).

    python run_medgemma.py [--split test] [--limit N] [--standard] [--part i/n]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import FINDINGS, OUT, split_table  # noqa: E402

MODEL = ("/raid/home/CAMCA/hj880/hf_home/hub/models--unsloth--medgemma-1.5-4b-it/snapshots/"
         "de0155051f541fb5016589feeec59e9cd078347d")
SYSTEM = "You are an expert radiologist."
QUESTION = "Does this chest X-ray show {}? Answer with only one word: yes or no."
PHRASE = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "pulmonary edema",
    "Enlarged Cardiomediastinum": "an enlarged cardiomediastinum",
    "Fracture": "a fracture",
    "Lung Lesion": "a lung lesion (nodule or mass)",
    "Lung Opacity": "a lung opacity",
    "Pleural Effusion": "a pleural effusion",
    "Pleural Other": "a pleural abnormality other than effusion (such as pleural thickening)",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "a pneumothorax",
    "Support Devices": "support devices (such as lines, tubes or pacemakers)",
}


def conversation(phrase, img):
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": QUESTION.format(phrase)},
                                         {"type": "image", "image": img}]}]


class Prompts(Dataset):
    """One item per image: the 13 yes / no prompts, tokenised with left padding.  Fast path: the text is built with a
    blank placeholder image (the image-token expansion does not depend on the pixels) and the real image is processed
    once; standard path: the processor handles the real image for every prompt."""

    def __init__(self, paths, proc, standard):
        self.paths, self.proc, self.standard = paths, proc, standard
        self.blank = Image.new("RGB", (896, 896))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        convs = [conversation(PHRASE[f], img if self.standard else self.blank) for f in FINDINGS]
        enc = dict(self.proc.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                 return_tensors="pt", padding=True))
        if not self.standard:
            enc["pixel_values"] = self.proc.image_processor(images=img, return_tensors="pt")["pixel_values"]
        return enc


def scores_fast(model, enc):
    m = model.model
    ids = enc["input_ids"]
    img_tok = model.config.image_token_id
    llm_ids = ids.clone()
    if img_tok >= m.vocab_size:
        llm_ids[ids == img_tok] = 0
    emb = m.get_input_embeddings()(llm_ids)
    feats = m.get_image_features(enc["pixel_values"].to(torch.bfloat16)).to(emb.dtype)        # (1, 256, d)
    mask = (ids == img_tok).unsqueeze(-1).expand_as(emb)
    emb = emb.masked_scatter(mask, feats.expand(ids.shape[0], -1, -1).contiguous())
    return model(inputs_embeds=emb, attention_mask=enc["attention_mask"], token_type_ids=enc["token_type_ids"],
                 use_cache=False, logits_to_keep=1).logits[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--standard", action="store_true")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--show", type=int, default=0, help="print the top next tokens for the first N images")
    a = ap.parse_args()
    t0 = time.time()
    df = split_table(a.split)
    if a.limit:
        df = df.head(a.limit)
    k, n = map(int, a.part.split("/"))
    df = df.iloc[k::n].reset_index(drop=True)
    proc = transformers.AutoProcessor.from_pretrained(MODEL, use_fast=True)
    proc.tokenizer.padding_side = "left"
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False) for w in ("Yes", "yes")]
    no = [tok.encode(w, add_special_tokens=False) for w in ("No", "no")]
    assert all(len(t) == 1 for t in yes + no), (yes, no)
    yes, no = [t[0] for t in yes], [t[0] for t in no]
    dl = DataLoader(Prompts(df.path.tolist(), proc, a.standard), batch_size=None, num_workers=12,
                    worker_init_fn=lambda _: torch.set_num_threads(1), prefetch_factor=4)
    scores, share = [], []
    with torch.inference_mode():
        for i, enc in enumerate(dl):
            enc = {key: v.to("cuda") for key, v in enc.items()}
            if a.standard:
                enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
                logits = model(**enc, logits_to_keep=1).logits[:, -1]
            else:
                logits = scores_fast(model, enc)
            p = logits.float().softmax(-1)
            py, pn = p[:, yes].sum(-1), p[:, no].sum(-1)
            scores.append((py / (py + pn)).cpu().numpy())
            share.append((py + pn).cpu().numpy())
            if i < a.show:
                for j, f in enumerate(FINDINGS):
                    v, t = p[j].topk(5)
                    print(f, [(tok.decode([x]), round(float(y), 3)) for x, y in zip(t.tolist(), v.tolist())])
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(df)} images, {time.time() - t0:.0f} s", flush=True)
    res = pd.DataFrame(np.stack(scores), columns=list(FINDINGS))
    res.insert(0, "key", df.key.values)
    share = np.stack(share)
    name = "medgemma" + ("" if a.split == "test" and not a.limit else f"_{a.split}{a.limit or ''}") \
        + ("_standard" if a.standard else "") + ("" if n == 1 else f"_part{k}of{n}")
    res.to_csv(f"{OUT}/{name}.csv", index=False)
    print("mass on yes/no tokens: median", float(np.median(share)), "min", float(share.min()),
          "share of prompts < 0.5:", float((share < 0.5).mean()))
    print(f"wrote {OUT}/{name}.csv ({len(res)} rows) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
