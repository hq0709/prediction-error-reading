"""Multi-label radiograph reader: one prompt per image, the reader answers yes or no for all 13 findings in a fixed order
(LoRA on language + vision, loss on the 13 answer tokens; uncertain labels carry no loss). One prompt per image
instead of one per finding makes training on every MIMIC training image affordable.

Designs:
  global   the current frontal radiograph;
  prior    the patient's previous frontal radiograph and the current one (the history as the world model's context);
           studies without a previous radiograph get the current image only and say so;
  wm       the radiograph with the world model's six addresses circled and numbered, and each address enlarged from
           the original full-resolution image (perr.cxr_reader.render_wm, the single-finding reader's evidence view);
  grid     control: the same layout at six fixed lung-field places (enlargement without the world model).
           wm / grid inputs are pre-rendered by make_cxr_ml_views.py into data/cxr_ml_<design>/<key>.png.
  focal    the radiograph left untouched, and a second image tiling six of its places enlarged at ~native detail:
           the world model's top focal addresses in the thorax (make_cxr_crops.py focal, first six crops, 160 px each);
  focal_random  control: the same with random thorax points (make_cxr_crops.py focal_random).

Scores: P(yes) at each answer position, read in two passes (answers filled greedily, then re-read) so that each
finding is conditioned on the reader's own previous answers, as in decoding. Writes results/sota_cxr/<name>.csv
(key + one column per finding), scored with sota_cxr/score.py like the baselines.

    CUDA_VISIBLE_DEVICES=0 python train_cxr_ml.py --design prior --out runs_reader/cxr_ml_prior
"""
import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from perr.cxr import CACHE, FINDINGS

Q8 = "/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
LABELS = [f for f in FINDINGS if f != "No Finding"]
SHORT = {"Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Consolidation": "consolidation",
         "Edema": "pulmonary edema", "Enlarged Cardiomediastinum": "enlarged cardiomediastinum", "Fracture": "fracture",
         "Lung Lesion": "lung lesion", "Lung Opacity": "lung opacity", "Pleural Effusion": "pleural effusion",
         "Pleural Other": "other pleural abnormality", "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax",
         "Support Devices": "support device"}
Q = {"global": "Frontal chest radiograph. For each finding below, answer yes or no.",
     "prior": "The first image is the patient's previous frontal chest radiograph, the second the current one. "
              "For the current radiograph, answer yes or no for each finding below.",
     "prior_none": "Frontal chest radiograph (no previous radiograph available). For each finding below, answer yes or no.",
     "places": "Frontal chest radiograph (left; numbered circles mark six places of interest) and those places enlarged "
               "(right). For each finding below, answer yes or no.",
     "tiles": "The first image is a frontal chest radiograph; the second shows six places of it enlarged at full "
              "resolution. For each finding below, answer yes or no."}
CROPS = "/raid/home/CAMCA/hj880/perr/cache/cxr_crops_{}"


def tiles(key, design, n=6, side=160):
    """Six enlarged places as one 2 x 3 image (missing crops stay black)."""
    p = f"{CROPS.format(design)}/{key}.npy"
    a = np.load(p)[:n] if os.path.exists(p) else np.zeros((n, side, side), np.uint8)
    a = [np.asarray(Image.fromarray(x).resize((side, side), Image.BILINEAR)) for x in a]
    a += [np.zeros((side, side), np.uint8)] * (n - len(a))
    return Image.fromarray(np.concatenate([np.concatenate(a[:3], 1), np.concatenate(a[3:6], 1)], 0)).convert("RGB")
VIEWS = "/raid/home/CAMCA/hj880/perr/data/cxr_ml_{}"


def images_of(row, design):
    if design in ("focal", "focal_random"):
        cur = Image.fromarray(np.load(f"{CACHE}/{row['key']}.npy")).convert("RGB")
        return [cur, tiles(row["key"], design)], Q["tiles"]
    if design in ("wm", "grid"):                                    # same question for both choices of places
        return [Image.open(f"{VIEWS.format(design)}/{row['key'].replace('/', '_')}.png").convert("RGB")], Q["places"]
    cur = Image.fromarray(np.load(f"{CACHE}/{row['key']}.npy")).convert("RGB")
    if design == "prior" and isinstance(row.get("prior_key"), str):
        return [Image.fromarray(np.load(f"{CACHE}/{row['prior_key']}.npy")).convert("RGB"), cur], Q["prior"]
    return [cur], Q["prior_none"] if design == "prior" else Q["global"]


class Codec:
    """Builds (text with answers, answer-token positions) and finds those positions after tokenization."""

    def __init__(self, proc):
        self.proc, self.tok = proc, proc.tokenizer
        self.yes = self.tok.encode(" yes", add_special_tokens=False)
        self.no = self.tok.encode(" no", add_special_tokens=False)
        assert len(self.yes) == 1 and len(self.no) == 1, "answers must be single tokens"
        self.yes, self.no = self.yes[0], self.no[0]
        self.lines = [self.tok.encode(f"{SHORT[f]}:", add_special_tokens=False) for f in LABELS]
        self.nl = self.tok.encode("\n", add_special_tokens=False)

    def prompt(self, n_images, question):
        content = [{"type": "image"}] * n_images + [{"type": "text", "text": question + "\n" +
                                                    "\n".join(f"{SHORT[f]}: yes or no?" for f in LABELS)}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    def answer_ids(self, answers):
        """Token ids of the answer block and the offsets of the 13 answer tokens within it."""
        ids, pos = [], []
        for line, a in zip(self.lines, answers):
            ids += line
            pos.append(len(ids))
            ids.append(self.yes if a else self.no)
            ids += self.nl
        return ids, pos


def batch_inputs(codec, rows, design, answers):
    """Tokenised prompts (processor, left-padded) with the answer block appended; returns inputs and answer positions."""
    texts, imgs = [], []
    for r in rows:
        im, q = images_of(r, design)
        texts.append(codec.prompt(len(im), q))
        imgs.append(im)
    enc = codec.proc(text=texts, images=[i for im in imgs for i in im], return_tensors="pt", padding=True)
    blocks = [codec.answer_ids(a) for a in answers]
    add = torch.tensor([b[0] for b in blocks])                      # same length for every row
    n = enc["input_ids"].shape[1]
    enc["input_ids"] = torch.cat([enc["input_ids"], add], 1)
    enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(add)], 1)
    for k in ("mm_token_type_ids", "token_type_ids"):              # per-token side tensors: text tokens are type 0
        if k in enc and enc[k].shape[1] == n:
            enc[k] = torch.cat([enc[k], torch.zeros_like(add)], 1)
    L = enc["input_ids"].shape[1] - add.shape[1]
    pos = torch.tensor([[L + p for p in b[1]] for b in blocks])
    return enc, pos


@torch.no_grad()
def score(model, codec, rows, design, bs=16):
    """P(yes) per finding: pass 1 with every answer 'no', fill greedy answers, pass 2 re-reads with them."""
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ans = [[0] * len(LABELS) for _ in chunk]
        for _ in range(2):
            enc, pos = batch_inputs(codec, chunk, design, ans)
            enc = enc.to(model.device)
            logits = model(**enc).logits
            lg = logits[torch.arange(len(chunk))[:, None], pos.to(model.device) - 1]          # predicts the answer token
            p = torch.stack([lg[..., codec.yes], lg[..., codec.no]], -1).float().softmax(-1)[..., 0]
            ans = (p > 0.5).int().tolist()
        out += p.tolist()
    model.train()
    return np.array(out)


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y)
    ok = ~np.isnan(y)
    p, y = p[ok], y[ok]
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def table(design):
    """Frontal images with labels (1 / 0 / -1 kept; not mentioned -> 0) and the previous frontal study's image."""
    from perr.cxr import mimic_frontal
    m = mimic_frontal()
    m["key"] = "mimic/" + m.dicom_id
    m["t"] = m.StudyDate.astype(int) * 1_000_000 + m.StudyTime.astype(float).astype(int)
    first = m.sort_values(["subject_id", "t"]).groupby("study_id").head(1)[["subject_id", "study_id", "t", "key"]]
    first = first.sort_values(["subject_id", "t"])
    first["prior_key"] = first.groupby("subject_id").key.shift(1)
    m = m.merge(first[["study_id", "prior_key"]], on="study_id", how="left")
    for f in LABELS:
        m[f] = m[f].fillna(0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["global", "prior", "wm", "grid", "focal", "focal_random"], default="global")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=0, help="cap the number of training images (0 = all)")
    ap.add_argument("--val-n", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=6000)
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    m = table(a.design)
    m = m[[os.path.exists(f"{CACHE}/{k}.npy") for k in m.key]]
    if a.design == "prior":                                      # a prior that is not cached is treated as absent
        m.loc[[not (isinstance(k, str) and os.path.exists(f"{CACHE}/{k}.npy")) for k in m.prior_key], "prior_key"] = None
    tr = m[m.split == "train"].to_dict("records")
    va = m[m.split == "validate"].sample(n=min(a.val_n, (m.split == "validate").sum()), random_state=0).to_dict("records")
    te = m[m.split == "test"].to_dict("records")
    random.shuffle(tr)
    if a.max_train:
        tr = tr[:a.max_train]
    print(f"{len(tr)} training images, {len(va)} validation, {len(te)} test; with a prior: "
          f"{np.mean([isinstance(r.get('prior_key'), str) for r in tr]):.2f}", flush=True)
    proc = AutoProcessor.from_pretrained(Q8)
    proc.tokenizer.padding_side = "left"
    codec = Codec(proc)
    model = AutoModelForImageTextToText.from_pretrained(Q8, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = (r"(.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))|"
               r"(.*visual.*\.(qkv|proj|linear_fc1|linear_fc2))")
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=targets))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(tr) / a.bs) * a.epochs
    sched = lambda g: a.lr * min(1.0, (g + 1) / 100) * 0.5 * (1 + math.cos(math.pi * g / steps))
    best, log, g = -1.0, [], 0

    def validate():
        p = score(model, codec, va, a.design)
        aucs = [auroc(p[:, j], [r[f] if r[f] != -1 else np.nan for r in va]) for j, f in enumerate(LABELS)]
        return float(np.nanmean(aucs))

    v0 = validate()
    print(f"step 0 validation mean AUROC {v0:.3f}", flush=True)
    for ep in range(a.epochs):
        random.shuffle(tr)
        tot = 0.0
        for st in range(0, len(tr), a.bs):
            chunk = tr[st:st + a.bs]
            ans = [[int(r[f] == 1) for f in LABELS] for r in chunk]
            keep = torch.tensor([[r[f] != -1 for f in LABELS] for r in chunk])
            enc, pos = batch_inputs(codec, chunk, a.design, ans)
            enc = enc.to(model.device)
            logits = model(**enc).logits
            lg = logits[torch.arange(len(chunk))[:, None], pos.to(model.device) - 1].float()
            tgt = torch.tensor(ans, device=model.device)
            two = torch.stack([lg[..., codec.no], lg[..., codec.yes]], -1)                 # class 1 = yes
            ce = torch.nn.functional.cross_entropy(two.view(-1, 2), tgt.view(-1), reduction="none").view(tgt.shape)
            loss = (ce * keep.to(ce.device)).sum() / keep.sum().clamp_min(1)
            for grp in opt.param_groups:
                grp["lr"] = sched(g)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            tot += loss.item(); g += 1
            if g % 100 == 0:
                print(f"step {g}/{steps} loss {tot / 100:.4f}", flush=True)
                tot = 0.0
            if g % a.eval_every == 0 or g == steps:
                v = validate()
                log.append({"step": g, "val_mean_auroc": v})
                print(f"step {g} validation mean AUROC {v:.3f}", flush=True)
                if v > best:
                    best = v
                    model.save_pretrained(f"{a.out}/lora_best")
                    json.dump({"step": g, "val_mean_auroc": v}, open(f"{a.out}/best.json", "w"))
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    # test with the best adapter
    from peft import PeftModel
    del model
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(Q8, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, f"{a.out}/lora_best").eval()
    p = score(model, codec, te, a.design)
    os.makedirs("results/sota_cxr", exist_ok=True)
    name = os.path.basename(a.out.rstrip("/"))
    pd.DataFrame(p, columns=LABELS).assign(key=[r["key"] for r in te])[["key"] + LABELS].to_csv(f"results/sota_cxr/{name}.csv", index=False)
    print(f"best validation mean AUROC {best:.3f}; test scores in results/sota_cxr/{name}.csv", flush=True)


if __name__ == "__main__":
    main()
