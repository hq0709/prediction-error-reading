"""Focal radiograph reader: verifies each candidate view of cxr_focal_data.py (context crop with the place outlined, and
the place at ~native detail) and answers yes or no for nodule, mass, calcified granuloma and fracture at that place.
Qwen3-VL-8B with LoRA on language + vision (the lung CT reader's recipe), loss on the four answer tokens.

Training rows: ChestX-Det train images minus a held-out tenth (validation, by image): every box-centre row, every
world-model address that lies in a box, and ten other world-model addresses per image per epoch. Validation: mean
address-level AUROC over the four findings on all addresses of the held-out images; the best epoch's adapter scores
every address of the evaluation sets (data/cxr_focal/<proposer>/<set>.jsonl) into results/cxr_focal/<name>/<set>.jsonl.

    CUDA_VISIBLE_DEVICES=0 python train_cxr_focal_reader.py --out runs_reader/cxr_focal
    CUDA_VISIBLE_DEVICES=0 python train_cxr_focal_reader.py --out runs_reader/cxr_focal --score-only wm/nih_test wm/mimic_test
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

Q8 = "/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DATA = "/raid/home/CAMCA/hj880/perr/data/cxr_focal"
TYPES = ("Nodule", "Mass", "Calcification", "Fracture")
SHORT = {"Nodule": "nodule", "Mass": "mass", "Calcification": "calcified granuloma", "Fracture": "fracture"}
QUESTION = ("Left: part of a frontal chest radiograph with one place outlined in yellow. Right: the outlined place "
            "enlarged. At the outlined place, answer yes or no for each finding below.")


def load(path):
    return [json.loads(l) for l in open(path)]


class Codec:
    def __init__(self, proc):
        self.proc, self.tok = proc, proc.tokenizer
        self.yes = self.tok.encode(" yes", add_special_tokens=False)[0]
        self.no = self.tok.encode(" no", add_special_tokens=False)[0]
        self.lines = [self.tok.encode(f"{SHORT[t]}:", add_special_tokens=False) for t in TYPES]
        self.nl = self.tok.encode("\n", add_special_tokens=False)
        content = [{"type": "image"}, {"type": "text", "text": QUESTION + "\n" + "\n".join(f"{SHORT[t]}: yes or no?" for t in TYPES)}]
        self.text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    def answer_ids(self, answers):
        ids, pos = [], []
        for line, a in zip(self.lines, answers):
            ids += line
            pos.append(len(ids))
            ids.append(self.yes if a else self.no)
            ids += self.nl
        return ids, pos

    def batch(self, rows, answers):
        enc = self.proc(text=[self.text] * len(rows), images=[Image.open(r["view"]).convert("RGB") for r in rows],
                        return_tensors="pt", padding=True)
        blocks = [self.answer_ids(a) for a in answers]
        add = torch.tensor([b[0] for b in blocks])
        n = enc["input_ids"].shape[1]
        enc["input_ids"] = torch.cat([enc["input_ids"], add], 1)
        enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(add)], 1)
        for k in ("mm_token_type_ids", "token_type_ids"):
            if k in enc and enc[k].shape[1] == n:
                enc[k] = torch.cat([enc[k], torch.zeros_like(add)], 1)
        pos = torch.tensor([[n + p for p in b[1]] for b in blocks])
        return enc, pos


@torch.no_grad()
def score(model, codec, rows, bs=32):
    """P(yes) per finding, two passes (answers filled greedily, then re-read), as train_cxr_ml.score."""
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ans = [[0] * len(TYPES) for _ in chunk]
        for _ in range(2):
            enc, pos = codec.batch(chunk, ans)
            enc = enc.to(model.device)
            lg = model(**enc).logits[torch.arange(len(chunk))[:, None], pos.to(model.device) - 1]
            p = torch.stack([lg[..., codec.yes], lg[..., codec.no]], -1).float().softmax(-1)[..., 0]
            ans = (p > 0.5).int().tolist()
        out += p.tolist()
    model.train()
    return np.array(out)


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def write_scores(model, codec, name, sets):
    for s in sets:
        rows = load(f"{DATA}/{s}.jsonl")
        p = score(model, codec, rows)
        os.makedirs(f"results/cxr_focal/{name}", exist_ok=True)
        with open(f"results/cxr_focal/{name}/{s.replace('/', '_')}.jsonl", "w") as f:
            for r, q in zip(rows, p):
                f.write(json.dumps({**{k: r[k] for k in ("key", "i", "r", "c", "wm", "src", "rank") + TYPES},
                                    **{f"p_{t}": float(v) for t, v in zip(TYPES, q)}}) + "\n")
        print(f"scored {s}: {len(rows)} addresses", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--neg", type=int, default=10)
    ap.add_argument("--score-only", nargs="*", default=None)
    ap.add_argument("--sets", nargs="*", default=["wm/cxdet_test"])
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    name = os.path.basename(a.out.rstrip("/"))
    proc = AutoProcessor.from_pretrained(Q8)
    proc.tokenizer.padding_side = "left"
    codec = Codec(proc)
    base = AutoModelForImageTextToText.from_pretrained(Q8, dtype=torch.bfloat16, device_map={"": 0})
    if a.score_only is not None:
        model = PeftModel.from_pretrained(base, f"{a.out}/lora_best").eval()
        return write_scores(model, codec, name, a.score_only)

    rows = load(f"{DATA}/wm/cxdet_train.jsonl")
    keys = sorted({r["key"] for r in rows})
    random.Random(0).shuffle(keys)
    val_keys = set(keys[:len(keys) // 10])
    tr = [r for r in rows if r["key"] not in val_keys]
    va = [r for r in rows if r["key"] in val_keys and r["src"] == "wm"]
    pos_tr = [r for r in tr if r["src"] == "gt" or any(r[t] for t in TYPES)]
    neg_by = {}
    for r in tr:
        if r["src"] == "wm" and not any(r[t] for t in TYPES):
            neg_by.setdefault(r["key"], []).append(r)
    per_epoch = len(pos_tr) + sum(min(a.neg, len(v)) for v in neg_by.values())
    print(f"train images {len(keys) - len(val_keys)}, positive rows {len(pos_tr)}, ~{per_epoch} rows per epoch; "
          f"validation {len(val_keys)} images, {len(va)} addresses", flush=True)

    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    targets = (r"(.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))|"
               r"(.*visual.*\.(qkv|proj|linear_fc1|linear_fc2))")
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                            target_modules=targets))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(per_epoch / a.bs) * a.epochs
    sched = lambda g: a.lr * min(1.0, (g + 1) / 100) * 0.5 * (1 + math.cos(math.pi * g / steps))  # noqa: E731

    def validate():
        p = score(model, codec, va)
        return float(np.nanmean([auroc(p[:, j], [r[t] for r in va]) for j, t in enumerate(TYPES)]))

    best, g, log = -1.0, 0, []
    for ep in range(a.epochs):
        epoch = pos_tr + [r for v in neg_by.values() for r in random.sample(v, min(a.neg, len(v)))]
        random.shuffle(epoch)
        tot = 0.0
        for st in range(0, len(epoch), a.bs):
            chunk = epoch[st:st + a.bs]
            ans = [[r[t] for t in TYPES] for r in chunk]
            enc, pos = codec.batch(chunk, ans)
            enc = enc.to(model.device)
            lg = model(**enc).logits[torch.arange(len(chunk))[:, None], pos.to(model.device) - 1].float()
            tgt = torch.tensor(ans, device=model.device)
            two = torch.stack([lg[..., codec.no], lg[..., codec.yes]], -1)
            loss = torch.nn.functional.cross_entropy(two.view(-1, 2), tgt.view(-1))
            for grp in opt.param_groups:
                grp["lr"] = sched(g)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            tot += loss.item(); g += 1
            if g % 100 == 0:
                print(f"step {g}/{steps} loss {tot / 100:.4f}", flush=True)
                tot = 0.0
        v = validate()
        log.append({"epoch": ep, "val_mean_auroc": v})
        print(f"epoch {ep} validation mean address AUROC {v:.3f}", flush=True)
        if v > best:
            best = v
            model.save_pretrained(f"{a.out}/lora_best")
            json.dump({"epoch": ep, "val_mean_auroc": v}, open(f"{a.out}/best.json", "w"))
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    del model
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(Q8, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, f"{a.out}/lora_best").eval()
    write_scores(model, codec, name, a.sets)


if __name__ == "__main__":
    main()
