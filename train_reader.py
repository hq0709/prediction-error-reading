"""LoRA fine-tuning of a VLM reader to verify world-model candidates (one-word answer: yes / no).

    python train_reader.py --model <path> --out runs_reader/qwen3vl8b --epochs 3
The loss is the cross-entropy of the answer token only. Each epoch keeps every positive and samples negatives
at --neg-ratio per positive. The checkpoint with the best validation AUROC is kept (LoRA merged for vLLM).
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from perr.reader_prompt import prepare

VARIANT = "full"


def build(proc, row, answer=None):
    """(chat text ending with the answer if given, image) for one candidate row."""
    im, q = prepare(row, VARIANT)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt + (answer or ""), im


def auroc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


@torch.no_grad()
def score(model, proc, rows, yes_id, no_id, bs=16):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        texts, imgs = zip(*[build(proc, r) for r in chunk])
        enc = proc(text=list(texts), images=list(imgs), return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        two = torch.stack([logits[:, yes_id], logits[:, no_id]], -1).float().softmax(-1)[:, 0]
        out += two.tolist()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/raid/home/CAMCA/hj880/perr/data/reader")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--neg-ratio", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--vision", action="store_true", help="also adapt the vision tower")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop each epoch after this many steps")
    ap.add_argument("--variant", choices=["full", "axial", "nomeas"], default="full", help="what the reader is shown")
    ap.add_argument("--wm-only", action="store_true", help="train on world-model candidates only (no gt strips)")
    ap.add_argument("--device-map", default="single", help='"single" (one GPU) or "auto" (split a large model)')
    a = ap.parse_args()
    global VARIANT
    VARIANT = a.variant
    random.seed(0); torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    dmap = {"": 0} if a.device_map == "single" else a.device_map
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dmap)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    if a.vision:
        targets = f"({targets})|(.*visual.*\\.(qkv|proj|linear_fc1|linear_fc2))"
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    tr = [json.loads(l) for l in open(f"{a.data}/train.jsonl")]
    if a.wm_only:
        tr = [r for r in tr if r["source"] == "wm"]
    va = [json.loads(l) for l in open(f"{a.data}/val.jsonl")]
    yes_id = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = proc.tokenizer.encode("no", add_special_tokens=False)[0]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    pos = [r for r in tr if r["label"] == 1]
    neg = [r for r in tr if r["label"] == 0]
    best, log = -1.0, []
    def validate(ep, extra=None):
        p = score(model, proc, va, yes_id, no_id)
        y = [r["label"] for r in va]
        wm = [i for i, r in enumerate(va) if r["source"] == "wm"]
        auc, auc_wm = auroc(p, y), auroc([p[i] for i in wm], [y[i] for i in wm])
        print(f"epoch {ep} val AUROC {auc:.3f} (world-model candidates only {auc_wm:.3f})", flush=True)
        log.append({"epoch": ep, "val_auroc": auc, "val_auroc_wm": auc_wm, **(extra or {})})
        json.dump([{k: r[k] for k in ("uid", "peak", "wm", "label", "source")} | {"p": q} for r, q in zip(va, p)],
                  open(f"{a.out}/val_pred_epoch{ep}.json", "w"))
        return auc

    validate(0)
    n_neg = min(len(neg), int(len(pos) * a.neg_ratio))
    steps = math.ceil((len(pos) + n_neg) / a.bs)
    if a.max_steps:
        steps = min(steps, a.max_steps)
    total = steps * a.epochs        # one warm-up and one cosine decay over the whole run
    sched_lr = lambda g: a.lr * min(1.0, (g + 1) / 20) * (0.5 * (1 + math.cos(math.pi * g / total)))
    print(f"{len(pos)} positives, {n_neg} of {len(neg)} negatives per epoch; {steps} steps per epoch", flush=True)
    for ep in range(1, a.epochs + 1):
        batch_rows = pos + random.sample(neg, n_neg)
        random.shuffle(batch_rows)
        tot = 0.0
        for st in range(steps):
            chunk = batch_rows[st * a.bs:(st + 1) * a.bs]
            texts, imgs = zip(*[build(proc, r, "yes" if r["label"] else "no") for r in chunk])
            enc = proc(text=list(texts), images=list(imgs), return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits[:, -2, :]          # position that predicts the answer token
            target = torch.tensor([yes_id if r["label"] else no_id for r in chunk], device=logits.device)
            assert (enc["input_ids"][:, -1] == target).all(), "answer must be the last token"
            loss = torch.nn.functional.cross_entropy(logits.float(), target)
            for g in opt.param_groups:
                g["lr"] = sched_lr((ep - 1) * steps + st)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            tot += loss.item()
            if st % 50 == 0:
                print(f"epoch {ep} step {st}/{steps} loss {tot / (st + 1):.4f}", flush=True)
        auc = validate(ep, {"train_loss": tot / steps})
        if auc > best:
            best = auc
            model.save_pretrained(f"{a.out}/lora_best")
            json.dump({"epoch": ep, "val_auroc": auc}, open(f"{a.out}/best.json", "w"))
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    # merge the best adapter for vLLM serving
    from peft import PeftModel
    del model
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dmap)
    merged = PeftModel.from_pretrained(base, f"{a.out}/lora_best").merge_and_unload()
    merged.save_pretrained(f"{a.out}/merged")
    proc.save_pretrained(f"{a.out}/merged")
    print(f"best val AUROC {best:.3f}; merged model at {a.out}/merged", flush=True)


if __name__ == "__main__":
    main()
