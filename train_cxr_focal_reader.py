"""Focal radiograph reader: verifies each candidate view of cxr_focal_data.py (context crop with the place outlined, and
the place at ~native detail) and answers yes or no for nodule, mass, calcified granuloma and fracture at that place.
Qwen3-VL-8B with LoRA on language + vision (the lung CT reader's recipe), loss on the four answer tokens.

Training rows: ChestX-Det train images minus a held-out tenth (validation, by image): every box-centre row, every
world-model address that lies in a box, and ten other world-model addresses per image per epoch. With --mimic-rows, also
MIMIC training images with report labels only (multiple-instance rows): on each image, the address where a previous
reader's P(nodule or mass) is highest carries the image's "Lung Lesion" label and the address where its P(fracture) is
highest carries the "Fracture" label; only those answers are trained (the others are filled with the previous reader's
answers and carry no loss), uncertain labels are skipped. With --mil-silence skip, a finding the report does not mention
gives no row (only stated findings and stated absences do), and with --mil-gate g a stated finding gives a row only where
the previous reader's probability there is at least g. Validation: mean
address-level AUROC over the four findings on all addresses of the held-out images; the best epoch's adapter scores
every address of the evaluation sets (data/cxr_focal/<proposer>/<set>.jsonl) into results/cxr_focal/<name>/<set>.jsonl.

    CUDA_VISIBLE_DEVICES=0 python train_cxr_focal_reader.py --out runs_reader/cxr_focal
    CUDA_VISIBLE_DEVICES=0 python train_cxr_focal_reader.py --out runs_reader/cxr_focal --score-only wm/nih_test wm/mimic_test
    CUDA_VISIBLE_DEVICES=0 python train_cxr_focal_reader.py --out runs_reader/cxr_focal_mil \
        --mimic-rows results/cxr_focal/cxr_focal/wm_mimic_train.jsonl
"""
import argparse
import gc
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


def mimic_rows(path, silence="negative", gate=0.0):
    """Multiple-instance rows from MIMIC report labels (see the module docstring); path: a reader's scored rows."""
    from perr.cxr import mimic_frontal
    m = mimic_frontal()
    m["key"] = "mimic/" + m.dicom_id
    lab = m.set_index("key")[["Lung Lesion", "Fracture"]]
    views = {(r["key"], r["i"]): r["view"] for r in load(f"{DATA}/wm/mimic_train.jsonl")}
    by = {}
    for r in load(path):
        by.setdefault(r["key"], []).append(r)
    out = []
    for k, rs in by.items():
        for finding, idx, score in (("Lung Lesion", (0, 1), lambda r: max(r["p_Nodule"], r["p_Mass"])),
                                    ("Fracture", (3,), lambda r: r["p_Fracture"])):
            y = lab.loc[k, finding]
            if y == -1 or (silence == "skip" and np.isnan(y)):
                continue
            r = max(rs, key=score)
            if y == 1 and score(r) < gate:
                continue
            prev = [int(r[f"p_{t}"] > 0.5) for t in TYPES]
            ans, mask = list(prev), [0] * len(TYPES)
            if y == 1:      # the more likely of nodule / mass at the chosen place
                j = idx[int(np.argmax([r[f"p_{TYPES[i]}"] for i in idx]))]
                ans[j], mask[j] = 1, 1
            else:
                for j in idx:
                    ans[j], mask[j] = 0, 1
            out.append({"key": k, "i": r["i"], "view": views[(k, r["i"])], "src": "mimic", "y": ans, "m": mask})
    return out


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
    ap.add_argument("--micro", type=int, default=0, help="micro-batch size (gradient accumulation to --bs); 0: --bs")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--neg", type=int, default=10)
    ap.add_argument("--score-only", nargs="*", default=None)
    ap.add_argument("--sets", nargs="*", default=["wm/cxdet_test"])
    ap.add_argument("--train-keys", default=None, help="only these training images (few-box settings)")
    ap.add_argument("--mimic-rows", default=None, help="a reader's scored MIMIC training addresses (report-label rows)")
    ap.add_argument("--mil-silence", default="negative", choices=("negative", "skip"),
                    help="a finding the report does not mention: a negative row, or no row")
    ap.add_argument("--mil-gate", type=float, default=0.0, help="a stated finding gives a row only where the previous "
                    "reader's probability is at least this")
    ap.add_argument("--proposer", default="wm", help="address source of the ChestX-Det training rows (wm, wmu)")
    ap.add_argument("--hard-from", default=None, help="a reader's scored ChestX-Det training addresses (hard negatives)")
    ap.add_argument("--hard", type=int, default=0, help="per image and epoch, this many of the --neg negatives are the "
                                                         "addresses that reader scored highest")
    ap.add_argument("--pos-share", type=float, default=0.0, help="class-balanced epochs: positive rows are repeated until "
                    "they make up at least this share of each epoch (0.16 = their share with every box; a no-op there)")
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

    rows = load(f"{DATA}/{a.proposer}/cxdet_train.jsonl")
    keys = sorted({r["key"] for r in rows})
    random.Random(0).shuffle(keys)
    val_keys = set(keys[:len(keys) // 10])
    tr = [r for r in rows if r["key"] not in val_keys]
    if a.train_keys:
        allowed = set(open(a.train_keys).read().split())
        tr = [r for r in tr if r["key"] in allowed]
    va = [r for r in rows if r["key"] in val_keys and r["src"] == "wm"]
    pos_tr = [r for r in tr if r["src"] == "gt" or any(r[t] for t in TYPES)]
    neg_by = {}
    for r in tr:
        if r["src"] == "wm" and not any(r[t] for t in TYPES):
            neg_by.setdefault(r["key"], []).append(r)
    for r in tr:
        r["y"], r["m"] = [r[t] for t in TYPES], [1] * len(TYPES)
    hard_by = {}
    if a.hard_from:     # negatives ranked by the previous reader's highest answer: the places that win the max falsely
        prev = {(r["key"], r["i"]): max(r[f"p_{t}"] for t in TYPES) for r in load(a.hard_from)}
        for k, v in neg_by.items():
            v = sorted(v, key=lambda r: -prev.get((r["key"], r["i"]), 0.0))
            hard_by[k], neg_by[k] = v[:a.hard], v[a.hard:]
    mil = mimic_rows(a.mimic_rows, a.mil_silence, a.mil_gate) if a.mimic_rows else []
    n_neg = sum(min(a.neg, len(v) + len(hard_by.get(k, []))) for k, v in neg_by.items()) + len(mil)
    n_pos = max(len(pos_tr), math.ceil(a.pos_share * n_neg / (1 - a.pos_share))) if pos_tr else 0
    per_epoch = n_pos + n_neg
    print(f"train images {len(keys) - len(val_keys)}, positive rows {len(pos_tr)} ({n_pos} per epoch), "
          f"MIMIC report-label rows {len(mil)} "
          f"({sum(1 for r in mil if any(y and m for y, m in zip(r['y'], r['m'])))} positive), ~{per_epoch} rows per epoch; "
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
        reps = [pos_tr[i % len(pos_tr)] for i in random.sample(range(max(n_pos, len(pos_tr))), n_pos)] if pos_tr else []
        epoch = reps + mil + [r for k, v in neg_by.items()
                                for r in hard_by.get(k, []) + random.sample(v, max(0, min(a.neg - len(hard_by.get(k, [])), len(v))))]
        random.shuffle(epoch)
        tot = 0.0
        for st in range(0, len(epoch), a.bs):
            chunk = epoch[st:st + a.bs]
            n_ans = float(sum(sum(r["m"]) for r in chunk))
            loss = 0.0
            for ms in range(0, len(chunk), a.micro or a.bs):        # micro-batches: the same loss and step, less memory
                sub = chunk[ms:ms + (a.micro or a.bs)]
                ans = [r["y"] for r in sub]
                enc, pos = codec.batch(sub, ans)
                enc = enc.to(model.device)
                lg = model(**enc).logits[torch.arange(len(sub))[:, None], pos.to(model.device) - 1].float()
                tgt = torch.tensor(ans, device=model.device)
                msk = torch.tensor([r["m"] for r in sub], device=model.device, dtype=torch.float32).view(-1)
                two = torch.stack([lg[..., codec.no], lg[..., codec.yes]], -1)
                ce = torch.nn.functional.cross_entropy(two.view(-1, 2), tgt.view(-1), reduction="none")
                part = (ce * msk).sum() / n_ans
                part.backward()
                loss += part.detach()
            for grp in opt.param_groups:
                grp["lr"] = sched(g)
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
    del model, base, opt          # the trained copy is dropped before the best adapter is loaded onto a fresh base
    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(Q8, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, f"{a.out}/lora_best").eval()
    write_scores(model, codec, name, a.sets)


if __name__ == "__main__":
    main()
