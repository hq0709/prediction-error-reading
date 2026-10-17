"""One-shot radiograph reader, the comparison for the focal reader (train_cxr_focal_reader.py): the same Qwen3-VL-8B
LoRA recipe reads the whole radiograph (512 canvas, perr/cxr.py) and (a) answers yes or no for nodule, mass, calcified
granuloma and fracture, (b) points at the finding when asked ("<finding>: (x, y)", 0-1000 across the image; the target
is the centre of the finding's largest box). Same ChestX-Det training images and held-out tenth as the focal reader.

Evaluation rows are written in the focal reader's format (one "address" per image and finding: the pointed place, with
P(yes) for that finding), so score_cxr_focal.py scores both alike.

    CUDA_VISIBLE_DEVICES=1 python train_cxr_oneshot.py --out runs_reader/cxr_oneshot
"""
import argparse
import json
import math
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from cxr_recall import boxes
from perr.cxr import CACHE
from train_cxr_focal_reader import Q8, SHORT, TYPES, auroc

ASK = "Frontal chest radiograph. For each finding below, answer yes or no."
POINT = "Frontal chest radiograph. Point at the most likely {} as (x, y), with x and y from 0 to 1000 across the image."


def image(key):
    return Image.fromarray(np.load(f"{CACHE}/{key}.npy")).convert("RGB")


class Codec:
    def __init__(self, proc):
        self.proc, self.tok = proc, proc.tokenizer
        self.yes = self.tok.encode(" yes", add_special_tokens=False)[0]
        self.no = self.tok.encode(" no", add_special_tokens=False)[0]
        self.lines = [self.tok.encode(f"{SHORT[t]}:", add_special_tokens=False) for t in TYPES]
        self.nl = self.tok.encode("\n", add_special_tokens=False)

    def prompt(self, q):
        content = [{"type": "image"}, {"type": "text", "text": q}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    def ask(self, keys, answers):
        q = ASK + "\n" + "\n".join(f"{SHORT[t]}: yes or no?" for t in TYPES)
        enc = self.proc(text=[self.prompt(q)] * len(keys), images=[image(k) for k in keys], return_tensors="pt", padding=True)
        ids, pos = [], []
        for a in answers:
            i, p = [], []
            for line, y in zip(self.lines, a):
                i += line
                p.append(len(i))
                i.append(self.yes if y else self.no)
                i += self.nl
            ids.append(i); pos.append(p)
        return self._append(enc, ids), pos

    def point(self, keys, types, targets=None):
        enc = self.proc(text=[self.prompt(POINT.format(SHORT[t])) for t in types], images=[image(k) for k in keys],
                        return_tensors="pt", padding=True)
        if targets is None:
            return enc
        ids = [self.tok.encode(f"({x}, {y})", add_special_tokens=False) + self.tok.encode(self.tok.eos_token or "", add_special_tokens=False)
               for x, y in targets]
        L = max(len(i) for i in ids)
        ids = [i + [self.tok.pad_token_id] * (L - len(i)) for i in ids]
        mask = [[1] * len(i0) + [0] * (L - len(i0)) for i0 in [self.tok.encode(f"({x}, {y})", add_special_tokens=False) for x, y in targets]]
        return self._append(enc, ids, mask)

    def _append(self, enc, ids, mask=None):
        add = torch.tensor(ids)
        n = enc["input_ids"].shape[1]
        enc["input_ids"] = torch.cat([enc["input_ids"], add], 1)
        enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(add)], 1)
        for k in ("mm_token_type_ids", "token_type_ids"):
            if k in enc and enc[k].shape[1] == n:
                enc[k] = torch.cat([enc[k], torch.zeros_like(add)], 1)
        enc["n_prompt"] = n
        enc["target_mask"] = torch.tensor(mask) if mask is not None else None
        return enc


def targets_of(b, geo_keys):
    """key -> {type: (x, y) in 0-1000 of the 512 canvas} from the largest box of each type."""
    out = {}
    for k in geo_keys:
        g = b[b.key == k]
        out[k] = {}
        for t in TYPES:
            bt = g[g.finding == t]
            if len(bt):
                r = bt.loc[(bt.w * bt.h).idxmax()]
                out[k][t] = (int(round((r.x + r.w / 2) / 512 * 1000)), int(round((r.y + r.h / 2) / 512 * 1000)))
    return out


def forward(model, enc):
    n, tm = enc.pop("n_prompt"), enc.pop("target_mask")
    return model(**enc.to(model.device)).logits, n, tm


@torch.no_grad()
def presence(model, codec, keys, bs=16):
    out = []
    for i in range(0, len(keys), bs):
        ks = keys[i:i + bs]
        ans = [[0] * len(TYPES) for _ in ks]
        for _ in range(2):
            enc, pos = codec.ask(ks, ans)
            lg, n, _ = forward(model, enc)
            pos = torch.tensor([[n + p for p in q] for q in pos], device=lg.device)
            lg = lg[torch.arange(len(ks))[:, None], pos - 1]
            p = torch.stack([lg[..., codec.yes], lg[..., codec.no]], -1).float().softmax(-1)[..., 0]
            ans = (p > 0.5).int().tolist()
        out += p.tolist()
    return np.array(out)


@torch.no_grad()
def locate(model, codec, keys, bs=16):
    """(x, y) on the 512 canvas for every key and finding, by greedy decoding of the pointing answer."""
    out = {}
    jobs = [(k, t) for k in keys for t in TYPES]
    for i in range(0, len(jobs), bs):
        ch = jobs[i:i + bs]
        enc = codec.point([k for k, _ in ch], [t for _, t in ch])
        g = model.generate(**enc.to(model.device), max_new_tokens=16, do_sample=False)
        txt = codec.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (k, t), s in zip(ch, txt):
            m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", s)
            out[(k, t)] = (float(m.group(1)) / 1000 * 512, float(m.group(2)) / 1000 * 512) if m else (256.0, 256.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    g = pd.read_csv("results/cxr_cxdet.csv")
    b = boxes("cxdet")
    train_keys = sorted(g[g.split == "train"].key)
    random.Random(0).shuffle(train_keys)                         # the focal reader's held-out tenth
    focal_rows = [json.loads(l) for l in open("data/cxr_focal/wm/cxdet_train.jsonl")]
    fkeys = sorted({r["key"] for r in focal_rows})
    random.Random(0).shuffle(fkeys)
    val_keys = fkeys[:len(fkeys) // 10]
    tr_keys = [k for k in train_keys if k not in set(val_keys)]
    test_keys = sorted(g[g.split == "test"].key)
    tg = targets_of(b, tr_keys + val_keys + test_keys)
    lab = {k: [int(t in tg[k]) for t in TYPES] for k in tg}
    print(f"train {len(tr_keys)}, validation {len(val_keys)}, test {len(test_keys)} images", flush=True)

    proc = AutoProcessor.from_pretrained(Q8)
    proc.tokenizer.padding_side = "left"
    codec = Codec(proc)
    base = AutoModelForImageTextToText.from_pretrained(Q8, dtype=torch.bfloat16, device_map={"": 0})
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    targets = (r"(.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))|"
               r"(.*visual.*\.(qkv|proj|linear_fc1|linear_fc2))")
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                            target_modules=targets))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    point_jobs = [(k, t) for k in tr_keys for t in TYPES if t in tg[k]]
    per_epoch = math.ceil(len(tr_keys) / a.bs) + math.ceil(len(point_jobs) / a.bs)
    steps = per_epoch * a.epochs
    sched = lambda s: a.lr * min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps))  # noqa: E731

    def validate():
        model.eval()
        p = presence(model, codec, val_keys)
        model.train()
        return float(np.nanmean([auroc(p[:, j], np.array([lab[k][j] for k in val_keys]).astype(bool)) for j in range(len(TYPES))]))

    best, step = -1.0, 0
    for ep in range(a.epochs):
        batches = [("ask", tr_keys[i:i + a.bs]) for i in range(0, len(tr_keys), a.bs)]
        random.shuffle(point_jobs)
        batches += [("point", point_jobs[i:i + a.bs]) for i in range(0, len(point_jobs), a.bs)]
        random.shuffle(batches)
        tot = 0.0
        for kind, ch in batches:
            if kind == "ask":
                ans = [lab[k] for k in ch]
                enc, pos = codec.ask(ch, ans)
                lg, n, _ = forward(model, enc)
                pos = torch.tensor([[n + p for p in q] for q in pos], device=lg.device)
                lg = lg[torch.arange(len(ch))[:, None], pos - 1].float()
                two = torch.stack([lg[..., codec.no], lg[..., codec.yes]], -1)
                loss = torch.nn.functional.cross_entropy(two.view(-1, 2), torch.tensor(ans, device=lg.device).view(-1))
            else:
                enc = codec.point([k for k, _ in ch], [t for _, t in ch], [tg[k][t] for k, t in ch])
                ids = enc["input_ids"].clone()
                lg, n, tm = forward(model, enc)
                tgt = ids[:, n:].to(lg.device)
                pred = lg[:, n - 1:-1].float()
                ce = torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), reduction="none")
                m = tm.to(lg.device).float().view(-1)
                loss = (ce * m).sum() / m.sum()
            for grp in opt.param_groups:
                grp["lr"] = sched(step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            tot += loss.item(); step += 1
            if step % 100 == 0:
                print(f"step {step}/{steps} loss {tot / 100:.4f}", flush=True)
                tot = 0.0
        v = validate()
        print(f"epoch {ep} validation mean presence AUROC {v:.3f}", flush=True)
        if v > best:
            best = v
            model.save_pretrained(f"{a.out}/lora_best")
    del model
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(Q8, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, f"{a.out}/lora_best").eval()
    for name, keys in (("cxdet_test", test_keys),):
        p = presence(model, codec, keys)
        loc = locate(model, codec, keys)
        os.makedirs("results/cxr_focal/cxr_oneshot", exist_ok=True)
        with open(f"results/cxr_focal/cxr_oneshot/{name}.jsonl", "w") as f:
            for k, q in zip(keys, p):
                for j, t in enumerate(TYPES):
                    x, y = loc[(k, t)]
                    row = {"key": k, "i": j, "r": y, "c": x, "wm": 1.0, "src": "wm", "rank": 0,
                           **{f"p_{u}": (float(q[j]) if u == t else 0.0) for u in TYPES}}
                    f.write(json.dumps(row) + "\n")
        print(f"scored {name}", flush=True)


if __name__ == "__main__":
    main()
