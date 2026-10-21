"""One-shot radiograph reader, the comparison for the focal reader (train_cxr_focal_reader.py): the same Qwen3-VL-8B
LoRA recipe reads the whole radiograph (512 canvas, perr/cxr.py) and (a) answers yes or no for nodule, mass, calcified
granuloma and fracture, (b) points at the finding when asked ("<finding>: (x, y)", 0-1000 across the image; the target
is the centre of the finding's largest box). Same ChestX-Det training images and held-out tenth as the focal reader.

Evaluation rows are written in the focal reader's format (one "address" per image and finding: the pointed place, with
P(yes) for that finding), so score_cxr_focal.py scores both alike.

    CUDA_VISIBLE_DEVICES=1 python train_cxr_oneshot.py --out runs_reader/cxr_oneshot
With --aug, every training radiograph is jittered as the detector baselines' are (cxr_aug.py); the presence answers and
the pointing targets are taken from the jittered boxes.
"""
import argparse
import gc
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

from cxr_aug import augment
from cxr_balance import balanced
from cxr_recall import boxes
from perr.cxr import CACHE
from score_cxr_focal import auroc as auroc_ties
from train_cxr_focal_reader import Q8, SHORT, TYPES, YesNoHead, auroc, load_base

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

    def ask(self, keys, answers, ims=None):
        q = ASK + "\n" + "\n".join(f"{SHORT[t]}: yes or no?" for t in TYPES)
        enc = self.proc(text=[self.prompt(q)] * len(keys), images=ims or [image(k) for k in keys], return_tensors="pt",
                        padding=True)
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

    def point(self, keys, types, targets=None, ims=None):
        enc = self.proc(text=[self.prompt(POINT.format(SHORT[t])) for t in types], images=ims or [image(k) for k in keys],
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


def jittered(k, bb, rng):
    """cxr_aug.augment on one training radiograph: (image, presence answers, {type: target}) from the moved boxes."""
    img, moved = augment(np.load(f"{CACHE}/{k}.npy"), bb.get(k, []), rng)
    tg = {}
    for t in TYPES:
        bt = [x for x in moved if x["finding"] == t]
        if bt:
            r = max(bt, key=lambda x: x["w"] * x["h"])
            tg[t] = (int(round((r["x"] + r["w"] / 2) / 512 * 1000)), int(round((r["y"] + r["h"] / 2) / 512 * 1000)))
    return Image.fromarray(img).convert("RGB"), [int(t in tg) for t in TYPES], tg


def forward(model, enc):
    n, tm = enc.pop("n_prompt"), enc.pop("target_mask")
    return model(**enc.to(model.device)).logits, n, tm


@torch.no_grad()
def presence(model, codec, keys, bs=16, readout="bf16"):
    """P(yes) per finding, two passes as the verifier's score(); readout as train_cxr_focal_reader.YesNoHead."""
    out = []
    with YesNoHead(model, codec, readout) as yn:
        for i in range(0, len(keys), bs):
            ks = keys[i:i + bs]
            ans = [[0] * len(TYPES) for _ in ks]
            for _ in range(2):
                enc, pos = codec.ask(ks, ans)
                lg, n, _ = forward(model, enc)
                pos = torch.tensor([[n + p for p in q] for q in pos], device=lg.device)
                p = yn(lg, (torch.arange(len(ks))[:, None], pos - 1)).softmax(-1)[..., 0]
                ans = (p > 0.5).int().tolist()
            out += p.tolist()
    return np.array(out)


def backbone(out):
    """(checkpoint, 4-bit) the run in <out> was trained on (backbone.json; runs before it: Qwen3-VL-8B in bf16)."""
    path = f"{out}/backbone.json"
    if os.path.exists(path):
        cfg = json.load(open(path))
        return cfg["base"], cfg["load_4bit"]
    return Q8, False


def rescore(out, res_name, sets, readout="bf16"):
    """Presence probabilities of a trained run read again (readout), the pointed places kept from its scored files:
    results/cxr_focal/<run>/<set>.jsonl -> results/cxr_focal/<res_name>/<set>.jsonl. "cxdet_val" (the held-out tenth,
    not written by training) is scored anew and its mean presence AUROC (ties one half) printed as in training."""
    ckpt, four_bit = backbone(out)
    proc = AutoProcessor.from_pretrained(ckpt)
    proc.tokenizer.padding_side = "left"
    codec = Codec(proc)
    model = PeftModel.from_pretrained(load_base(ckpt, four_bit), f"{out}/lora_best").eval()
    run = os.path.basename(out.rstrip("/"))
    os.makedirs(f"results/cxr_focal/{res_name}", exist_ok=True)
    for st in sets:
        if st == "cxdet_val":
            fkeys = sorted({json.loads(l)["key"] for l in open("data/cxr_focal/wm/cxdet_train.jsonl")})
            random.Random(0).shuffle(fkeys)
            keys = sorted(fkeys[:len(fkeys) // 10])
            place = {}
        else:
            old = [json.loads(l) for l in open(f"results/cxr_focal/{run}/{st}.jsonl")]
            keys = sorted({r["key"] for r in old})
            place = {(r["key"], r["i"]): (r["r"], r["c"]) for r in old}
        p = presence(model, codec, keys, readout=readout)
        with open(f"results/cxr_focal/{res_name}/{st}.jsonl", "w") as f:
            for k, q in zip(keys, p):
                for j, t in enumerate(TYPES):
                    y, x = place.get((k, j), (256.0, 256.0))
                    f.write(json.dumps({"key": k, "i": j, "r": y, "c": x, "wm": 1.0, "src": "wm", "rank": 0,
                                        **{f"p_{u}": (float(q[j]) if u == t else 0.0) for u in TYPES}}) + "\n")
        if st == "cxdet_val":
            b = boxes("cxdet")
            pos = {t: set(b[b.finding == t].key) for t in TYPES}
            v = float(np.nanmean([auroc_ties(p[:, j], np.array([k in pos[t] for k in keys])) for j, t in enumerate(TYPES)]))
            print(f"epoch - validation mean presence AUROC {v:.4f}", flush=True)
        print(f"rescored {st}: {len(keys)} images", flush=True)


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


def score_sets(out, sets=("mimic_validate", "mimic_test", "nih_test")):
    """P(yes) per finding on the world model's other evaluation images, and the pointed place on NIH (MIMIC has no
    boxes)."""
    ckpt, four_bit = backbone(out)
    proc = AutoProcessor.from_pretrained(ckpt)
    proc.tokenizer.padding_side = "left"
    codec = Codec(proc)
    model = PeftModel.from_pretrained(load_base(ckpt, four_bit), f"{out}/lora_best").eval()
    run = os.path.basename(out.rstrip("/"))
    os.makedirs(f"results/cxr_focal/{run}", exist_ok=True)
    for st in sets:
        keys = sorted({json.loads(l)["key"] for l in open(f"data/cxr_focal/wm/{st}.jsonl")})
        p = presence(model, codec, keys)
        loc = locate(model, codec, keys) if st.startswith("nih") else {}
        with open(f"results/cxr_focal/{run}/{st}.jsonl", "w") as f:
            for k, q in zip(keys, p):
                for j, t in enumerate(TYPES):
                    x, y = loc.get((k, t), (256.0, 256.0))
                    f.write(json.dumps({"key": k, "i": j, "r": y, "c": x, "wm": 1.0, "src": "wm", "rank": 0,
                                        **{f"p_{u}": (float(q[j]) if u == t else 0.0) for u in TYPES}}) + "\n")
        print(f"scored {st}: {len(keys)} images", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=0, help="micro-batch size (gradient accumulation to --bs); 0: --bs")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--train-keys", default=None, help="only these training images (few-box settings)")
    ap.add_argument("--balance", action="store_true", help="class-balanced epochs (cxr_balance.py)")
    ap.add_argument("--aug", action="store_true", help="scale, shift and intensity jitter of the training images (cxr_aug.py)")
    ap.add_argument("--score-sets", action="store_true", help="MIMIC validate/test and NIH with the trained adapter")
    ap.add_argument("--readout", default="bf16", choices=("bf16", "fp32"), help="yes/no logits (YesNoHead)")
    ap.add_argument("--rescore", nargs="*", default=None, help="sets to read again with --readout (rescore())")
    ap.add_argument("--res-name", default=None, help="results directory name for --rescore")
    ap.add_argument("--base", default=Q8, help="backbone checkpoint (default Qwen3-VL-8B-Instruct)")
    ap.add_argument("--load-4bit", action="store_true", help="4-bit NF4 backbone with bf16 LoRA (QLoRA), for the 32B model")
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    if a.rescore is not None:
        return rescore(a.out, a.res_name or os.path.basename(a.out.rstrip("/")) + f"_{a.readout}", a.rescore, a.readout)
    if a.score_sets:
        return score_sets(a.out)
    g = pd.read_csv("results/cxr_cxdet.csv")
    b = boxes("cxdet")
    train_keys = sorted(g[g.split == "train"].key)
    random.Random(0).shuffle(train_keys)                         # the focal reader's held-out tenth
    focal_rows = [json.loads(l) for l in open("data/cxr_focal/wm/cxdet_train.jsonl")]
    fkeys = sorted({r["key"] for r in focal_rows})
    random.Random(0).shuffle(fkeys)
    val_keys = fkeys[:len(fkeys) // 10]
    tr_keys = [k for k in train_keys if k not in set(val_keys)]
    if a.train_keys:
        allowed = set(open(a.train_keys).read().split())
        tr_keys = [k for k in tr_keys if k in allowed]
    test_keys = sorted(g[g.split == "test"].key)
    tg = targets_of(b, tr_keys + val_keys + test_keys)
    lab = {k: [int(t in tg[k]) for t in TYPES] for k in tg}
    rng = np.random.default_rng(0) if a.aug else None
    bb = {k: [{"finding": r.finding, "x": r.x, "y": r.y, "w": r.w, "h": r.h} for r in g.itertuples() if r.finding in TYPES]
          for k, g in b[b.key.isin(set(tr_keys))].groupby("key")} if a.aug else {}
    if a.balance:
        tr_keys = balanced(tr_keys, lambda k: any(lab[k]))
    print(f"train {len(tr_keys)}, validation {len(val_keys)}, test {len(test_keys)} images", flush=True)

    proc = AutoProcessor.from_pretrained(a.base)
    proc.tokenizer.padding_side = "left"
    codec = Codec(proc)
    base = load_base(a.base, a.load_4bit)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"base": a.base, "load_4bit": a.load_4bit}, open(f"{a.out}/backbone.json", "w"))
    if a.load_4bit:
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    targets = (r"(.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))|"
               r"(.*visual.*\.(qkv|proj|linear_fc1|linear_fc2))")
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                            target_modules=targets))
    if a.load_4bit:          # from_pretrained returns eval mode: gradient checkpointing and dropout act only in train mode
        model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    point_jobs = [(k, t) for k in tr_keys for t in TYPES if t in tg[k]]
    per_epoch = math.ceil(len(tr_keys) / a.bs) + math.ceil(len(point_jobs) / a.bs)
    steps = per_epoch * a.epochs
    sched = lambda s: a.lr * min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps))  # noqa: E731

    def validate():
        model.eval()
        p = presence(model, codec, val_keys, readout=a.readout)
        model.train()
        return float(np.nanmean([auroc_ties(p[:, j], np.array([lab[k][j] for k in val_keys]).astype(bool)) for j in range(len(TYPES))]))

    best, step = -1.0, 0
    for ep in range(a.epochs):
        batches = [("ask", tr_keys[i:i + a.bs]) for i in range(0, len(tr_keys), a.bs)]
        random.shuffle(point_jobs)
        batches += [("point", point_jobs[i:i + a.bs]) for i in range(0, len(point_jobs), a.bs)]
        random.shuffle(batches)
        tot = 0.0
        for kind, ch in batches:
            mb = a.micro or a.bs                    # micro-batches: the same loss and step as the whole batch, less memory
            subs = [ch[i:i + mb] for i in range(0, len(ch), mb)]
            if kind == "ask":
                n_tot = float(len(ch) * len(TYPES))
            elif rng is None:
                encs = [codec.point([k for k, _ in sb], [t for _, t in sb], [tg[k][t] for k, t in sb]) for sb in subs]
                n_tot = float(sum(int(e["target_mask"].sum()) for e in encs))     # answer tokens of the whole batch
            else:                               # a jittered image whose box left the canvas keeps the original
                encs = []
                for sb in subs:
                    js = [jittered(k, bb, rng) for k, _ in sb]
                    ims = [j[0] if t in j[2] else image(k) for j, (k, t) in zip(js, sb)]
                    tgs = [j[2][t] if t in j[2] else tg[k][t] for j, (k, t) in zip(js, sb)]
                    encs.append(codec.point([k for k, _ in sb], [t for _, t in sb], tgs, ims))
                n_tot = float(sum(int(e["target_mask"].sum()) for e in encs))
            loss = 0.0
            for si, sb in enumerate(subs):
                if kind == "ask":
                    ans, ims = [lab[k] for k in sb], None
                    if rng is not None:
                        js = [jittered(k, bb, rng) for k in sb]
                        ans, ims = [j[1] for j in js], [j[0] for j in js]
                    enc, pos = codec.ask(sb, ans, ims)
                    lg, n, _ = forward(model, enc)
                    pos = torch.tensor([[n + p for p in q] for q in pos], device=lg.device)
                    lg = lg[torch.arange(len(sb))[:, None], pos - 1].float()
                    two = torch.stack([lg[..., codec.no], lg[..., codec.yes]], -1)
                    part = torch.nn.functional.cross_entropy(two.view(-1, 2), torch.tensor(ans, device=lg.device).view(-1),
                                                             reduction="sum") / n_tot
                else:
                    enc = encs[si]
                    ids = enc["input_ids"].clone()
                    lg, n, tm = forward(model, enc)
                    tgt = ids[:, n:].to(lg.device)
                    pred = lg[:, n - 1:-1].float()
                    ce = torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), reduction="none")
                    m = tm.to(lg.device).float().view(-1)
                    part = (ce * m).sum() / n_tot
                part.backward()
                loss += part.detach()
            for grp in opt.param_groups:
                grp["lr"] = sched(step)
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
    del model, base, opt          # the trained copy is dropped before the best adapter is loaded onto a fresh base
    gc.collect()
    torch.cuda.empty_cache()
    base = load_base(a.base, a.load_4bit)
    model = PeftModel.from_pretrained(base, f"{a.out}/lora_best").eval()
    for name, keys in (("cxdet_test", test_keys),):
        p = presence(model, codec, keys)
        loc = locate(model, codec, keys)
        run = os.path.basename(a.out.rstrip("/"))
        os.makedirs(f"results/cxr_focal/{run}", exist_ok=True)
        with open(f"results/cxr_focal/{run}/{name}.jsonl", "w") as f:
            for k, q in zip(keys, p):
                for j, t in enumerate(TYPES):
                    x, y = loc[(k, t)]
                    row = {"key": k, "i": j, "r": y, "c": x, "wm": 1.0, "src": "wm", "rank": 0,
                           **{f"p_{u}": (float(q[j]) if u == t else 0.0) for u in TYPES}}
                    f.write(json.dumps(row) + "\n")
        print(f"scored {name}", flush=True)


if __name__ == "__main__":
    main()
