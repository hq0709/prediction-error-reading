"""Score reader strips with P(yes), then the selection table: world model alone, reader alone, the two fused, oracle.

    python score_reader.py hf  <model_dir>          <name> data/reader/eval_60_160.jsonl [more.jsonl ...]
    python score_reader.py api <served_name>:<port> <name> data/reader/eval_60_160.jsonl [more.jsonl ...]
    python score_reader.py table results/reader_<name>_eval_60_160.json
Writes results/reader_<name>_<eval>.json (one row per candidate, with p) and prints the table at K = 5 and 10.
Fusion rule (fixed in advance): pick argmax over the top-K of (world-model error / top error) x P(yes).
"""
import argparse
import asyncio
import base64
import io
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from perr.reader_prompt import prepare

YES, NO = ("yes", "Yes", "YES"), ("no", "No", "NO")


def table(rows, ks=(5, 10, 20, 30)):
    by = defaultdict(list)
    for r in rows:
        by[r["uid"]].append(r)
    lines = []
    kmax = max(len(v) for v in by.values())
    for K in [k for k in ks if k <= kmax]:
        cases = [sorted(v, key=lambda r: r["peak"])[:K] for v in by.values()]
        rate = lambda pick: np.mean([pick(cs)["label"] for cs in cases])
        wm1 = rate(lambda cs: cs[0])
        maxp = rate(lambda cs: max(cs, key=lambda c: c["p"]))
        fuse = rate(lambda cs: max(cs, key=lambda c: c["wm"] / cs[0]["wm"] * c["p"]))
        oracle = np.mean([any(c["label"] for c in cs) for cs in cases])
        pos = [c["p"] for cs in cases for c in cs if c["label"]]
        neg = [c["p"] for cs in cases for c in cs if not c["label"]]
        auc = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        lines.append(f"K={K:2d} n={len(cases)}  WM top-1 {wm1:.2f} | reader max-p {maxp:.2f} | WM x reader {fuse:.2f} "
                     f"| oracle {oracle:.2f} | AUROC {auc:.3f} (pos {len(pos)}, neg {len(neg)})")
    return "\n".join(lines)


def make_hf(model_dir, variant="full", bs=32):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_dir)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": 0}).eval()
    ids = lambda words: sorted({proc.tokenizer.encode(w, add_special_tokens=False)[0] for w in words})
    yes, no = ids(YES), ids(NO)
    nested = "Gemma" in type(proc).__name__      # Gemma 3 wants one image list per prompt

    @torch.no_grad()
    def score(rows):
        out = []
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            imgs, qs = zip(*[prepare(r, variant) for r in chunk])
            texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
                                              tokenize=False, add_generation_prompt=True) for q in qs]
            imgs = list(imgs)
            if nested:
                imgs = [[im] for im in imgs]
            enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
            pr = model(**enc).logits[:, -1, :].float().softmax(-1)
            py, pn = pr[:, yes].sum(-1), pr[:, no].sum(-1)
            out += (py / (py + pn).clamp_min(1e-9)).tolist()
        return out

    return score


def make_api(spec, variant="full", conc=16):
    from openai import AsyncOpenAI
    name, port = spec.rsplit(":", 1)
    async def one(r, sem, client):
        im, q = prepare(r, variant)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        img = base64.b64encode(buf.getvalue()).decode()
        async with sem:
            res = await client.chat.completions.create(
                model=name, temperature=0, max_tokens=1, logprobs=True, top_logprobs=10,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                    {"type": "text", "text": q}]}])
        tl = res.choices[0].logprobs.content[0].top_logprobs
        py = sum(math.exp(t.logprob) for t in tl if t.token.strip().lower().startswith("yes"))
        pn = sum(math.exp(t.logprob) for t in tl if t.token.strip().lower().startswith("no"))
        return py / max(py + pn, 1e-9)

    async def run(rows):           # client and semaphore belong to this event loop
        sem = asyncio.Semaphore(conc)
        async with AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="x", timeout=900) as client:
            return await asyncio.gather(*[one(r, sem, client) for r in rows])

    return lambda rows: asyncio.run(run(rows))


if __name__ == "__main__":
    if sys.argv[1] == "table":
        for path in sys.argv[2:]:
            print(path)
            print(table(json.load(open(path))))
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["hf", "api"])
    ap.add_argument("spec")
    ap.add_argument("name")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--variant", choices=["full", "axial", "nomeas"], default="full")
    a = ap.parse_args()
    score = make_hf(a.spec, a.variant) if a.backend == "hf" else make_api(a.spec, a.variant)
    for path in a.files:
        rows = [json.loads(l) for l in open(path)]
        ps = score(rows)
        keep = ("uid", "peak", "wm", "label", "small", "source", "rating", "patient", "enh_prior", "enh_current")
        rows = [{k: r[k] for k in keep if k in r} | {"p": p} for r, p in zip(rows, ps)]
        stem = os.path.basename(path).rsplit(".", 1)[0]
        tag = "fu_" if "rating" in rows[0] else ""
        json.dump(rows, open(f"results/reader_{a.name}_{tag}{stem}.json", "w"))
        print(f"{a.name} on {tag}{stem}")
        if not tag:
            print(table(rows), flush=True)
