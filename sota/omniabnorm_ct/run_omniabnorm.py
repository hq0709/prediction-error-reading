#!/usr/bin/env python
"""OmniAbnorm-CT (Zhao et al., CVPR 2026) on our benchmark: axial slice sweep -> stacked 3D mask -> answer.

OmniAbnorm-CT works on one 2D key slice: the VLM (Qwen2.5-VL-7B + LoRA) sees the key slice, the segmentation module
(2D U-Net + query decoder) sees the key slice and its 8 adjacent slices (9 channels), and the VLM triggers the
segmentation module by emitting <SEG> (text-guided grounded report generation). The repo has no whole-volume
procedure, so every axial slice of the cached array is used as a key slice once:

  cached HU slice -> official window (PROMPT.txt) -> uint8 image (what inference.py reads from PNG/JPG files) ->
  inference.py `preprocess_input` (per-slice z-score, 9-slice context with zero padding at the volume ends, square
  padding, 512 x 512 bilinear resize; VLM image = min-max uint8 of the key slice at 512 x 512 pixels through
  qwen-vl-utils) -> first generation round of evaluate_refer_grounding.py, batched as there (DataLoader workers
  run the processor in the collate function): greedy VLM generation (base generation config), last-layer hidden
  state of the first <SEG> -> U-Net -> sigmoid -> > 0.5 -> official `recover_single_slice` -> 3D mask on the
  cached grid (slice z of the mask = slice z of the cached array).

The second generation round (the report written after the box overlay) does not change the mask; it is skipped in
the sweep and run once per case, with the official `generate_text_and_grounding`, on the key slice of the answer
(the slice where the answer component is largest), so `raw` carries the model's own report.
"""
import argparse
import collections
import json
import os
import sys
import time

sys.dont_write_bytecode = True  # importing the official repo must not write __pycache__ into the clone
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # workers are forked after the main process used the tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/raid/home/CAMCA/hj880/ext_src/OmniAbnorm-CT"
BASE = ("/raid/home/CAMCA/hj880/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
        "cc594898137f460bfe9f0759e9844b3ce807cfb5")
CKPT = ("/raid/home/CAMCA/hj880/hf_home/hub/models--zzh99--OminiAbnorm-CT/snapshots/"
        "f9448ea3b786f79e3fdef388a85300a6d52434fd/ominiabnorm-ct-7b")
sys.path.insert(0, HERE)
sys.path.insert(1, REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import repeat  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

import inference as oa  # noqa: E402  official inference.py (functions only; its __main__ block is not run)
from common import CACHE, load_cached, mask_to_answer  # noqa: E402

# Fixed before any evaluation run (see PROMPT.txt).
PROMPTS = {
    "luna": "Evaluate the lungs for any lung nodule. If any, localize it on the image and describe the findings.",
    "liver": "Evaluate the liver for any liver tumor. If any, localize it on the image and describe the findings.",
}
WINDOWS = {"luna": (-1150.0, 350.0), "liver": (-140.0, 260.0)}  # dataset/config/other_datasets.json
# External liver test set (HCC-TACE-Seg, cache/liver_ext): exactly the MSD liver protocol (same query, same window).
PROMPTS["liver_ext"] = PROMPTS["liver"]
WINDOWS["liver_ext"] = WINDOWS["liver"]
CONTEXT = 4  # inference.py: context_range = 4 -> 9 slices


def load_model():
    """Official inference.py `load_model`, with tokenizer_path=None and without torch.compile (see NOTES.md)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.load_adapter(CKPT)
    processor = Qwen2_5_VLProcessor.from_pretrained(BASE, padding_side="left", use_fast=True)
    # The released checkpoint has no `extended_tokenizer/`; the official tokenizer_path=None branch adds <SEG>.
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
    model.resize_token_embeddings(len(processor.tokenizer))
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids("<SEG>")
    ck = torch.load(f"{CKPT}/seg_model_lm_head_embed_tokens.pt", map_location="cpu")
    loaded, missed = [], []
    for n, p in model.named_parameters():
        for key, sd in (("lm_head", "lm_head_state_dict"), ("embed_tokens", "embed_tokens_state_dict")):
            if key in n:
                if n in ck.get(sd, {}):
                    p.data = ck[sd][n].data
                    loaded.append(n)
                else:
                    missed.append(n)
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    info = {"seg_token_idx": int(seg_token_idx), "vocab": len(processor.tokenizer), "loaded": loaded,
            "missed": missed, "lora_B_tensors": len(lora_b),
            "lora_B_nonzero": int(sum(bool(p.detach().abs().sum() > 0) for p in lora_b))}
    assert not missed and len(loaded) == 2 and seg_token_idx == 151665, info
    model = model.cuda().eval()

    from model.unet import UNET

    seg_model = UNET(input_channels=9).cuda()
    seg_model.load_saved(ck)
    seg_model.eval()
    msd = set(seg_model.state_dict())
    csd = set(ck["seg_model"])
    info.update(seg_missing=sorted(msd - csd), seg_unexpected=sorted(csd - msd))
    assert not info["seg_missing"] and not info["seg_unexpected"], info
    return (model, seg_model, processor, seg_token_idx), info


def to_u8(vol: np.ndarray, window) -> np.ndarray:
    """Windowed uint8 slices: the 8-bit image a user exports to PNG for inference.py."""
    lo, hi = window
    return np.round((np.clip(vol.astype(np.float32), lo, hi) - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def zscore(u8_slice: np.ndarray) -> np.ndarray:
    """inference.py: img = (img - np.mean(img)) / (np.std(img) + 1e-10) on one uint8 slice (float64)."""
    return (u8_slice - np.mean(u8_slice)) / (np.std(u8_slice) + 1e-10)


class SliceSource:
    """Windowed, z-scored axial slices of one volume, computed on demand (each slice once, LRU-cached); the values
    are identical to z-scoring every slice of every 9-slice window as inference.py does."""

    def __init__(self, vol, window, cache=48):
        self.vol, self.window, self.cache, self._lru = vol, window, cache, collections.OrderedDict()
        self.nz = vol.shape[0]

    def __getitem__(self, z):
        if z not in self._lru:
            self._lru[z] = zscore(to_u8(np.asarray(self.vol[z]), self.window))
            if len(self._lru) > self.cache:
                self._lru.popitem(last=False)
        return self._lru[z]


def slice_sample(src: SliceSource, z: int, prompt: str):
    """inference.py `preprocess_input` for key slice z of a cranio-caudal stack (superior slices come first)."""
    nz = src.nz
    pre = [zz for zz in range(z + CONTEXT, z, -1) if zz < nz]  # shown before z: z+4 ... z+1 (more superior)
    post = [zz for zz in range(z - 1, z - CONTEXT - 1, -1) if zz >= 0]  # shown after z: z-1 ... z-4
    stack = np.stack([src[zz] for zz in pre + [z] + post], axis=-1)
    seg_img, tinfo = oa.pad_truncate_and_resize(stack, 512, 512, CONTEXT - len(pre), CONTEXT - len(post))
    seg_img = torch.tensor(repeat(seg_img, "h w d -> d h w"))
    raw = oa.format_data(oa.get_pil_image(src[z]), prompt)
    return raw, seg_img, tinfo


class SliceSet(torch.utils.data.Dataset):
    """All key slices of all cases, in order. `arrays` (in-memory volumes, e.g. the synthetic sphere) or the cached
    .vol.npy files (memory-mapped, read slice by slice inside the workers)."""

    def __init__(self, ds, items, arrays=None):
        self.ds, self.items, self.arrays, self._src = ds, items, arrays, {}

    def __len__(self):
        return len(self.items)

    def _source(self, case):
        if case not in self._src:
            if len(self._src) >= 3:
                self._src.pop(next(iter(self._src)))
            vol = self.arrays[case] if self.arrays is not None else np.load(f"{CACHE}/{self.ds}/{case}.vol.npy", mmap_mode="r")
            self._src[case] = SliceSource(vol, WINDOWS[self.ds])
        return self._src[case]

    def __getitem__(self, i):
        case, z = self.items[i]
        raw, seg_img, tinfo = slice_sample(self._source(case), z, PROMPTS[self.ds])
        # float64 -> float32 here, -> bfloat16 on the GPU: torch casts double to bfloat16 through float, so the
        # U-Net input is bit-identical to the official float64 -> bfloat16 cast.
        return case, z, raw, seg_img.float(), tinfo


class Collate:
    """Refer_Grounding_Collator of the official evaluation (processor call per batch)."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items):
        cases, zs, raws, segs, tinfos = zip(*items)
        texts = [self.processor.apply_chat_template(r, tokenize=False, add_generation_prompt=True) for r in raws]
        ims = [process_vision_info(r)[0] for r in raws]
        inp = self.processor(text=texts, images=ims, return_tensors="pt", padding=True, truncation=True)
        return list(cases), list(zs), dict(inp), torch.stack(segs), list(tinfos)


def _worker_init(_):
    torch.set_num_threads(1)  # the box is CPU-oversubscribed; multi-threaded CPU ops thrash (same outputs)
    cv2.setNumThreads(1)


@torch.no_grad()
def first_round(M, inp, seg_imgs):
    """First round of evaluate_refer_grounding.py `generate_text_and_grounding_from_batch`, returning the sigmoid
    map (not only its 0.5 threshold) for samples that emitted <SEG>; None for the others."""
    model, seg_model, processor, seg_idx = M
    inp = {k: v.cuda() for k, v in inp.items()}
    out = model.generate(**inp, output_scores=True, output_hidden_states=True, return_dict_in_generate=True,
                         max_new_tokens=1024)
    gen = out.sequences
    n_in = inp["input_ids"].shape[1]
    dec = processor.batch_decode(gen[:, n_in:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    strip = [t for t in processor.tokenizer.all_special_tokens if t != "<SEG>"]
    for i in range(len(dec)):
        for t in strip:
            dec[i] = dec[i].replace(t, "")
    hs = torch.cat([s[-1] for s in out.hidden_states], dim=1)
    nb = gen.shape[0]
    q = torch.zeros(nb, 1, hs.size(-1), device=hs.device)
    has = []
    is_seg = gen == seg_idx
    for b in range(nb):
        pos = torch.where(is_seg[b])[0]
        if len(pos) > 0:
            q[b, 0] = hs[b, pos[0]]
            has.append(b)
    prob = [None] * nb
    if has:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            imgs = seg_imgs[has].cuda().to(torch.bfloat16)
            p = torch.sigmoid(seg_model(imgs, q[has]))  # (n, 1, 512, 512)
        p = p.float().cpu().numpy()[:, 0]
        for j, b in enumerate(has):
            prob[b] = p[j]
    del out, hs
    return dec, prob


@torch.no_grad()
def report_on_slice(M, src, z, prompt):
    """The official two-round `generate_text_and_grounding` (inference.py) on one key slice."""
    model, seg_model, processor, seg_idx = M
    raw, seg_img, tinfo = slice_sample(src, z, prompt)
    texts = [processor.apply_chat_template(raw, tokenize=False, add_generation_prompt=True)]
    vin = processor(text=texts, images=[process_vision_info(raw)[0]], return_tensors="pt", padding=True,
                    truncation=True)
    data = {"vllm_input": vin, "vllm_raw_input": [raw], "seg_img": seg_img[None], "transform_info": tinfo}
    text, seg_pred = oa.generate_text_and_grounding(model, seg_model, processor, seg_idx, data)
    return text, seg_pred is not None


class CaseAcc:
    """Collects the per-slice results of one case into the 3D mask / probability map on the cached grid."""

    def __init__(self, shape):
        self.mask = np.zeros(shape, bool)
        self.prob = np.zeros(shape, np.float16)
        self.n, self.seg_z, self.mask_z, self.texts, self.t0 = 0, [], [], collections.Counter(), time.time()

    def add(self, z, tinfo, text, p):
        self.n += 1
        if p is None:
            self.texts[text.strip()[:120]] += 1
            return
        self.seg_z.append(int(z))
        binm = oa.recover_single_slice(np.where(p > 0.5, 1.0, 0.0).astype(np.float32), tinfo) > 0.5
        if binm.any():
            self.mask_z.append(int(z))
            self.mask[z] = binm
            self.prob[z] = oa.recover_single_slice(p.astype(np.float32), tinfo)

    def diag(self):
        return {"n_key_slices": self.n, "n_seg": len(self.seg_z), "n_mask": len(self.mask_z),
                "seg_z": sorted(self.seg_z), "mask_z": sorted(self.mask_z), "non_seg_texts": dict(self.texts.most_common(5))}


def sweep_cases(M, ds, cases, shapes, batch_size, workers, arrays=None, key_slices=None):
    """Every axial slice of every case (or `key_slices[case]`) through the first round; yields (case, CaseAcc) as
    soon as the last key slice of a case is done. One persistent set of DataLoader workers for all cases."""
    order = {c: (key_slices[c] if key_slices else list(range(shapes[c][0] - 1, -1, -1))) for c in cases}  # cranio-caudal
    items = [(c, z) for c in cases for z in order[c]]
    loader = torch.utils.data.DataLoader(SliceSet(ds, items, arrays), batch_size=batch_size, shuffle=False,
                                         num_workers=workers, collate_fn=Collate(M[2]), worker_init_fn=_worker_init,
                                         prefetch_factor=4 if workers else None, persistent_workers=False)
    accs = {}
    for bc, bz, inp, seg_imgs, tinfos in loader:
        dec, pr = first_round(M, inp, seg_imgs)
        for c, z, tinfo, t, p in zip(bc, bz, tinfos, dec, pr):
            if c not in accs:
                accs[c] = CaseAcc(shapes[c])
            accs[c].add(z, tinfo, t, p)
            if accs[c].n == len(order[c]):
                yield c, accs.pop(c)


def finalize(M, ds, case, acc, vol, sp, report=True):
    t1 = time.time()
    ans = mask_to_answer(acc.mask, acc.prob, sp)
    rep, rep_seg, key_z = None, None, None
    if report and ans["present"]:
        key_z = ans["candidates"][0]["key_z"]
        rep, rep_seg = report_on_slice(M, SliceSource(vol, WINDOWS[ds]), key_z, PROMPTS[ds])
    t2 = time.time()
    raw = {"prompt": PROMPTS[ds], "window": WINDOWS[ds], "report_key_z": key_z, "report": rep,
           "report_seg": rep_seg, "n_components": ans["n_components"], "mask_vox": int(acc.mask.sum()),
           "max_prob": float(acc.prob.max()), "shape": list(vol.shape), "spacing": list(sp),
           "sweep_s": round(t1 - acc.t0, 1), "finalize_s": round(t2 - t1, 1), **acc.diag()}
    return {"case": case, "present": ans["present"], "zyx": ans["zyx"], "diameter_mm": ans["diameter_mm"],
            "score": ans["score"], "candidates": ans["candidates"], "raw": json.dumps(raw)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["luna", "liver", "liver_ext"])
    ap.add_argument("--cases", required=True, help="text file with one case id per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-masks", default=None, help="optional directory for <case>.mask.npz (smoke tests)")
    args = ap.parse_args()
    torch.set_num_threads(2)
    cases = open(args.cases).read().split()
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["case"] for line in open(args.out) if line.strip()}
    todo = [c for c in cases if c not in done]
    print(f"{len(cases)} cases, {len(done)} already in {args.out}, {len(todo)} to run", flush=True)
    shapes = {c: np.load(f"{CACHE}/{args.ds}/{c}.vol.npy", mmap_mode="r").shape for c in todo}
    t0 = time.time()
    M, info = load_model()
    print("model", json.dumps(info), f"load {time.time() - t0:.1f}s", flush=True)
    for k, (c, acc) in enumerate(sweep_cases(M, args.ds, todo, shapes, args.batch_size, args.workers)):
        vol, sp = load_cached(args.ds, c)
        rec = finalize(M, args.ds, c, acc, vol, sp)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if args.save_masks:
            os.makedirs(args.save_masks, exist_ok=True)
            np.savez_compressed(f"{args.save_masks}/{c}.mask.npz", mask=acc.mask, prob=acc.prob)
        r = json.loads(rec["raw"])
        print(f"[{k + 1}/{len(todo)}] {c} present={rec['present']} zyx={rec['zyx']} d={rec['diameter_mm']} "
              f"score={rec['score']} comps={r['n_components']} seg={r['n_seg']}/{r['n_key_slices']} "
              f"mask={r['n_mask']} sweep={r['sweep_s']}s finalize={r['finalize_s']}s "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
