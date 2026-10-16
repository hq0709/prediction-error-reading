"""Per-candidate reader probability P(nodule) via logprobs, for every top-5 candidate; then offline fusion rules."""
import asyncio, base64, io, json, math, re, sys
import numpy as np
from openai import AsyncOpenAI
from PIL import Image
from perr.study import load_luna, pilot_cases
from perr.harness import Session
from perr.score import score_peaks, _mm

MODEL, PORT, MODE, SIZE, LO, HI = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
K = int(sys.argv[7]) if len(sys.argv) > 7 else 5
client = AsyncOpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x", timeout=900)
sem = asyncio.Semaphore(8)
Q = ("One candidate from a chest CT: axial (with measured outline), coronal and sagittal views, 40 mm each; magenta "
     "ticks mark the exact address. Measurement: {m}. Is the structure at the marked address a lung nodule (a compact "
     "blob, round in all three planes), rather than a vessel, airway or chest wall? Answer with one word: yes or no.")

async def p_yes(png, meas):
    async with sem:
        r = await client.chat.completions.create(model=MODEL, temperature=0, max_tokens=1, logprobs=True, top_logprobs=10,
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                                                    {"type": "text", "text": Q.format(m=json.dumps(meas))}]}])
    tl = r.choices[0].logprobs.content[0].top_logprobs
    py = sum(math.exp(t.logprob) for t in tl if t.token.strip().lower().startswith("yes"))
    pn = sum(math.exp(t.logprob) for t in tl if t.token.strip().lower().startswith("no"))
    return py / max(py + pn, 1e-9)

async def one(uid):
    s = load_luna(uid); S = Session(s, "harness_wm", MODE, SIZE); S.overview()
    items = []
    for lo in range(1, min(K, len(S.peaks)) + 1, 5):
        out, recs = S.call("measure", {"peaks": list(range(lo, min(lo + 5, min(K, len(S.peaks)) + 1)))})
        table = json.loads(out)["candidates"]
        im = Image.open(io.BytesIO(recs[0].png)).convert("RGB"); h = im.height // len(table)
        items += [(c, im.crop((0, i * h, im.width, (i + 1) * h - 4))) for i, c in enumerate(table)]
    cands = []
    for c, crop in items:
        buf = io.BytesIO(); crop.save(buf, format="PNG")
        meas = {k: c.get(k) for k in ("equivalent_diameter_mm", "long_axis_mm_on_slice", "mean_HU", "warning") if c.get(k) is not None}
        p = await p_yes(buf.getvalue(), meas)
        hit = bool(score_peaks(s, S.peaks[c["peak"] - 1:c["peak"]], 1))
        cands.append({"peak": c["peak"], "p": p, "hit6": hit, "wm": S.peaks[c["peak"] - 1][3], "d": c.get("equivalent_diameter_mm"), "warn": "warning" in c})
    return {"case": uid, "cands": cands}

async def main():
    res = await asyncio.gather(*[one(u) for u in pilot_cases(HI)[LO:HI]])
    json.dump(res, open(f"results/probe_lp_{MODEL}_{LO}_{HI}_k{K}.json", "w"))
    def rate(choose):
        return np.mean([bool(choose(r["cands"]) is not None and choose(r["cands"])["hit6"]) for r in res])
    first = lambda cs: cs[0]
    best_p = lambda cs: max(cs, key=lambda c: c["p"])
    first_yes = lambda cs: next((c for c in cs if c["p"] > 0.5), cs[0])
    fuse = lambda cs: max(cs, key=lambda c: (c["wm"] / cs[0]["wm"]) * c["p"])
    fuse_lr = lambda cs: max(cs, key=lambda c: math.log(c["wm"] / cs[0]["wm"]) + math.log(max(c["p"], 1e-6)))
    print(f"{MODEL} [{LO}:{HI}] K={K} WM top-1 {rate(first):.2f} | reader max-p {rate(best_p):.2f} | first p>0.5 in WM order {rate(first_yes):.2f} "
          f"| WM x reader {rate(fuse):.2f} | oracle top-K {np.mean([any(c['hit6'] for c in r['cands']) for r in res]):.2f}")
    ps = [c["p"] for r in res for c in r["cands"]]; ys = [c["hit6"] for r in res for c in r["cands"]]
    pos = [p for p, y in zip(ps, ys) if y]; neg = [p for p, y in zip(ps, ys) if not y]
    auc = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    print(f"reader AUROC for candidate = nodule>=6mm: {auc:.2f} (pos {len(pos)}, neg {len(neg)})")

asyncio.run(main())
