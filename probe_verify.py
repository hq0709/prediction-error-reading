"""Isolated verification probe: can the model tell nodules from vessels on the candidate sheets, outside the agent loop?"""
import asyncio, base64, json, re, sys
import numpy as np
from openai import AsyncOpenAI
from PIL import Image
import io
from perr.study import load_luna, pilot_cases
from perr.harness import Session
from perr.score import score_peaks, _mm

MODEL, PORT, MODE, SIZE = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
client = AsyncOpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x", timeout=900)
sem = asyncio.Semaphore(8)

def label(s, z, y, x):
    """1 = inside a nodule >= 6 mm, 0 = not inside any annotated nodule, None = inside a smaller nodule (skip)."""
    p = _mm((z, y, x), s.spacing)
    for n in s.nodules:
        if np.linalg.norm(p - _mm(n.zyx, s.spacing)) <= n.diameter_mm / 2 + 3:
            return 1 if n.diameter_mm >= 6 else None
    return 0

def img_part(png):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}

async def ask(content, max_tokens=200):
    async with sem:
        r = await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": content}], temperature=0, max_tokens=max_tokens)
        return r.choices[0].message.content or ""

async def one(uid):
    s = load_luna(uid); S = Session(s, "harness_wm", MODE, SIZE); S.overview()
    out, recs = S.call("measure", {"peaks": [1, 2, 3, 4, 5]})
    table = json.loads(out)["candidates"]
    brief = [{k: c.get(k) for k in ("peak", "equivalent_diameter_mm", "long_axis_mm_on_slice", "mean_HU", "warning") if c.get(k) is not None} for c in table]
    sheet = recs[0].png
    q = ("These are the five largest world-model candidates in one chest CT, one row per candidate: axial view with the "
         "measured outline (green), coronal, sagittal; 40 mm each; magenta ticks mark the exact address. Measurements: "
         + json.dumps(brief) + ". A nodule is a compact blob, round in all three planes; a vessel is a tube in at least "
         "one plane; chest wall sits on the lung edge. Which candidate is the largest lung nodule at least 6 mm in "
         'diameter? Reply with JSON only: {"peak": n} or {"peak": null} if none is.')
    sel_txt = await ask([img_part(sheet), {"type": "text", "text": q}])
    m = re.search(r'"peak"\s*:\s*(null|\d+)', sel_txt)
    pick = None if (not m or m.group(1) == "null") else int(m.group(1))
    sel_hit = pick is not None and 1 <= pick <= len(S.peaks) and score_peaks(s, S.peaks[pick - 1:pick], 1)
    # per-candidate yes/no on each row of the sheet
    im = Image.open(io.BytesIO(sheet)).convert("RGB"); h = im.height // len(table)
    rows = []
    for i, c in enumerate(table):
        z, y, x, _ = S.peaks[c["peak"] - 1]
        lab = label(s, z, y, x)
        if lab is None:
            continue
        buf = io.BytesIO(); im.crop((0, i * h, im.width, (i + 1) * h - 4)).save(buf, format="PNG")
        qq = ("One candidate from a chest CT: axial (with measured outline), coronal and sagittal views, 40 mm each; "
              f"magenta ticks mark the exact address. Measurement: {json.dumps(brief[i])}. Is the structure at the "
              "marked address a lung nodule (a compact blob, round in all three planes), rather than a vessel, airway "
              'or chest wall? Reply with JSON only: {"nodule": true} or {"nodule": false}.')
        a = await ask([img_part(buf.getvalue()), {"type": "text", "text": qq}], 60)
        yes = bool(re.search(r'"nodule"\s*:\s*true', a))
        rows.append((lab, yes))
    return {"case": uid, "pick": pick, "sel_hit": bool(sel_hit), "rows": rows,
            "top1_hit": bool(score_peaks(s, S.peaks[:1], 1)), "top5_hit": bool(score_peaks(s, S.peaks[:5], 5))}

async def main():
    res = await asyncio.gather(*[one(u) for u in pilot_cases(260)[160:260]])
    json.dump(res, open(f"results/probe_verify_{MODEL}.json", "w"))
    rows = [r for x in res for r in x["rows"]]
    pos = [y for l, y in rows if l == 1]; neg = [y for l, y in rows if l == 0]
    print(f"{MODEL}: single-turn selection hit {np.mean([x['sel_hit'] for x in res]):.2f} | WM top-1 {np.mean([x['top1_hit'] for x in res]):.2f} | top-5 oracle {np.mean([x['top5_hit'] for x in res]):.2f}")
    print(f"per-candidate: sensitivity {np.mean(pos):.2f} (n={len(pos)} true nodules), specificity {1 - np.mean(neg):.2f} (n={len(neg)} non-nodules), balanced {(np.mean(pos) + 1 - np.mean(neg)) / 2:.2f}")
    print("picks:", {k: sum(1 for x in res if x['pick'] == k) for k in (None, 1, 2, 3, 4, 5)})

asyncio.run(main())
