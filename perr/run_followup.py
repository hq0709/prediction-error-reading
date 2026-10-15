"""Follow-up reading on LUMIERE: does the current MRI show progression (RANO) compared with the prior?

    python -m perr.run_followup --model qwen3vl32b --ports 8001 --cond harness_wm --n 60 --out runs/fu_pilot
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import time

from openai import AsyncOpenAI

from .agent import CALL_FORMAT, LOCATE_SPEC, run_episode
from .followup import FollowSession, score_followup
from .lumiere import load_pair, pilot_pairs
from .run import FAMILY

CONV = ("Images use radiological convention: the patient's right is on the image left; axial views are seen from "
        "the feet with anterior at the top. Axial slice indices run from inferior (0) to superior. Both studies are "
        "aligned to the same anatomical space, so one address shows the same anatomy at both times. Locations are "
        "boxes, {{\"bbox_2d\": [x1, y1, x2, y2]}}, in {coords}.")

RANO = ("RANO in brief: progressive disease if the sum of products of perpendicular diameters of the enhancing "
        "lesions grows by 25% or more over the smallest earlier measurement, if any new measurable enhancing lesion "
        "appears, or if the non-enhancing T2/FLAIR abnormality clearly worsens. Otherwise the disease is stable, in "
        "partial response (products down by 50% or more) or in complete response (no enhancing tumour left).")

VIEW = ('- view: render up to 4 views from the raw studies. Arguments: {"question": "...", "expect": "...", '
        '"views": [items]}. Each item is one of\n'
        '    {"sequence": "CT1", "time": "current", "plane": "axial", "index": 90}  a whole slice (sequences: CT1 = '
        "T1 after contrast, T1, T2, FLAIR; time: current or prior);\n"
        '    {"sequence": "FLAIR", "time": "prior", "plane": "axial", "at": {"view": "V3", "bbox_2d": [...]}, '
        '"fov_mm": 80}  the slice through a box seen elsewhere, centred on it;\n'
        '    {"from": "V5", "bbox_2d": [x1, y1, x2, y2]}  zoom into a box of a view (on a montage: inside one tile);\n'
        '    {"from": "V5", "time": "prior"}  the same address at the other time; {"from": "V5", "sequence": '
        '"FLAIR"} the same address in another sequence; {"from": "V5", "step": -2} two slices lower.')
PREDICT = ('- predict: the no-change world model shows what a current single slice should look like if nothing '
           'had changed since the prior, and the change over the current image (red-yellow = brighter than '
           'before, blue = darker). Arguments: {"question": "...", "views": ["V5"]} (1-2 ids of current slices). '
           'In view items, "at" also accepts {"peak": n}: the n-th largest increase listed in the first message. '
           "A change is a candidate, not a finding: check it on both time points.")
MEASURE = ('- measure: {"view": "V5", "target": "the enhancing lesion near the centre"} (or a tight bbox_2d) on an '
           "axial slice: returns the longest diameter, the perpendicular diameter and their product (RANO), with the "
           'outline drawn. Add "both_times": true to measure the same box on the current and the prior study and get '
           "the change of the product in percent.")
REPORT = ('- report: final answer, ends the task. Arguments: {"progression": true or false, "view": "V7", '
          '"target": "the progressing lesion" (or "bbox_2d"), "rationale": "<one sentence>"}. If progression is '
          "true, the view must be an observed slice of the current study showing the main progressing lesion.")
PROTO = ("Compare matching slices of the two studies on CT1 and FLAIR, find the enhancing lesions, measure them at "
         "both times, and look for new lesions. Before each view, state what it should settle and what you expect "
         "to see; afterward, state what it showed, expected or not.{wm} Report only when the evidence supports one "
         "answer. You have at most {max_calls} tool calls.")


def system_prompt(s: FollowSession, max_calls: int) -> str:
    conv = CONV.format(coords=s.coord_note())
    head = ("You are a neuroradiologist reading the follow-up brain MRI of a patient treated for glioblastoma, "
            "against the prior MRI. ")
    if s.cond == "oneshot":
        return (head + "Answer from the views given below. " + conv + " You must answer now with a single report "
                "call.\n\n" + CALL_FORMAT + "\n\nTool:\n" + REPORT + " A box on a montage tile fixes the location.\n\n" + RANO)
    tools = [VIEW] + ([PREDICT] if s.use_wm else []) + [LOCATE_SPEC.replace("the round nodule near the centre", "the enhancing lesion near the centre"), MEASURE, REPORT]
    wm = (" Check the largest world-model changes first, then compare the rest of the brain. Every finding must "
          "rest on observed views.") if s.use_wm else ""
    return (head + "A viewer holds both studies losslessly; you see only the views you ask for. " + conv + "\n\n"
            + CALL_FORMAT + "\n\nTools:\n" + "\n".join(tools) + "\n\n" + RANO + "\n\n"
            + PROTO.format(wm=wm, max_calls=max_calls))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ports", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=24)
    ap.add_argument("--max-ctx-images", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--out", default="runs/fu_pilot")
    a = ap.parse_args()
    mode, size = FAMILY[a.model]
    clients = itertools.cycle([AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="x", timeout=900)
                               for p in a.ports.split(",")])
    out = f"{a.out}/{a.model}/{a.cond}"
    os.makedirs(out, exist_ok=True)
    pairs = pilot_pairs(a.n, offset=a.offset)
    sem = asyncio.Semaphore(a.concurrency)
    loop = asyncio.get_running_loop()

    async def one(p):
        cid = f"{p['patient']}_{p['prior']}_{p['current']}"
        path = f"{out}/{cid}.json"
        if os.path.exists(path):
            return json.load(open(path))["score"]
        async with sem:
            fu = await loop.run_in_executor(None, load_pair, p)
            sess = FollowSession(fu, a.cond, mode, size)
            task = (f"Clinical question: compared with the prior study ({p['weeks']:g} weeks earlier), does this "
                    "follow-up MRI show progressive disease by RANO criteria? If it does, locate the main progressing "
                    "lesion on the current study.\n\nInitial views (current and prior, same slices):")
            trace = await run_episode(next(clients), a.model, sess, a.max_calls, a.max_ctx_images,
                                      system=system_prompt(sess, a.max_calls), task=task)
            sc = score_followup(fu, sess.final)
            json.dump({"case": cid, "model": a.model, "cond": a.cond, "rating": p["rating"], "score": sc,
                       "trace": trace}, open(path, "w"), indent=1)
            print(f"{a.model} {a.cond} {cid} truth={p['rating']} pred_pd={sc['pred_pd']} ev={sc['evidence_ok']} "
                  f"calls={trace['calls']} {trace['seconds']}s", flush=True)
            return sc

    t0 = time.time()
    scores = await asyncio.gather(*[one(p) for p in pairs])
    pdc = [s for s in scores if s["truth_pd"]]
    npc = [s for s in scores if not s["truth_pd"]]
    sens = sum(s["pred_pd"] is True for s in pdc) / max(1, len(pdc))
    spec = sum(s["pred_pd"] is False for s in npc) / max(1, len(npc))
    ev = sum(bool(s["evidence_ok"]) for s in pdc) / max(1, len(pdc))
    print(json.dumps({"model": a.model, "cond": a.cond, "n": len(scores), "balanced_acc": round((sens + spec) / 2, 3),
                      "sens": round(sens, 3), "spec": round(spec, 3), "pd_with_evidence": round(ev, 3),
                      "minutes": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    asyncio.run(main())
