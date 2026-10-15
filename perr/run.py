"""Run one model x one condition over the pilot cases; one JSON per episode.

    python -m perr.run --model qwen3vl32b --ports 8001,8002 --cond harness_wm --n 50 --out runs/pilot
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import time

from openai import AsyncOpenAI

from .agent import run_episode
from .harness import Session
from .score import score_case
from .study import load_luna, pilot_cases

FAMILY = {  # display size and coordinate convention each model family was trained with
    "qwen3vl32b": ("norm1000", 512),
    "qwen25vl72b": ("pixel", 504),
    "qwen25vl32b": ("pixel", 504),
    "internvl3_78b": ("norm1000", 448),
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ports", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=20)
    ap.add_argument("--max-ctx-images", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--out", default="runs/pilot")
    a = ap.parse_args()
    mode, size = FAMILY[a.model]
    clients = itertools.cycle([AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="x", timeout=900)
                               for p in a.ports.split(",")])
    out = f"{a.out}/{a.model}/{a.cond}"
    os.makedirs(out, exist_ok=True)
    cases = pilot_cases(a.offset + a.n)[a.offset:]
    sem = asyncio.Semaphore(a.concurrency)
    loop = asyncio.get_running_loop()

    async def one(uid):
        path = f"{out}/{uid}.json"
        if os.path.exists(path):
            return json.load(open(path))["score"]
        async with sem:
            study = await loop.run_in_executor(None, load_luna, uid)
            sess = await loop.run_in_executor(None, lambda: Session(study, a.cond, mode, size))
            trace = await run_episode(next(clients), a.model, sess, a.max_calls, a.max_ctx_images)
            sc = score_case(study, sess.final)
            rec = {"case": uid, "model": a.model, "cond": a.cond, "score": sc, "trace": trace,
                   "config": {"coord_mode": mode, "size": size, "max_calls": a.max_calls,
                              "max_ctx_images": a.max_ctx_images, "temperature": 0.0}}
            json.dump(rec, open(path, "w"), indent=1)
            print(f"{a.model} {a.cond} {uid[-12:]} hit6={sc['hit6']} dist={sc['dist_mm']} calls={trace['calls']} "
                  f"{trace['seconds']}s", flush=True)
            return sc

    t0 = time.time()
    scores = await asyncio.gather(*[one(u) for u in cases])
    n = len(scores)
    agg = {k: round(sum(bool(s[k]) for s in scores) / n, 3) for k in ("answered", "hit6", "hit_largest", "hit_any")}
    print(json.dumps({"model": a.model, "cond": a.cond, "n": n, **agg, "minutes": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    asyncio.run(main())
