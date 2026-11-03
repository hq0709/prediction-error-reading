"""Fix IterMask3D's free inference setting, gamma (the derivative level of the fitted threshold curve at which the mask
stops shrinking; official defaults 0.01 in argumentlib.py and 0.05 in the README), on the world-model dev split: the
largest dev recall@10, ties broken by recall@5, then recall@1, then the value closest to the argumentlib default 0.01.
Writes sota/itermask3d/chosen.json, keeps the dev maps of the chosen gamma as cache/itermask3d/<uid>.npy and removes the
others.

    python sota/itermask3d/choose_gamma.py dev
"""
import json
import math
import os
import sys

ROOT = "/raid/home/CAMCA/hj880/perr"
RUNS = f"{ROOT}/sota/itermask3d/runs"
MAPS = f"{ROOT}/cache/itermask3d"


def main():
    out = sys.argv[1]
    res = {float(g): v for g, v in json.load(open(f"{RUNS}/dev_recall_{out}.json")).items()}
    key = lambda g: (res[g]["recall"]["10"], res[g]["recall"]["5"], res[g]["recall"]["1"], -abs(math.log(g / 0.01)))
    g = max(res, key=key)
    json.dump({"gamma": g, "dev_recall": res[g]["recall"], "all": res}, open(f"{ROOT}/sota/itermask3d/chosen.json", "w"),
              indent=1)
    print(f"chosen gamma {g:g}: dev recall {res[g]['recall']}", flush=True)
    dev = json.load(open(f"{ROOT}/runs_wm/ctx_free/split.json"))["dev"]
    for u in dev:
        for h in res:
            p = f"{MAPS}/{u}.g{h:g}.npy"
            if os.path.exists(p):
                if h == g:
                    os.replace(p, f"{MAPS}/{u}.npy")
                else:
                    os.remove(p)


if __name__ == "__main__":
    main()
