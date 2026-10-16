"""External liver test (HCC-TACE-Seg): reader strips at the liver world models' top-10 addresses, for readers and world
models trained on MSD Task03 only (no case of this set is used for any choice).

    python make_liver_ext_data.py iso                    # 1 mm cache for the learned liver model
    (then: bash run_ctx_infer.sh runs_wm/liver_nll/final.pt lnll sota/eval_liver_ext.txt "<gpus>" liver_ext)
    python make_liver_ext_data.py strips liver           # local-mean world model
    python make_liver_ext_data.py strips fused           # local-mean + learned, reciprocal-rank fused (as for MSD)
    LIVER_TAG=lcrlm python make_liver_ext_data.py strips fused    # with the liver model trained on MSD + CRLM
"""
import json
import os
import sys
from multiprocessing import Pool

from perr.context_wm import ContextPrior, build_iso, fuse_rrf
from perr.liver import QUESTION, LiverPrior, liver_strips
from perr.liver_ext import CACHE, label_at, load_case

CASES = open("sota/eval_liver_ext.txt").read().split()
PROPOSAL = "liver"
TAG = os.environ.get("LIVER_TAG", "lnll")


def top(s, k=10):
    prior = LiverPrior(cache=CACHE)
    if PROPOSAL == "fused":
        learned = ContextPrior(tag=TAG, sigma=3.0, organ="liver_ext").top(s, 50, nms_mm=15.0)
        return fuse_rrf([prior.top(s, 50), learned], s.spacing, c=10.0, nms_mm=15.0)[:k]
    return prior.top(s, k)


def work(case):
    s = load_case(case)
    out_dir = f"data/liver_ext_reader_{PROPOSAL}{'' if TAG == 'lnll' else '_' + TAG}"
    rows = []
    for r in liver_strips(s, top(s), label_fn=label_at):
        path = os.path.abspath(f"{out_dir}/img/{case}_wm{r['peak']}.png")
        r.pop("image").save(path)
        rows.append({**r, "uid": case, "image": path, "source": "wm", "small": r["label"] == "small",
                     "label": int(r["label"] == "big"), "question": QUESTION})
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "iso":
        with Pool(24) as p:
            p.starmap(build_iso, [(c, "liver_ext") for c in CASES], chunksize=1)
        print("iso done", len(CASES))
    else:
        PROPOSAL = sys.argv[2]
        out_dir = f"data/liver_ext_reader_{PROPOSAL}{'' if TAG == 'lnll' else '_' + TAG}"
        os.makedirs(f"{out_dir}/img", exist_ok=True)
        with Pool(24) as p:
            rows = [r for rs in p.map(work, CASES, chunksize=1) for r in rs]
        with open(f"{out_dir}/eval.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{PROPOSAL}: {len(rows)} rows, {sum(r['label'] for r in rows)} positive, {len(CASES)} cases")
