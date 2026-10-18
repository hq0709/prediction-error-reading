"""Scoring queue for focal readers on a shared box: each job (a reader's saved adapter scoring one set of addresses) is
started, once its adapter is ready, on whichever of this project's GPUs has the most free memory (at least --need GB),
one start per minute so the memory of the last start shows; when every shard of a set is scored, the shards are joined.

    python score_queue.py --runs cxr_focal_wmu_mil cxr_focal_wmu_milpos cxr_focal_wmu_milgate --ablations
"""
import argparse
import json
import os
import subprocess
import time

ROOT = "/raid/home/CAMCA/hj880/perr"
PY = "/raid/home/CAMCA/hj880/venvs/grl/bin/python"
GPUS = (0, 1, 4, 5, 6, 7)          # 2 and 3 are other users' cards
UNION = ["wmu/nih_test", "wmu/mimic_validate_s0", "wmu/mimic_validate_s1", "wmu/mimic_test_s0", "wmu/mimic_test_s1",
         "wmu/mimic_test_s2"]
JOINS = {"wmu_mimic_validate": ["wmu_mimic_validate_s0", "wmu_mimic_validate_s1"],
         "wmu_mimic_test": ["wmu_mimic_test_s0", "wmu_mimic_test_s1", "wmu_mimic_test_s2"]}


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(i): (int(t) - int(u)) / 1024 for i, u, t in (l.split(", ") for l in out.strip().splitlines())}


def adapter_ready(run):
    log = f"{ROOT}/logs/tr_{run}.log"
    if not os.path.exists(log):
        return os.path.exists(f"{ROOT}/runs_reader/{run}/lora_best")      # a finished reader
    return "epoch 1 validation" in open(log).read() and time.time() - os.path.getmtime(log) > 120


def shard(proposer, split, n):
    """data/cxr_focal/<proposer>/mimic_<split>.jsonl in n shards by image; returns the shard set names."""
    src = f"{ROOT}/data/cxr_focal/{proposer}/mimic_{split}.jsonl"
    names = [f"{proposer}/mimic_{split}_s{j}" for j in range(n)]
    if not all(os.path.exists(f"{ROOT}/data/cxr_focal/{s}.jsonl") for s in names):
        rows = open(src).readlines()
        keys = sorted({json.loads(l)["key"] for l in rows})
        part = {k: i * n // len(keys) for i, k in enumerate(keys)}
        outs = [open(f"{ROOT}/data/cxr_focal/{s}.jsonl", "w") for s in names]
        for l in rows:
            outs[part[json.loads(l)["key"]]].write(l)
        for o in outs:
            o.close()
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--ablations", action="store_true", help="also cxr_focal on the random / whole-radiograph MIMIC views")
    ap.add_argument("--need", type=float, default=26.0)
    a = ap.parse_args()
    env = {**os.environ, "HF_HOME": "/raid/home/CAMCA/hj880/hf_home", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
    jobs = [(r, s) for r in a.runs for s in UNION]
    pending_abl = ["random", "wmr"] if a.ablations else []
    joins = [(r, JOINS) for r in a.runs]
    running = []
    while jobs or pending_abl or running:
        for p in list(pending_abl):
            if os.path.exists(f"{ROOT}/data/cxr_focal/{p}/mimic_test.jsonl"):
                names = shard(p, "validate", 2) + shard(p, "test", 3)
                jobs += [("cxr_focal", s) for s in names]
                joins.append(("cxr_focal", {f"{p}_mimic_validate": [f"{p}_mimic_validate_s{j}" for j in range(2)],
                                            f"{p}_mimic_test": [f"{p}_mimic_test_s{j}" for j in range(3)]}))
                pending_abl.remove(p)
        running = [(j, pr) for j, pr in running if pr.poll() is None]
        ready = [j for j in jobs if adapter_ready(j[0])]
        if ready:
            free = free_gb()
            g = max(GPUS, key=lambda i: free[i])
            if free[g] >= a.need:
                run, s = ready[0]
                log = open(f"{ROOT}/logs/score_{run}_{s.replace('/', '_')}.log", "w")
                pr = subprocess.Popen([PY, "train_cxr_focal_reader.py", "--out", f"runs_reader/{run}", "--score-only", s],
                                      cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": str(g)}, stdout=log, stderr=log)
                running.append(((run, s), pr))
                jobs.remove((run, s))
                print(time.strftime("%H:%M"), f"{run} {s} on GPU {g} ({free[g]:.0f} GB free)", flush=True)
                time.sleep(60)
                continue
        time.sleep(30)
    for run, spec in joins:
        R = f"{ROOT}/results/cxr_focal/{run}"
        for whole, parts in spec.items():
            if all(os.path.exists(f"{R}/{p}.jsonl") for p in parts):
                with open(f"{R}/{whole}.jsonl", "w") as f:
                    for p in parts:
                        f.write(open(f"{R}/{p}.jsonl").read())
            else:
                print(f"missing shards for {run} {whole}", flush=True)
    print("QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
