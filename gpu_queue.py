"""GPU job queue for a shared box: jobs are lines of queue/jobs.tsv,

    <GB needed at peak> <TAB> <name> <TAB> <names it waits for, comma-separated, or -> <TAB> <bash command, run in perr/>

started in file order once their prerequisites have finished with exit code 0, each on whichever of this project's GPUs
has the most free memory after reserving, for every job of ours running there, what it may still grow into (its
declared peak minus what its processes use now), at most --per-gpu of our jobs per card and one start per --gap
seconds. The file is re-read every loop, so jobs can be appended while the queue runs. Output goes to logs/<name>.log,
state to queue/state.json (a job whose state is not "failed" or missing is never started twice; delete its entry to
re-run it). --also-state names the state files of other queue processes: their jobs count as prerequisites and their
running jobs hold reservations too.

    nohup python gpu_queue.py > logs/gpu_queue.log 2>&1 &
"""
import argparse
import json
import os
import subprocess
import time

ROOT = "/raid/home/CAMCA/hj880/perr"
GPUS = (0, 1, 4, 5, 6, 7)          # 2 and 3 are other users' cards
ENV = {"HF_HOME": "/raid/home/CAMCA/hj880/hf_home", "TORCH_HOME": "/raid/home/CAMCA/hj880/torch_home",
       "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def smi(query, kind):
    out = subprocess.run(["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [l.split(", ") for l in out.strip().splitlines() if l.strip()]


def free_gb():
    return {int(i): (int(t) - int(u)) / 1024 for i, u, t in smi("index,memory.used,memory.total", "gpu")}


def used_by_session():
    """GB of GPU memory per (session id, GPU index): our jobs are session leaders, so a job's sid is its pid."""
    idx = {u: int(i) for i, u in smi("index,uuid", "gpu")}
    out = {}
    for uuid, pid, mem in smi("gpu_uuid,pid,used_memory", "compute-apps"):
        try:
            sid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[3])
        except (OSError, ValueError, IndexError):
            continue
        key = (sid, idx.get(uuid))
        out[key] = out.get(key, 0.0) + float(mem) / 1024
    return out


def read_jobs(path):
    jobs = []
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        need, name, deps, cmd = line.rstrip("\n").split("\t", 3)
        jobs.append({"need": float(need), "name": name, "deps": [] if deps == "-" else deps.split(","), "cmd": cmd})
    return jobs


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=f"{ROOT}/queue/jobs.tsv")
    ap.add_argument("--state", default=f"{ROOT}/queue/state.json")
    ap.add_argument("--also-state", nargs="*", default=[])
    ap.add_argument("--peaks", default=f"{ROOT}/queue/peaks.json", help="{job name: measured peak GB}, replacing declared needs in reservations")
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--big", type=float, default=25.0, help="GB: a job this large that fits nowhere holds a card")
    ap.add_argument("--gap", type=int, default=60)
    a = ap.parse_args()
    state = json.load(open(a.state)) if os.path.exists(a.state) else {}
    procs = {}
    last = 0.0
    while True:
        for name, pr in list(procs.items()):
            if pr.poll() is not None:
                state[name].update(status="done" if pr.returncode == 0 else "failed", rc=pr.returncode, end=time.time())
                print(time.strftime("%m-%d %H:%M"), name, state[name]["status"], flush=True)
                del procs[name]
        for name, s in state.items():          # jobs of an earlier queue process that are still marked running
            if s["status"] == "running" and name not in procs and not alive(s["pid"]):
                rc = f"{ROOT}/queue/rc/{name}"       # exit code written by the job's wrapper (jobs started since 10-19)
                if os.path.exists(rc):
                    code = int(open(rc).read().strip() or 1)
                    s.update(status="done" if code == 0 else "failed", rc=code, end=time.time())
                else:
                    s.update(status="lost")
        other = {}
        for p in a.also_state:
            try:
                other.update(json.load(open(p)))
            except (OSError, ValueError):
                pass
        everyone = {**other, **state}
        jobs = read_jobs(a.jobs)
        pending = [j for j in jobs if state.get(j["name"], {}).get("status") is None and j["name"] not in other]
        ready = [j for j in pending if all(everyone.get(d, {}).get("status") == "done" for d in j["deps"])]
        if ready and time.time() - last >= a.gap:
            free = free_gb()
            used = used_by_session()
            try:
                peaks = json.load(open(a.peaks))
            except (OSError, ValueError):
                peaks = {}
            running = [(n, s) for n, s in everyone.items() if s.get("status") == "running" and alive(s["pid"])]
            ours = {g: [(n, s) for n, s in running if s["gpu"] == g] for g in GPUS}
            eff = {g: free[g] - sum(max(0.0, peaks.get(n, s["need"]) - used.get((s["pid"], g), 0.0))
                                    for n, s in ours[g]) for g in GPUS}
            held = set()                        # a big job that fits nowhere holds the emptiest card for itself
            for j in ready:
                cand = [g for g in GPUS if g not in held and len(ours[g]) < a.per_gpu and eff[g] >= j["need"]]
                if not cand:
                    if j["need"] >= a.big and not held:   # only the first big job waiting holds a card
                        free_cards = [g for g in GPUS if g not in held and len(ours[g]) < a.per_gpu]
                        if free_cards:
                            held.add(max(free_cards, key=lambda i: eff[i]))
                    continue
                g = max(cand, key=lambda i: eff[i])
                log = open(f"{ROOT}/logs/{j['name']}.log", "w")
                wrapped = f"{j['cmd']}\nrc=$?\necho $rc > {ROOT}/queue/rc/{j['name']}\nexit $rc"
                pr = subprocess.Popen(["bash", "-c", wrapped], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(g)}, start_new_session=True)
                procs[j["name"]] = pr
                state[j["name"]] = {"status": "running", "gpu": g, "need": j["need"], "start": time.time(), "pid": pr.pid}
                print(time.strftime("%m-%d %H:%M"), f"start {j['name']} on GPU {g} ({eff[g]:.0f} GB free after reservations)",
                      flush=True)
                last = time.time()
                break
        json.dump(state, open(a.state + ".tmp", "w"), indent=1)
        os.replace(a.state + ".tmp", a.state)
        time.sleep(20)


if __name__ == "__main__":
    main()
