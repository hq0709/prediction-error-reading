"""Choose MAD-AD's free inference settings on the verifier's held-out tenth of ChestX-Det train (302 images, 165 focal
boxes), by focal-box recall@45 of the map's candidates (dev_recall.py: the peak finding, suppression and thorax search
mask of cxr_focal_data.py --proposer map:<name> --k 45).

Candidates: every sampling setting of infer_madad.SWEEP (raw |difference| maps in cache/uad_dev/madad/<setting>/)
x every post-processing in POSTS. Rule fixed before seeing any result: highest recall@45; ties go to the setting with
the fewest departures from the repo's defaults (10 DDIM steps from the top, eta 0.5, gamma 0.25, sigma 10 px on the
512 canvas = 5 px at the repo's 256), then to the higher recall@30.

    python -I -B sota/madad/select_madad.py        # writes sota/madad/dev_recall.json and sota/madad/chosen.json
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.append(HERE)
SWEEP = [(10, 10, 0.5), (10, 7, 0.5), (10, 5, 0.5), (10, 3, 0.5), (5, 5, 0.5), (5, 3, 0.5), (2, 2, 0.5), (2, 1, 0.5),
         (10, 10, 0.0), (10, 10, 1.0)]                       # = infer_madad.SWEEP (not imported: torch-free here)
POSTS = ["g0.25_s10", "g0.25_s5", "g0.25_s20", "g0.2_s10", "s10", "s5", "s20", "id"]
OFFICIAL = {"steps": 10, "start": 10, "eta": 0.5, "gamma": 0.25, "sigma": 10.0}


def setting_name(n, s, eta):
    return f"n{n}_s{s}_eta{eta:g}"


def parse(variant):
    setting, post = variant.split(":")
    n, s, eta = setting.split("_")
    g, sig = (post[1:].split("_s") if post.startswith("g") else (0.0, post[1:] if post != "id" else 0.0))
    return {"steps": int(n[1:]), "start": int(s[1:]), "eta": float(eta[3:]), "gamma": float(g), "sigma": float(sig)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", default="cache/uad_dev/madad")
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    dirs = [f"{a.dev_dir}/{setting_name(*st)}:{','.join(POSTS)}" for st in SWEEP]
    out = f"{a.out_dir}/dev_recall.json"
    subprocess.run([sys.executable, "-I", "-B", f"{HERE}/dev_recall.py", "--keys", f"{HERE}/keys_dev.txt", "--maps", *dirs,
                    "--out", out], check=True)
    table = json.load(open(out))["table"]

    def departures(v):
        p = parse(v)
        return sum(p[k] != OFFICIAL[k] for k in OFFICIAL)

    best = max(table, key=lambda v: (round(table[v]["recall@45"], 6), -departures(v), round(table[v]["recall@30"], 6)))
    off = f"{setting_name(10, 10, 0.5)}:g0.25_s10"
    chosen = {"variant": best, **parse(best), "dev": table[best], "official_variant": off, "official_dev": table[off]}
    json.dump(chosen, open(f"{a.out_dir}/chosen.json", "w"), indent=1)
    print("chosen:", json.dumps(chosen, indent=1), flush=True)


if __name__ == "__main__":
    main()
