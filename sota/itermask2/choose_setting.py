"""Fix IterMask2's free inference settings on radiographs on the verifier's held-out tenth of ChestX-Det train (302
images), by focal-box recall@45 (dev_recall.py: thorax search domain, 16 px suppression, K = 45):
  brain mask at test time   the radiograph (as in training) or the thorax search domain (the validation threshold is
                            recomputed with masks drawn in the same region)
  smoothing                 Gaussian sigma in {0, 2, 4, 8} px on the 512 canvas before the address search
                            (0 = the official squared-error map unchanged)
The largest box recall@45 wins; ties go to image recall@45, then the radiograph mask, then the smaller sigma.
Writes sota/itermask2/chosen.json; the queue job imk2_test then maps the test images with the chosen setting and writes
cache/cxr_maps/itermask2/<key>.npy for the held-out tenth and the test images.

    python sota/itermask2/choose_setting.py
"""
import json
import subprocess
import sys

ROOT = "/raid/home/CAMCA/hj880/perr"
DOMAINS = ("radiograph", "thorax")
SIGMAS = (0, 2, 4, 8)


def main():
    out = f"{ROOT}/sota/itermask2/dev_recall_variants.jsonl"
    open(out, "w").close()
    props = [f"map:itermask2_{d}_s{s:g}" for d in DOMAINS for s in SIGMAS]
    subprocess.run([sys.executable, f"{ROOT}/sota/itermask2/dev_recall.py", *props, "--out", out], check=True)
    res = {json.loads(l)["proposer"]: json.loads(l)["recall"] for l in open(out)}
    key = lambda v: (res[f"map:itermask2_{v[0]}_s{v[1]:g}"]["45"]["box"], res[f"map:itermask2_{v[0]}_s{v[1]:g}"]["45"]["image"],
                     -DOMAINS.index(v[0]), -v[1])
    d, s = max([(d, s) for d in DOMAINS for s in SIGMAS], key=key)
    ch = {"domain": d, "sigma": s, "raw": "itermask2_raw" if d == "radiograph" else "itermask2_raw_thorax",
          "thres_file": f"{ROOT}/sota/itermask2/runs/thres_256.json" if d == "radiograph" else
          f"{ROOT}/sota/itermask2/runs/thres_256_thorax.json",
          "dev_recall45": res[f"map:itermask2_{d}_s{s:g}"]["45"], "all": res}
    json.dump(ch, open(f"{ROOT}/sota/itermask2/chosen.json", "w"), indent=1)
    print(f"chosen: brain mask {d}, sigma {s:g}; held-out recall@45 {ch['dev_recall45']}", flush=True)


if __name__ == "__main__":
    main()
