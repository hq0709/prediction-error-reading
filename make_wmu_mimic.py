"""Union address rows (both world models, taken in turn with the 512 model first, repeats within 16 px dropped: the rule
of cxr_focal_data.py --proposer wmu and score_cxr_focal_mimic.union) for the MIMIC evaluation splits, built from the
two world models' rows, so a reader trained on union addresses scores each place once.

    python make_wmu_mimic.py mimic_validate mimic_test
"""
import json
import sys

D = "data/cxr_focal"


def load(path):
    by = {}
    for l in open(path):
        r = json.loads(l)
        if r["src"] == "wm" and r["rank"] < 30:
            by.setdefault(r["key"], []).append(r)
    return {k: sorted(v, key=lambda r: r["rank"]) for k, v in by.items()}


def main():
    for s in sys.argv[1:]:
        a, b = load(f"{D}/wm/{s}.jsonl"), load(f"{D}/wmh/{s}.jsonl")
        keys = sorted(a)
        n = 0
        with open(f"{D}/wmu/{s}.jsonl", "w") as f:
            for key in keys:
                xa, xb, kept = a[key], b.get(key, []), []
                for i in range(max(len(xa), len(xb))):
                    for x in (xa, xb):
                        if i < len(x) and all(max(abs(x[i]["r"] - q["r"]), abs(x[i]["c"] - q["c"])) > 16 for q in kept):
                            kept.append(x[i])
                for j, r in enumerate(kept[:60]):
                    f.write(json.dumps({**r, "i": j, "rank": j}) + "\n")
                    n += 1
        print(f"{s}: {len(keys)} images, {n} union addresses ({len(set(keys) - set(b))} without 1024 rows)")


if __name__ == "__main__":
    main()
