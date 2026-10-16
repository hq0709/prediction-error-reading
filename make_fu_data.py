"""Follow-up reader data: candidate strips at the no-change world model's top-k error peaks, one file per split.

    python make_fu_data.py pilot            # the 60 pilot pairs used by the agent runs
    python make_fu_data.py pool             # every other pair of patients outside the pilot (training pool)
    python make_fu_data.py pilot tumour     # candidates from the tumour-space world model instead
"""
import json
import sys
from multiprocessing import Pool

from perr.fu_candidates import QUESTION, fu_strips
from perr.lumiere import load_pair, pairs, pilot_pairs

OUT = "/raid/home/CAMCA/hj880/perr/data/fu_reader"
PROPOSAL = "intensity"


def work(p):
    try:
        fu = load_pair(p)
        rows = []
        for r in fu_strips(fu, 6 if PROPOSAL == "intensity" else 4, PROPOSAL):
            path = f"{OUT}/img/{fu.case_id}_{PROPOSAL[:3]}{r['peak']}.png"
            r.pop("image").save(path)
            rows.append({**r, "uid": fu.case_id, "patient": p["patient"], "rating": p["rating"],
                         "label": int(p["rating"] == "PD"), "image": path, "question": QUESTION,
                         "source": "wm", "small": False})
        return rows
    except Exception as e:
        return [{"uid": p["patient"], "error": repr(e)[:200]}]


if __name__ == "__main__":
    split = sys.argv[1]
    PROPOSAL = sys.argv[2] if len(sys.argv) > 2 else "intensity"
    pilot = pilot_pairs(60)
    if split == "pilot":
        ps = pilot
    else:
        held = {p["patient"] for p in pilot}
        ps = [p for p in pairs() if p["patient"] not in held]
    with Pool(16) as pool:
        res = pool.map(work, ps, chunksize=1)
    rows = [r for rs in res for r in rs if "error" not in r]
    errs = [r for rs in res for r in rs if "error" in r]
    with open(f"{OUT}/{split}{'' if PROPOSAL == 'intensity' else '_' + PROPOSAL}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{split}: {len(ps)} pairs, {len(rows)} strips, PD pairs {sum(p['rating'] == 'PD' for p in ps)}, "
          f"errors {len(errs)} {errs[:2]}")
