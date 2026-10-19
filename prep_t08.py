"""MSD Task08 as extra liver training data: TotalSegmentator liver mask, liver cache and 1 mm grid per case
(perr/liver_t08.py), then the usable list.

    TOTALSEG_HOME_DIR=/raid/home/CAMCA/hj880/totalseg_home venvs/totalseg/bin/python prep_t08.py --shard 0 3
    venvs/totalseg/bin/python prep_t08.py --list
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perr.context_wm import build_iso  # noqa: E402
from perr.liver_t08 import cases, eligibility, load_case, segment, usable_cases  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
ap.add_argument("--list", action="store_true")
a = ap.parse_args()
if a.list:
    for c in usable_cases():
        build_iso(c, "liver_t08")          # cases made usable after their first pass
    ok = usable_cases(write=True)
    bad = {c: eligibility(c) for c in cases() if c not in ok}
    print(f"{len(ok)} usable of {len(cases())}; left out: {bad}")
else:
    for c in cases()[a.shard[0]::a.shard[1]]:
        try:
            segment(c)
            s = load_case(c)
            if not eligibility(c):
                build_iso(c, "liver_t08")
            print(c, len(s.nodules), eligibility(c), flush=True)
        except Exception as e:          # e.g. a GPU shared with other jobs ran out of memory: left for a later pass
            print(c, "FAILED", repr(e)[:200], flush=True)
