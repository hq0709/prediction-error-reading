"""Progress of an evaluation run: finished BL pieces and lesions (done.json files)."""
import glob
import json
import sys
import time

plan = json.load(open(sys.argv[1]))
work = sys.argv[2]
tot_p = sum(len(p["bl"]) for p in plan)
tot_l = sum(len(b["lesions"]) for p in plan for b in p["bl"])
done = [json.load(open(f)) for f in glob.glob(f"{work}/*/bl*/done.json")]
nl = sum(d.get("n_bl_lesions_annot", len(d.get("lesions", []))) for d in done)
nt = sum(1 for d in done for x in d.get("lesions", []) if x.get("fu_seg", {}).get("nvox", 0) > 0)
secs = sum(d.get("seconds", {}).get("total", 0) for d in done)
print(f"{time.strftime('%H:%M:%S')} pieces {len(done)}/{tot_p}, annotated lesions {nl}/{tot_l}, tracked {nt}, "
      f"process-seconds {secs:.0f} ({secs / max(nl, 1):.0f} s/lesion)")
