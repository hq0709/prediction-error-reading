"""Print one field of the setting fixed by choose_setting.py (domain, sigma, raw, thres_file) for queue command lines."""
import json
import sys

print(json.load(open("/raid/home/CAMCA/hj880/perr/sota/itermask2/chosen.json"))[sys.argv[1]])
