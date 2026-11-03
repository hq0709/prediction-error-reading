"""Print the gamma fixed on the dev split by choose_gamma.py (for queue command lines)."""
import json

print(json.load(open("/raid/home/CAMCA/hj880/perr/sota/itermask3d/chosen.json"))["gamma"])
