"""Download CheXlocalize from Redivis (Stanford AIMI) into /raid/home/CAMCA/hj880/dl_chexlocalize.

Run it yourself, in your own terminal (not through Claude), with your Redivis API token in the environment:

    export REDIVIS_API_TOKEN=...            # your token; never paste it into the chat
    /raid/home/CAMCA/hj880/venvs/redivis/bin/python /raid/home/CAMCA/hj880/perr/scripts/download_chexlocalize.py

Optional: --dataset owner.name:refid  (if the default reference is not found; copy it from the dataset's
"API" tab on Redivis), --out DIR.  File tables are downloaded as files; tabular tables are saved as CSV.
"""
import argparse
import os
import sys
from pathlib import Path

import redivis

CANDIDATES = ["aimi.chexlocalize:efx9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="/raid/home/CAMCA/hj880/dl_chexlocalize")
    args = ap.parse_args()
    if not os.getenv("REDIVIS_API_TOKEN"):
        sys.exit("Set REDIVIS_API_TOKEN in this shell first.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = None
    for ref in [args.dataset] if args.dataset else CANDIDATES:
        try:
            cand = redivis.dataset(ref)
            if cand.exists():
                ds = cand
                print("dataset:", ref)
                break
            print("not found:", ref)
        except Exception as e:  # noqa: BLE001
            print("not found:", ref, "-", e)
    if ds is None:
        sys.exit("Dataset reference not found. Copy the reference from the dataset's API tab and pass --dataset.")

    for t in ds.list_tables():
        t.get()
        props = getattr(t, "properties", {}) or {}
        name = props.get("name", t.name)
        is_files = props.get("isFileIndex", False)
        print(f"table {name}: rows={props.get('numRows')} file_index={is_files}", flush=True)
        dest = out / name
        if is_files:
            dest.mkdir(exist_ok=True)
            t.download_files(path=str(dest), overwrite=False, progress=True)
        else:
            t.to_pandas_dataframe().to_csv(str(dest) + ".csv", index=False)
    print("done:", out)


if __name__ == "__main__":
    main()
