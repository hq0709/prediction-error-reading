# Baselines on our benchmark: conventions

The task, for every case: is there a target lesion, where is it, and how big is it? There are two datasets.

**LUNA16 lung nodules.**
- Target: a nodule >= 6 mm.
- Cases: 200 scans, in `eval_luna_fresh1.txt` (100) and `eval_luna_fresh2.txt` (100). Every scan has at least one nodule >= 6 mm.
- Volumes: `cache/luna/<uid>.vol.npy`, int16 HU. Spacing is `np.load(cache/luna/<uid>.meta.npy, allow_pickle=True).item()["spacing"]` = (sz, sy, sx) in mm.
- Originals: `/raid/home/CAMCA/hj880/data_ext/luna16/subset*/<uid>.mhd`.

**MSD Task03 liver tumours.**
- Target: a tumour >= 10 mm.
- Cases: 53 scans, in `eval_liver.txt`. Every case has at least one target tumour.
- Volumes: `cache/liver/<case>.vol.npy`, int16 HU. Spacing is in `cache/liver/<case>.meta.npy`, the same way.
- Originals: `/raid/home/CAMCA/hj880/msd/Task03_Liver/imagesTr/<case>.nii.gz`.

**Array convention.** Every cached array is indexed `[z, y, x]` in LPS order:
- x grows toward the patient's left;
- y grows toward posterior;
- z grows toward superior.

The arrays are the originals reoriented with `SimpleITK.DICOMOrient(img, "LPS")` and then `GetArrayFromImage`. If a method needs NIfTI input, write it from the cached array with an explicit affine, or reorient the original file. Either way, map every output back to this index space.

**Output.** Write `results/sota/<method>_luna.jsonl` and/or `results/sota/<method>_liver.jsonl`, one JSON line per case:

    {"case": "<uid or liver_N>", "present": true, "zyx": [z, y, x], "diameter_mm": 12.3,
     "score": 0.87, "candidates": [{"zyx": [...], "diameter_mm": ..., "score": ...}, ...], "raw": "<model text, if any>"}

- `zyx` is a continuous voxel index into the cached array.
- `diameter_mm` is the reported size. For a mask, give the equivalent-sphere diameter (6V/π)^(1/3).
- If the model says "absent" or gives no location, write `"present": false` and `"zyx": null`.
- One answer per case. If a method outputs several lesions, the answer is the one the method itself ranks first: by its confidence if it has one, otherwise the largest. Keep the others in `candidates`.

**Scoring.** `python score_sota.py <method>`, run from the project root with the venv `/raid/home/CAMCA/hj880/venvs/perr`. It applies exactly the rules used for our own reports:
- LUNA16: a hit is a point within radius + 3 mm of a nodule >= 6 mm.
- Liver: a hit is a point inside a target tumour mask grown by 3 mm.

**Rules.**
1. Run each method as published: official code, official weights, the documented preprocessing and inference. Use the method's own prompt format, and say "lung nodule" or "liver tumor" in the method's own vocabulary. For VLMs, use one fixed prompt per dataset that asks for the largest lesion of the target size, with a machine-readable location and size. Record the prompt in `sota/<method>/PROMPT.txt`.
2. Never read labels or annotations of the evaluation cases. Never tune anything on them: prompts, thresholds and component selection are fixed before the first eval run.
   - Tuning and smoke tests may use training-pool scans: `results/train_uids.txt` for LUNA16; for the liver, any MSD Task03 case not in `eval_liver.txt`.
3. Coordinate check, before any eval run. Write a synthetic volume (the cached array of one pool scan, with a 10 mm sphere of +300 HU at a known index) through the method's own I/O path and the inverse mapping. Confirm the index comes back within 1 voxel, and record this in `sota/<method>/NOTES.md`.
4. Put code in `sota/<method>/` and environments in `/raid/home/CAMCA/hj880/venvs/sota_<method>`. Do not modify any other project file. Models are in `HF_HOME=/raid/home/CAMCA/hj880/hf_home` (`hub/`) and `/raid/home/CAMCA/hj880/hf_cache`. Code clones are in `/raid/home/CAMCA/hj880/ext_src/<name>`.
5. GPUs are shared with other users. Use only the GPUs you were assigned. Check `nvidia-smi` for free memory first, and never kill a process you did not start.
6. Record in `sota/<method>/NOTES.md`:
   - versions;
   - deviations from the official recipe and why;
   - runtime;
   - failures (cases with no answer, crashes);
   - known training-data overlap with LUNA16/LIDC-IDRI or MSD Task03.
