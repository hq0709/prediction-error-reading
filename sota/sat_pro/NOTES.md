# SAT-Pro on our benchmark

SAT (Zhao et al., "Large-vocabulary segmentation for medical images with text prompts", npj Digital Medicine 8:566,
2025), SAT-Pro variant: a text-promptable 3D segmentation model (UNET-L Maskformer + knowledge-enhanced text encoder).

## What was run

| item | value |
|---|---|
| code | `ext_src/SAT` = github.com/zhaoziheng/SAT, commit `5676be5` (2025-12-29), unmodified (git status clean) |
| weights | HF `zzh99/SAT` snapshot `56d13778`: `Pro/SAT_Pro.pth`, `Pro/text_encoder.pth` |
| text backbone | `FremyCompany/BioLORD-2023-C` snapshot `4bb5abd` (architecture/tokenizer; its weights are overwritten by `text_encoder.pth`) |
| prompts | LUNA16 `"lung nodule"`, liver `"liver tumor"`, modality `"ct"` (see `PROMPT.txt`); automatic mode (SAT has no point prompts) |
| runner | `run_sat.py` (eval), `run_eval.sh` (the one eval invocation), `common.py` (NIfTI I/O + mask rule), `coord_check.py`, `smoke_check.py` |
| outputs | `results/sota/sat_pro_luna.jsonl` (200 lines), `results/sota/sat_pro_liver.jsonl` (53 lines) |
| GPU | GPU 0 only (`CUDA_VISIBLE_DEVICES=0`), shared with another user's 10 GB job |

Pipeline per case: cached LPS array → uncompressed NIfTI with the explicit RAS affine `diag(-sx, -sy, sz)`
(`common.write_nifti`) → SAT's own `data/inference_dataset.py` (`load_image`: LoadImaged, EnsureChannelFirstd,
Orientationd RAS, Spacingd 1×1×3 mm bilinear, CropForegroundd, ToTensord, CT clip [-500, 1000] + z-score;
`split_3d` into 288×288×96 patches at 50 % overlap) → SAT-Pro + text encoder → the loop of
`evaluate/inference_engine.py:inference`, copied verbatim (autocast, sigmoid, Gaussian-weighted accumulation,
`prediction / accumulation`, threshold 0.5) → mask → answer rule below on the processed grid → component centroids
mapped to the cached index through the affine that MONAI tracks for the processed grid.

## Mask → answer rule (fixed before the eval run, identical for VISTA3D)

`common.mask_to_answer`: 3D connected components of the 0.5-thresholded mask (26-connectivity); point = component
centroid (mean voxel index) mapped to the cached `[z, y, x]` index; size = equivalent-sphere diameter (6V/π)^(1/3)
with V = voxel count × 3 mm³ (1×1×3 mm grid); score = mean predicted probability (the accumulated sigmoid map)
over the component; answer = highest score (ties → larger); `candidates` = top 10 by score, `candidates[0]` is the
answer; empty mask → `present: false`, `zyx: null`. No size filter, no threshold other than SAT's own 0.5.
`raw` holds diagnostics as a JSON string (prompt, component count, mask voxels, max probability, grid, timings).

## Environment (`/raid/home/CAMCA/hj880/venvs/sota_sat`, full list in `pip_freeze.txt`, recipe in `setup_env.sh`)

Python 3.10.22, torch 2.1.2+cu121, numpy 1.21.5, monai 1.1.0, transformers 4.21.3, tokenizers 0.12.1,
nibabel 4.0.2, einops 0.6.1, positional_encodings 6.0.1, scipy 1.9.3, pandas 1.5.3, SimpleITK 2.3.1, and the
customised `dynamic-network-architectures` 0.2 shipped inside the SAT repo (installed from a copy, not editable).
All version pins of the SAT README ("key requirements") are met; `requirements.txt` itself is unpinned.

## Deviations from the official recipe

1. **torch 2.1.2+cu121** instead of an unspecified `torch>=1.10`: H100 (sm_90) needs a CUDA ≥ 11.8 build; 2.1.2 is the
   oldest CUDA 12 wheel and runs with the pinned numpy 1.21.5.
2. **BioLORD-2023-C read from a local snapshot.** The pinned transformers 4.21.3 cannot download from today's
   Hugging Face hub (it does not follow the hub's relative redirects), so `Knowledge_Encoder`'s default checkpoint
   argument is pointed at the local copy of the same repo. Only the config/tokenizer matter: `text_encoder.pth`
   overwrites every weight (only the deterministic `position_ids` buffer is not in the checkpoint).
3. **Single process instead of `torchrun`**, world size 1 with the same DDP/SyncBatchNorm wrappers;
   `--batchsize_3d 1` instead of the README example's 2 to keep headroom on the shared GPU (patches are processed
   independently with instance norm, so outputs do not depend on it); our loader iterates cases in order instead of
   the DistributedSampler's shuffled order.
4. **Output mapping added.** The official engine thresholds the probability map and saves masks in the resampled,
   cropped grid with an identity affine; it never maps back to the input. We keep the probability map (for the
   component score) and map centroids back with the processed-grid affine, captured by wrapping SAT's
   `Normalization` (the one point where the MetaTensor still carries it).
5. Checkpoint loading: all model parameters load; the checkpoint additionally contains `mid_mask_embed_proj.*`
   (deep-supervision heads), unused at inference with the official default `--deep_supervision False`.
6. `meta.npy` files were pickled under numpy 2; `common.load_spacing` aliases `numpy._core` to `numpy.core` to read
   the spacing under numpy 1.21. In this copy of `common.py` the alias stays installed after reading (harmless here:
   torch is already imported); the VISTA3D copy removes it again because a later `import torch` segfaults there.
   The mask rule and NIfTI code are byte-identical between the two copies.

## Coordinate check (README rule 3) — PASS

`coord_check.py` on pool scan `…100953483028192176989979435275` (spacing 2.5×0.703×0.703 mm, shape 133×512×512):
10 mm sphere of +300 HU at cached index (83, 240, 352), deepest lung voxel of the pool scan's lung mask. Both the
original and the synthetic array went through `write_nifti` and SAT's `load_image`; the processed-image difference
was reduced with the same rule and affine back-mapping as `run_sat.py`.
Recovered (83.30, 240.02, 351.93): error (+0.30, +0.02, −0.07) voxels = (0.75, 0.01, −0.05) mm. Processed grid
360×269×111 at 1×1×3 mm.
End-to-end, SAT-Pro itself segments the synthetic sphere as "lung nodule": component at (82.99, 239.96, 351.97),
error < 0.05 voxel, equivalent diameter 10.18 mm (the real 10.4 mm nodule of that scan stays rank 1).

## Smoke tests (training-pool cases only)

* LUNA16 pool scans `…979435275`, `…764325545`, `…502185026`: the rank-1 component lies 0.4, 1.1 and 0.6 mm from an
  annotated nodule centre (10.4, 12.7, 9.2 mm); in the third scan ranks 1–3 hit all three annotated nodules
  (9.2, 6.1, 10.3 mm). Equivalent diameters 9.0–9.6 mm.
* Liver pool cases `liver_0`, `liver_5`, `liver_14`: the rank-1 component lies inside a tumour in all three
  (17, 9, 13 mm tumours; predicted 16.7, 9.8, 12.6 mm).
  Pool cases may be in SAT's own training split, so these only confirm that I/O and orientation are right.

## Runtime

* LUNA16: 200 scans in 520 s (2.6 s/scan; model 2.4 s/scan mean) on a shared H100.
* Liver: 53 cases in 252 s (4.8 s/case; model 4.4 s/case mean).
* Setup (imports, first NIfTI files, 6 GB checkpoint): 27 s (LUNA16) and 36 s (liver) per invocation.

## Failures

* No crashes; every eval case has a line.
* Empty masks (→ `present: false`): LUNA16 27 / 200 (13.5 %; median max-probability in those scans 0.15);
  liver 1 / 53 (max probability in that case 0.04).
* No centroid fell outside the cached array (no clipping).

## Training-data overlap

SAT is trained on SAT-DS (72 public datasets), split per dataset by patient into train/test; the split files are
published in github.com/zhaoziheng/SAT-DS `data/split_json/`. Checked against our case lists:

* **LUNA16** is a SAT-DS dataset, but only with lung masks (classes left lung, right lung, trachea, lung; SAT-DS
  `loader.py`), no nodule labels. **165 / 200** of our eval scans are in SAT's LUNA16 *train* split
  (fresh1 80/100, fresh2 85/100), 35 in its test split.
* SAT's **"lung nodule"** class comes from **LNDb** (236 CTs from Portugal, not LIDC). SAT-DS ships a `LIDC_IDRI`
  loader with a "lung nodule" class, but the SAT-DS README marks LIDC-IDRI as not included (no split file) and it is
  absent from SAT's 72-dataset training config (`data/dataset_config/72.json`). "lung tumor" comes from MSD Task06.
* **MSD Task03 (MSD_Liver)** is in SAT-DS with classes liver and liver tumor: **45 / 53** of our liver eval cases
  are in SAT's *train* split (tumour masks seen in training), 8 are in its test split
  (liver_2, 9, 21, 24, 49, 85, 102, 110).

## Score (one run of `score_sota.py sat_pro vista3d`, after all outputs and these notes were final)

    sat_pro  fresh1 hit 0.66 largest 0.51 diamMAE 1.5 answered 100 | fresh2 hit 0.64 largest 0.53 diamMAE 1.5 answered 100
    sat_pro  liver hit 0.94 (n=53, answered 53)

The liver number is not a held-out result: 45 of the 53 cases are in SAT's training split with tumour masks.

## External liver test set: HCC-TACE-Seg (93 cases)

Same pipeline, prompt ("liver tumor"), threshold and mask → answer rule as for MSD liver; nothing was tuned or
smoke-tested on these cases.

* Invocation `run_eval_liver_ext.sh` → `results/sota/sat_pro_liver_ext.jsonl` (93 lines). GPU 3 (shared with a 36 GB
  job), `OMP_NUM_THREADS=8` and 2 loader workers because the CPU was saturated; neither changes outputs.
* Code changes: dataset key `liver_ext` (cache dir `cache/liver_ext`, prompt "liver tumor") in `run_sat.py`;
  `coord_check.py --ds liver_ext`.
* Coordinate check on HCC_001 (spacing 5.0×0.781×0.781 mm, shape 36×512×512). There is no pool for this set, so the
  sphere sits at a fixed fraction (0.50, 0.45, 0.40) of the array shape and no label or mask is read. Known
  (18, 230, 204), recovered (18.00, 230.24, 203.44): error (0.00, +0.24, −0.56) voxels = (0.00, 0.19, −0.44) mm,
  PASS (`logs/coord_check_liver_ext.log`). A one-voxel interpolation fringe formed a second, lower-ranked component
  beside the sphere.
* Runtime: 93 cases in 293 s of inference (3.2 s/case, model 2.7 s/case), about 1 min setup.
* Failures: no crashes; empty masks 3 / 93 (HCC_017, HCC_048, HCC_095; max probability 0.25, 0.06, 0.32);
  no clipping.
* Training-data overlap: none known. HCC-TACE-Seg is not in SAT-DS (not among the 72 datasets of
  `data/dataset_config/72.json`, not in the SAT-DS README or loader).
* Score (one run of `score_sota.py sat_pro vista3d` after the outputs and this section were final; earlier lines
  unchanged): `sat_pro  liver_ext hit 0.88 (n=93, answered 93)`.
