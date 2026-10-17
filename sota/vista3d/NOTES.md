# VISTA3D on our benchmark

VISTA3D (He et al., "VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging", CVPR 2025), run
through its official MONAI bundle in automatic (class-prompt) mode.

## What was run

| item | value |
|---|---|
| bundle | HF `MONAI/vista3d` snapshot `c6dbe159` = MONAI model-zoo bundle "vista3d" v0.5.11 (`configs/inference.json`, `scripts/inferer.py`, `scripts/evaluator.py`), unmodified |
| weights | bundle `models/model.pt` (sha256 `c92bab26d00b4a5d…`), network `monai.networks.nets.vista3d132` |
| research repo | `ext_src/VISTA3D` = github.com/Project-MONAI/VISTA commit `d4a8fe0`; read only for the training data lists (its MONAI 1.3 inference code expects a different checkpoint file) |
| not used | `MONAI/VISTA3D-HF` (same model wrapped as a transformers pipeline; the bundle is the documented inference path) |
| prompts | LUNA16 `label_prompt [23]` "lung tumor"; liver `label_prompt [26]` "hepatic tumor" (see `PROMPT.txt` and below); no point prompts |
| runner | `run_vista3d.py` (eval), `run_eval.sh` (the one eval invocation), `common.py` (NIfTI I/O + mask rule), `coord_check.py`; pool reports with `../sat_pro/smoke_check.py` |
| outputs | `results/sota/vista3d_luna.jsonl` (200 lines), `results/sota/vista3d_liver.jsonl` (53 lines) |
| GPU | GPU 1 only (`CUDA_VISIBLE_DEVICES=1`; the bundle's `cuda:0`), shared with another job |

Pipeline per case: cached LPS array → uncompressed NIfTI with the explicit RAS affine `diag(-sx, -sy, sz)`
(`common.write_nifti`) → everything else from `configs/inference.json`, instantiated with `monai.bundle.ConfigParser`:
LoadImaged, EnsureChannelFirstd, EnsureTyped (GPU), Spacingd 1.5 mm bilinear, CropForegroundd (margin 10),
VistaPreTransformd, ScaleIntensityRanged [-963.8, 1053.7] → [0, 1], Orientationd RAS; `Vista3dInferer` (sliding
window 128³, overlap 0.3, replicate padding); `Vista3dEvaluator` (amp, `user_prompt` mode, `set_determinism(123)`,
CheckpointLoader); postprocessing VistaPostTransformd (logit > 0 → label) → Invertd (nearest) back to the input
NIfTI grid → nan_to_num. The inverted mask is transposed (x, y, z) → (z, y, x), which is the cached index.

## Class prompts (label IDs)

* **LUNA16: 23 "lung tumor".** VISTA3D has no nodule class; 23 is its only lung-lesion class (supervised by MSD
  Task06 lung tumours, `label_mappings.json`: `"Task06": [[1, 23]]`). The other lesion classes are organ-specific
  (24 pancreatic tumor, 26 hepatic tumor, 27 colon cancer primaries, 128 bone lesion, 129 kidney mass,
  130 liver tumor), so 23 alone was used.
* **Liver: 26 "hepatic tumor".** The label dict also lists 130 "liver tumor", but the bundle documents it as
  unsupported ("130, # liver tumor already have hepatic tumor", `docs/inference.md`) and its evaluator rejects it
  without point prompts: running `--label 130` on pool case liver_0 stops with "Undefined label prompt detected.
  Provide point prompts for zero-shot." VISTA3D's own training maps the MSD Task03 tumour label to 26
  (`"Task03": [[1, 1], [2, 26]]`), so 26 is "liver tumor" in VISTA3D's vocabulary.

## Mask → answer rule (fixed before the eval run, identical for SAT-Pro)

`common.mask_to_answer` on the inverted mask in the cached grid: 26-connected components; point = centroid (cached
`[z, y, x]` index); size = equivalent-sphere diameter (6V/π)^(1/3), V = voxels × sz·sy·sx; score = mean predicted
probability over the component, where the probability is sigmoid(logit) inverted with the same nearest-neighbour
Invertd as the mask (so mask = probability > 0.5 voxel for voxel); answer = highest score (ties → larger);
`candidates` = top 10, `candidates[0]` = answer; empty mask → `present: false`, `zyx: null`. `raw` holds
diagnostics as a JSON string.

## Environment (`/raid/home/CAMCA/hj880/venvs/sota_vista3d`, full list in `pip_freeze.txt`, recipe in `setup_env.sh`)

Python 3.10.22, torch 2.4.0+cu124, monai[fire,ignite] 1.4.0, numpy 1.24.4, nibabel 5.2.1, einops 0.7.0,
scikit-image 0.23.2, pytorch-ignite 0.4.11, matplotlib 3.9.1, tensorboard 2.17.0, mlflow 2.17.2,
cucim-cu12 24.6.0, SimpleITK 2.3.1. These are exactly the versions in the bundle's `metadata.json`
(`monai_version`, `pytorch_version`, `numpy_version`, `required_packages_version`).

## Deviations from the official recipe

1. **Postprocessing.** `SaveImaged` dropped (results are reduced in memory) and one side branch added: a copy of the
   logits → sigmoid → the same `Invertd`, used only to score components. The official entries
   (VistaPostTransformd, Invertd with `nearest_interp`, nan_to_num) are reused verbatim from `inference.json`.
   The disabled `ToDeviced` entry is dropped (no-op).
2. **Batch input.** All cases go in one `dataset#data` list (the mechanism of the bundle's `batch_inference.json`)
   with a per-case `label_prompt`; results are collected by an ignite `ITERATION_COMPLETED` handler.
3. NIfTI files are written from the cached arrays (16 processes, before the run) and deleted after use.
4. `sys.dont_write_bytecode` is set so that importing the bundle's `scripts` does not write into the HF cache.
5. `meta.npy` files were pickled under numpy 2; `common.load_spacing` aliases `numpy._core` to `numpy.core` only
   while unpickling the spacing (left installed, a later `import torch` segfaults in this env).

## Coordinate check (README rule 3) — PASS

`coord_check.py` on pool scan `…100953483028192176989979435275` (spacing 2.5×0.703×0.703 mm, shape 133×512×512):
10 mm sphere of +300 HU at cached index (83, 240, 352). Original and synthetic arrays went through `write_nifti` and
the bundle preprocessing (processed grid 241×199×221 at 1.5 mm); the processed difference was sent through the run's
postprocessing chain (VistaPostTransformd, Invertd, nan_to_num), transposed and reduced with the same rule.
Recovered (83.00, 239.95, 352.04): error (0.00, −0.05, +0.04) voxels = (0.00, −0.04, +0.03) mm.
End-to-end the model does not call the synthetic sphere a "lung tumor": its probability inside the sphere rises
from < 0.01 to at most 0.22, below the 0.5 threshold (the real nodule of that scan is still found).

## Smoke tests (training-pool cases only)

* LUNA16 pool scans `…979435275`, `…764325545`, `…502185026` (label 23): the rank-1 component lies 0.6, 2.5 and
  1.1 mm from an annotated nodule centre (10.4, 12.7, 10.3 mm nodules); equivalent diameters 10.1, 9.6 and 3.6 mm
  (the last a 28-voxel component, max probability 0.71).
* Liver pool cases `liver_0`, `liver_5`, `liver_14` (label 26): the rank-1 component lies inside a tumour in all
  three (17, 9, 13 mm tumours; predicted 18.4, 8.6, 14.7 mm).

## Runtime

* LUNA16: 200 scans in 1014 s of inference (5.1 s/scan; the GPU was shared with another job at 100 % utilisation,
  so later scans took 10–16 s), plus 38 s to write the NIfTI inputs and ~10 s to build the bundle and load the checkpoint.
* Liver: 53 cases in 390 s (7.4 s/case), plus 28 s to write the NIfTI inputs.

## Failures

* No crashes; every eval case has a line; no unprocessed (NaN/255) voxels; no centroid outside the cached array.
* Empty masks (→ `present: false`): **LUNA16 55 / 200 (27.5 %)**, median max-probability in those scans 0.20, i.e.
  the "lung tumor" head responds weakly but stays below 0.5 (in the pool smoke test it likewise missed the 9.2 and
  6.1 mm nodules of scan `…502185026`);
  liver 1 / 53 (liver_54, max probability 0.15).
* Answer sizes differ from SAT-Pro on LUNA16 (median equivalent diameter 14.4 mm vs 7.8 mm): the tumour class tends
  to segment larger regions.

## Training-data overlap

From the research repo's data lists (`vista3d/data/jsons/*_5_folds.json`, `label_mappings.json`) and the training
configs (fold 0 = validation, `configs/train/hyper_parameters_stage*.yaml`):

* **MSD Task03** is a VISTA3D training set (liver → 1, tumour → 26 "hepatic tumor"): 105 cases in its training list
  (5 folds), 26 held out as test. Of our 53 liver eval cases, **31 are in folds 1–4 (trained on, with tumour
  masks)**, 13 in fold 0 (its validation fold) and 9 in its test list (liver_18, 19, 33, 54, 67, 77, 99, 113, 120).
  Whether the released checkpoint also saw fold 0 is not documented.
* **LIDC-IDRI**: 422 CTs of 420 LIDC patients are in VISTA3D's training lists, each with only a model-generated
  `pseudo_label` file and a SAM supervoxel file (`label_sv`); LIDC has no entry in `label_mappings.json`, so none of
  its expert nodule annotations is used (whether the pseudo-labels contain class 23 is not documented). Mapping our LUNA16 series UIDs
  to LIDC patient IDs (TCIA LIDC-IDRI series list, one CT series per matching patient): **84 / 200** of our eval
  scans belong to those patients, i.e. their images were seen in training without nodule supervision.
* The "lung tumor" class is supervised by MSD Task06 only (NSCLC tumours, typically much larger than our 6 mm+
  nodules).

## Score (one run of `score_sota.py sat_pro vista3d`, after all outputs and these notes were final)

    vista3d  fresh1 hit 0.42 largest 0.35 diamMAE 2.2 answered 100 | fresh2 hit 0.46 largest 0.45 diamMAE 2.3 answered 100
    vista3d  liver hit 0.94 (n=53, answered 53)

The liver number is not a held-out result: 31 of the 53 cases are in VISTA3D's training folds with tumour masks
(13 more in its validation fold).

## External liver test set: HCC-TACE-Seg (93 cases)

Same pipeline, class prompt (26 "hepatic tumor") and mask → answer rule as for MSD liver; nothing was tuned or
smoke-tested on these cases.

* Invocation `run_eval_liver_ext.sh` → `results/sota/vista3d_liver_ext.jsonl` (93 lines). GPU 3 (shared with a
  36 GB job), `OMP_NUM_THREADS=8`, NIfTI inputs written by 4 processes (new `--nproc` option, default 16 as before)
  because the CPU was saturated; neither changes outputs.
* Code changes: dataset key `liver_ext` → label 26 in `run_vista3d.py`, `--nproc`, `coord_check.py --ds liver_ext`.
* Coordinate check on HCC_001 (spacing 5.0×0.781×0.781 mm, shape 36×512×512; processed grid 267×247×118 at
  1.5 mm). The sphere sits at a fixed fraction (0.50, 0.45, 0.40) of the array shape; no label or mask is read.
  Known (18, 230, 204), recovered (18.00, 230.37, 203.63): error (0.00, +0.37, −0.37) voxels =
  (0.00, 0.29, −0.29) mm, PASS (`logs/coord_check_liver_ext.log`).
* Runtime: 93 cases in 282 s of inference (3.0 s/case), plus 50 s to write the NIfTI inputs; 347 s wall.
* Failures: no crashes; empty masks 1 / 93 (HCC_095, max probability 0.14); no NaN voxels, no clipping.
* Training-data overlap: none known. HCC-TACE-Seg is in no VISTA3D training or validation list (`data/jsons`,
  `configs/train`); the research repo keeps it under `data/external/` for its zero-shot interactive evaluation
  (`configs/zeroshot_eval/infer_iter_point_hcc.yaml`). We use automatic mode, no points.
* Score (one run of `score_sota.py sat_pro vista3d` after the outputs and this section were final; earlier lines
  unchanged): `vista3d  liver_ext hit 0.85 (n=93, answered 93)`.
