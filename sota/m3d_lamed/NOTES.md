# M3D-LaMed-Phi-3-4B (BAAI M3D, 2024)

Results: `results/sota/m3d_lamed_luna.jsonl` (200 rows) and `results/sota/m3d_lamed_liver.jsonl` (53 rows).
Code: `run_m3d.py` (check / smoke / eval) and `common.py` (case lists, LPS volumes, stored-order transforms).
The prompts are in `PROMPT.txt`.

## Versions
- Official code: BAAI-DCAI/M3D at commit 63e7200 (`/raid/home/CAMCA/hj880/ext_src/M3D`).
- Weights: `GoodBaiBai88/M3D-LaMed-Phi-3-4B`, HF snapshot 329bed510c2c (remote code `modeling_m3d_lamed.py`), run in fp32 as stored.
- venv `/raid/home/CAMCA/hj880/venvs/sota_m3d_lamed` (`setup_env.sh`):
  - Python 3.10.22, torch 2.2.1+cu118, transformers 4.42.4, monai 1.3.0;
  - numpy 1.26.4, scipy 1.13.0, SimpleITK 2.3.1, nibabel 5.2.1;
  - einops 0.8.0, safetensors 0.4.3, sentencepiece 0.2.2, accelerate 0.31.0.
- Hardware: one H100 (GPU 2).

## Pipeline as run
1. **Volume.** Original voxel values in the file's stored voxel order `[k, j, i]`, with no reorientation. M3D-Seg (`Data/process/m3d_seg_data_preprocess.py`) loads NIfTI without `Orientationd`, so this matches how its training volumes were built. How the M3D-Seg LUNA16 `.nii.gz` files were converted from `.mhd` is not documented.
   - LUNA16: the cached array is a lossless copy, transposed and flipped back to the `.mhd` order. Among the eval scans, 195 have an identity direction and 5 have x/y flipped.
   - Liver: the original NIfTI, stored RAS. The cache is clipped to [-1024, 3071], so it is not used as input; the original agrees with the cache after clipping, which `common.load_lps` asserts.
2. **Preprocessing.** The repo's positioning-data path, verbatim:
   - SegVol `normalize`: clip to the 0.05/99.95 percentiles of the voxels above the volume mean, then z-score;
   - min-max to [0, 1] (`m3d_seg_data_prepare.py`);
   - `CropForeground` (voxels > 0, margin 0);
   - `Resize` to 32x256x256 in MONAI 'bilinear', which is trilinear for 3D with align_corners=False.
3. **Prompt.** `"<im_patch>"*256 + " " + question`, with no chat template (`PosRECDataset`). The input ids are asserted to be `<s>` followed by 256 `<im_patch>` tokens. Decoding is greedy, `max_new_tokens=256`.
4. **Parsing.** The first 6-number box found by the repo's regex (`Bench/utils.py`). No answer contained more than one box.
5. **Inverse.** For each axis, s = cropped size / resized size and o_min, o_max = box * resized size.
   - Centre: `(o_min + o_max)/2 * s - 0.5 + crop_start` in stored order, then permuted and flipped to the cached LPS index.
   - Size: `diameter_mm` is the mean over the three axes of `(o_max - o_min + 1) * s * spacing`.

## Deviations from the official recipe
- **transformers 4.42.4 instead of the pinned 4.39.1.** `requirements.txt` targets the older Llama-2 model. The Phi-3 checkpoint imports `Phi3Config`/`Phi3Model`, which are absent in 4.39.1, and its config records 4.42.0.dev0. Everything else is as pinned, except that accelerate and sentencepiece were added for loading and the slow tokenizer.
- **Preprocessing source.** The README quickstart says only "1x32x256x256, min-max to [0, 1], consider resize". I used the exact code path that produced the M3D-Seg volumes, because the box (REC) task was trained on those volumes; that path adds percentile clipping and foreground cropping. The online demo resizes raw NIfTI without normalisation; I did not follow it.
- **Size floor in the question.** The training template is "Where is {} in this image? Please output the box."; I filled {} with "the largest lung nodule of at least 6 mm" (or "the largest liver tumor of at least 10 mm"), because the protocol requires a size floor. M3D-Seg has no nodule class; its closest class is "lung tumours" (MSD Task06).
- **Box-centre convention.** M3D's box labels come from masks resized with *nearest* interpolation and then passed to `mask2box` (min/max index / size). The inverse above is the exact inverse of that label path.
  - The alternative, which treats box indices as trilinear sample centres, shifts every answer by +s/2 voxels: about 4 slices in z for LUNA16 and about 1 voxel in-plane.
  - A Monte-Carlo test on 60 random 40 mm spheres pushed through the label path gave a mean z error of +0.01 voxels for the convention used and +3.96 voxels for the alternative.
- The official code passes no attention mask; neither did I (batch size 1, no padding).

## Coordinate check (`coord_check.json`, pool scans only)
I put a 10 mm, +300 HU sphere deep inside the lung (LUNA16 lung mask) or the liver (pool liver mask), centred on a fed-voxel centre. I then measured its intensity-weighted centroid in the actual 1x32x256x256 tensor given to the model and mapped that centroid back.

| scan | stored order vs LPS | crop | cropped per resized voxel (z, y, x) | error back in cached voxels (z, y, x) |
|---|---|---|---|---|
| LUNA ...870245028 | identity | none | 7.91, 2.00, 2.00 | 0.00, -0.02, 0.02 |
| LUNA ...415322775 | x, y flipped | y 512 to 499 | 8.84, 1.95, 2.00 | 0.00, 0.15, 0.01 |
| liver_3 | x, y flipped | y 512 to 505 | 16.69, 1.97, 2.00 | 0.00, 0.24, -0.07 |

All three are within 1 voxel. On the box-label path, a 10 mm sphere spans at most two of the 32 resized slices, so `mask2box` on the nearest-resized mask is quantised to s/2 in z.
- LUNA ...870245028: the error was (-3.95, -1.24, -1.19) voxels, i.e. -4.9 mm in z.
- The other two scans: the sphere fell between nearest-resize samples, so there was no box.

This bounds how precisely any M3D box can place a small lesion in z: ±s/2 ≈ ±4 to 8 slices.

The model itself, run on the synthetic volumes, returned lung- or organ-sized boxes 58 to 142 mm from the sphere.

## Smoke test (pool scans, `smoke.jsonl`)
- 3 LUNA16 pool scans (one with a flipped direction) and 3 liver pool cases.
- All 6 gave a parseable box.
- None hit (pool labels): boxes had mean extents of 45 to 143 mm, and centres were 27 to 181 mm from the target.
- The prompt was not changed after the smoke test.

## Runtime (GPU 2, CPU shared with a load average of about 175)
- LUNA16: 200 scans in 464 s (2.3 s per scan).
- Liver: 53 cases in 582 s (11 s per case). Decompressing the original NIfTI and computing the percentile normalisation on 512x512x(74 to 861) volumes dominate under CPU contention; generation takes about 1 s.

## Failures
- Parse failures: 0/200 LUNA16 and 0/53 liver. Every answer is a single box, so present = true for all 253.
- The boxes are organ-sized. The median of the mean box extent is 122 mm for LUNA16 (p10 to p90: 52 to 178 mm) and 52 mm for the liver (16 to 140 mm).
- There were no crashes.

## Known training-data overlap
M3D-LaMed's box and mask tasks were trained on M3D-Seg (25 public datasets; see the HF dataset card and `LaMed/src/dataset/dataset_info.py`). The overlaps are:
- **0024 LUNA16.** The same CT volumes as every LUNA16 scan, including our 200 eval scans, with left lung, right lung and trachea masks. The images were seen in training with organ boxes and masks, but no nodule labels.
- **0020 MSD-Liver = MSD Task03.** Labels are liver and tumour, and REC/REG boxes for "tumour" were trained. M3D-Seg's per-dataset train/test split JSON is not available here, so most or all of our 53 liver eval cases should be assumed seen *with tumour boxes*.
- **0012 CT-ORG** (built largely on the LiTS volumes that make up MSD Task03) and **0009 3D-IRCADb** (whose 20 volumes are part of LiTS). The labels include "liver tumor" for 3D-IRCADb.
- **0021 MSD-Lung tumours.** Different patients from LIDC-IDRI.
- LUNA16 itself is a subset of LIDC-IDRI.
- M3D-Cap and M3D-VQA (Radiopaedia) are unrelated.

## Score (one run of `score_sota.py m3d_lamed hulumed14b medgemma15`, after all three eval runs; nothing changed afterwards)
```
m3d_lamed                fresh1 hit 0.00 largest 0.00 diamMAE nan answered 100 | fresh2 hit 0.00 largest 0.00 diamMAE nan answered 100
m3d_lamed                liver hit 0.08 (n=53, answered 53)
hulumed14b               fresh1 hit 0.00 largest 0.00 diamMAE nan answered 100 | fresh2 hit 0.00 largest 0.00 diamMAE nan answered 100
hulumed14b               liver hit 0.00 (n=53, answered 53)
medgemma15               fresh1 hit 0.00 largest 0.00 diamMAE nan answered 100 | fresh2 hit 0.00 largest 0.00 diamMAE nan answered 100
medgemma15               liver hit 0.00 (n=53, answered 53)
```

## External liver set: HCC-TACE-Seg (93 cases; run after the scores above)
- **Unchanged.** Same input pipeline, the MSD liver question ("Where is the largest liver tumor of at least 10 mm in this image? Please output the box.") and the same parsing. Nothing was tuned on these cases.
  - Code: `run_m3d.py check|eval --ds liver_ext`. In `common.py`, the cases come from `sota/eval_liver_ext.txt` and the originals from `data_ext/hcc_tace_seg/nifti/imagesTr`.
  - The converter wrote these originals as int16 already clipped to [-1024, 3071], with an identity (LPS) direction for all 93 cases. So the stored order equals the cached order and the values equal the cache; `load_lps` asserts this.
- **Coordinate check** (`coord_check_liver_ext.json`) on HCC_002: 109 x 2.5 mm, cropped to y 45-484 and x 7-512, s = (3.41, 1.72, 1.97).
  - Every cached liver_ext case is an evaluation case, so the sphere was placed at an image-only location (`common.label_free_center`: on the middle slice, the voxel of the largest -100..150 HU region farthest from its edge), with no labels read.
  - Fed-tensor centroid back in cached voxels: (0.00, 0.00, -0.21). Label-box path: (-0.01, -0.08, -0.94). Both are within 1 voxel.
  - The model's box was 38 mm from the sphere.
- **Runtime** (GPU 5, `OMP_NUM_THREADS=8`): 93 cases in 243 s (2.6 s per case).
- **Failures:** 0/93 parse failures; every answer is a box. Median mean box extent: 65.5 mm.
- **Training-data overlap:** HCC-TACE-Seg (TCIA, MD Anderson) is not among the 25 M3D-Seg datasets, and M3D-Cap/VQA come from Radiopaedia. No known overlap.

### Score including liver_ext (second and final scoring run, after the three liver_ext eval runs; the LUNA16/MSD lines are unchanged)
```
m3d_lamed                liver_ext hit 0.14 (n=93, answered 93)
hulumed14b               liver_ext hit 0.04 (n=93, answered 93)
medgemma15               liver_ext hit 0.00 (n=93, answered 93)
```
