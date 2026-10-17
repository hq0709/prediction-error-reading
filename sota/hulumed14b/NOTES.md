# Hulu-Med-14B (ZJU, 2025)

Results: `results/sota/hulumed14b_luna.jsonl` (200 rows) and `results/sota/hulumed14b_liver.jsonl` (53 rows).
Code: `run_hulumed.py` (check / smoke / eval) and `common.py`. The prompts are in `PROMPT.txt`.

## Versions
- Official code: ZJUI-AI4H/Hulu-Med at commit 6d4d47a (`/raid/home/CAMCA/hj880/ext_src/Hulu-Med`). Inference used the HF-native path from the README ("Option 1"), not the legacy `hulumed` package.
- Weights: `ZJU-AI4H/Hulu-Med-14B` (Qwen3-14B LLM), HF snapshot b30d9161b8c2, with remote code `modeling_hulumed_qwen3.py`, `processing_hulumed.py` and `image_processing_hulumed.py`. Run in bf16 with `attn_implementation="flash_attention_2"`, as in the README.
- venv `/raid/home/CAMCA/hj880/venvs/sota_hulumed14b` (`setup_env.sh`). It follows the README install block; from `requirements.txt` I installed only the subset needed for inference:
  - Python 3.10.22, torch 2.4.0+cu118, flash-attn 2.7.3 (prebuilt cu11/torch2.4 wheel);
  - transformers 4.51.2, accelerate 1.7.0, tokenizers 0.21.4;
  - numpy 1.24.4, nibabel 5.3.2, timm 1.0.3, decord 0.6.0, einops 0.6.1;
  - SimpleITK 2.3.1, which I added for my own I/O.
- Hardware: one H100 (GPU 3).

## Pipeline as run
1. **Volume.** The cached LPS volume with original voxel values. For LUNA16 this is the lossless cache. For the liver it is the original NIfTI reoriented exactly as the cache was built, because the cache is clipped to [-1024, 3071] and this path normalises min-max per slice.
2. **NIfTI.** Written to a temporary `.nii` with an explicit affine, `volume[i, j, k] = vol_lps[k, i, j]`, with i pointing posterior, j pointing patient left and k pointing superior. The processor's `load_nii(axis=2)` slice `volume[:, :, k]` is then the axial slice in radiological display orientation: rows run anterior to posterior, columns run patient right to left, and k runs inferior to superior. I checked this visually in `check_frame_luna.png` and `check_frame_liver.png`.
3. **Documented 3D path.** `{"type": "3d", "3d": {"image_path", "nii_num_slices": 180, "nii_axis": 2}}`:
   - slice choice is `np.linspace(0, N-1, 180, dtype=int)` if N > 180, otherwise all N slices;
   - there is no windowing (the default); each slice is min-max scaled to uint8 and copied to 3 channels;
   - the slices are processed as a video with merge size 2, and the processor resizes the frames to fit 16384 visual tokens. Frames are 252x252 for 180 slices and up to 336x336 for fewer slices.
   - The processor is called with `add_system_prompt=True, add_generation_prompt=True`. Decoding is greedy (`do_sample=False`) with `max_new_tokens=1024`. The other `generation_config.json` fields were kept, including `repetition_penalty=1.05`. The answer is `batch_decode(use_think=False)`.
4. **Parsing.** The last JSON object that parses, with keys present, slice, x, y and diameter_mm. After this run, the parser in `common.py` gained two cases for MedGemma: JSON wrapped in LaTeX `\boxed{}` and a stated-text fallback. Re-parsing all 253 Hulu-Med answers with the final version gives identical fields.
5. **Inverse.** z = fed_slice_indices[slice], y = y, x = x, as cached LPS indices. Out-of-range coordinates give zyx = null.

## Deviations from the official recipe
- **In-plane orientation.** The README's 3D demo NIfTI (AMOS, LAS) shows slices rotated by 90° under `load_nii`. I wrote the NIfTI so that the slices appear in the standard radiological display (the orientation of the Radiopaedia-derived M3D-Cap and M3D-VQA volumes in Hulu-Med's training data). The slice order (inferior to superior, k pointing superior) matches the demo file. The README does not prescribe a file orientation, so I count this as a choice rather than a change of method.
- **Prompt fields.** The prompt states each case's slice count, slice spacing and pixel spacing so the model can answer in fed-slice and pixel units and in mm. These come from geometry only; the template is fixed (`PROMPT.txt`).
- **Token budget.** On scans with fewer than 180 slices the processor enlarges the frames, and 29 of 253 cases then exceed the tokenizer's `model_max_length` of 16384 (maximum 16,672 tokens). The cause is that the processor budgets 16384 visual tokens before adding the text. The model has 40,960 positions and ran normally, but these inputs are slightly longer than the training length. This is the documented processing, unchanged.
- The training-only parts of `requirements.txt` (deepspeed, gradio, wandb and similar) were not installed.

## Coordinate check (`coord_check.json`, pool scans only)
I put a 10 mm, +300 HU sphere deep in the lung or liver, centred on a fed slice. I located it in the actual `pixel_values` given to the model by inverting the processor's patch flattening and differencing against the scan without the sphere. I then mapped the resulting frame and pixel back.

| scan | fed slices | frame seen by model | sphere found in frame (expected) | error back in cached voxels (z, y, x) |
|---|---|---|---|---|
| LUNA ...870245028 | 180 of 253 | 252x252 | 95 (95) | 0.00, -0.05, 0.22 |
| LUNA ...979435275 | 133 of 133 | 308x308 | 83 (83) | 0.00, -0.10, 0.11 |
| liver_3 | 180 of 534 | 252x252 | 142 (142) | 0.00, -0.11, 0.02 |

All three are within 1 voxel. Run on the synthetic volumes, the model answered with `{"present": true, "slice": 103 to 105, "x": 234 to 245, ...}` in each case, 93 to 158 mm from the sphere.

## Smoke test (pool scans, `smoke.jsonl`)
- 3 LUNA16 and 3 liver pool scans; all 6 answers were valid JSON.
- For LUNA16, all three answers were "present" with diameter 8.9 mm, 112 to 197 mm from the nearest nodule of at least 6 mm.
- For the liver, liver_8 was answered "absent", and the other two answers were 77 and 164 mm from the nearest target tumour.
- The prompt was not changed.

## Runtime (GPU 3)
- LUNA16: 200 scans in 1261 s (6.3 s per scan, including writing the NIfTI and loading it in the processor).
- Liver: 53 cases in 465 s (8.8 s per case).

## Failures
- LUNA16: 199/200 answers parsed. One answer (...805925269324902055566754756843, `"y": 789` on a 512-pixel image) was out of range, so it stays present = true with zyx = null.
  - Present: 196/200. With coordinates: 195/200.
- Liver: 53/53 answers parsed. Present with coordinates: 34. Answered "absent": 19.
- There were no crashes.

**Answers look prior-driven.** They barely vary across scans:
- LUNA16: slice 105 appears 65 times, 102 appears 61 times and 103 appears 32 times out of 200; x = 345 appears 70 times; diameter_mm = 8.9 appears 137 times.
- Liver: slice 105 appears 21 times out of 53; diameter_mm = 18.9 appears 19 times.
- The synthetic-sphere runs gave the same pattern.

## Known training-data overlap
Data are from the paper (arXiv 2510.08668, Extended Data Tables 4 to 6 and 9).
- **No LIDC-IDRI, LUNA16, MSD or LiTS volumes are listed.** The 3D training sets are M3D-Cap, M3D-VQA (Radiopaedia), CT-RATE and AMOS-derived sets.
- **Indirect 2D overlap is possible:**
  - SLAKE (Stage-3 VQA) contains 2D CT slices taken from the Medical Segmentation Decathlon, which includes liver and tumour cases.
  - MedTrinity-25M (the 161K subset is used in Stage 2) includes ULS23, whose public training data contain 2,246 LIDC-IDRI lung lesions and 888 LiTS liver lesions (LiTS = MSD Task03), as 2D crops with text.
  - Whether any of our eval cases are in those subsets is unknown.

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
- **Unchanged.** Same NIfTI path, the MSD liver prompt template and the same parsing. Nothing was tuned on these cases.
  - Code: `run_hulumed.py check|eval --ds liver_ext`. The volumes are the cached LPS arrays; the originals in `data_ext/hcc_tace_seg/nifti/imagesTr` are identical (already clipped to [-1024, 3071] by the converter), and `load_lps` asserts this.
  - 92 of the 93 scans have at most 180 slices (36 to 185, median 89, mostly 2.5 mm), so they are fed whole.
  - 6/93 inputs exceed 16,384 tokens.
- **Coordinate check** (`coord_check_liver_ext.json`, `check_frame_liver_ext.png`) on HCC_002: 109 slices fed whole, frames 336x336.
  - The sphere was placed at an image-only location (no labels) on a fed slice.
  - It was found in frame 54 (expected 54). Error back in cached voxels: (0.00, 0.35, 0.09), within 1 voxel.
  - The model answered `{"slice": 102, "x": 345, "y": 267, "diameter_mm": 18.9}`, 219 mm away.
- **Runtime** (GPU 5, `OMP_NUM_THREADS=8`): 93 cases in 445 s (4.8 s per case).
- **Failures:** 0/93 parse failures. 83 answers are present with coordinates and 10 are "absent". Answers again cluster: slice 45 and slice 35 appear 18 times each, slice 102 appears 13 times, and diameter 18.9 mm appears 23 times.
- **Training-data overlap:** HCC-TACE-Seg is not in the paper's training tables. The CT sources inside MedTrinity-25M (CT-RATE, DeepLesion, FLARE23, KiPA22, ULS23) do not include it. No known overlap.

### Score including liver_ext (second and final scoring run, after the three liver_ext eval runs; the LUNA16/MSD lines are unchanged)
```
m3d_lamed                liver_ext hit 0.14 (n=93, answered 93)
hulumed14b               liver_ext hit 0.04 (n=93, answered 93)
medgemma15               liver_ext hit 0.00 (n=93, answered 93)
```
