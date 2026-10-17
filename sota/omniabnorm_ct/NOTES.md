# OmniAbnorm-CT on our benchmark

OmniAbnorm-CT (Zhao, Dai, Zhang, Xie, Wang, "Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric
Approach", CVPR 2026, arXiv 2506.03238): Qwen2.5-VL-7B + LoRA, a 2D U-Net segmentation module prompted by the hidden
state of a `<SEG>` token, and a second VLM round that writes the report after the mask is drawn as a box. It works
on one axial (or coronal/sagittal) key slice plus 8 adjacent slices for the U-Net. Run here in its
"text-guided grounded report generation" mode, swept over every axial slice.

## What was run

| item | value |
|---|---|
| code | `ext_src/OmniAbnorm-CT` = github.com/zhaoziheng/OmniAbnorm-CT commit `b4fd478` (2026-06-11), unmodified (git status clean, no bytecode written: `sys.dont_write_bytecode`) |
| weights | HF `zzh99/OminiAbnorm-CT` snapshot `f9448ea3`: `ominiabnorm-ct-7b/adapter_model.safetensors` (LoRA r 32 on q/v_proj), `ominiabnorm-ct-7b/seg_model_lm_head_embed_tokens.pt` (U-Net + lm_head + embed_tokens, 151666 rows); `pretrained_unet.pth` not used (stage-1 U-Net, superseded by the joint checkpoint) |
| base model | `Qwen/Qwen2.5-VL-7B-Instruct` snapshot `cc594898` (hf_cache) |
| prompts / windows | see `PROMPT.txt`: LUNA16 "Evaluate the lungs for any lung nodule. If any, localize it on the image and describe the findings." with window [-1150, 350]; liver "Evaluate the liver for any liver tumor. If any, localize it on the image and describe the findings." with window [-140, 260] |
| runner | `run_omniabnorm.py` (sweep + answer), `common.py` (I/O + mask rule), `coord_check.py`, `run_eval.sh` (the one eval invocation), `setup_env.sh`; pool reports with `../sat_pro/smoke_check.py` |
| outputs | `results/sota/omniabnorm_ct_luna.jsonl` (200 lines), `results/sota/omniabnorm_ct_liver.jsonl` (53 lines) |
| GPU | GPU 4 only (`CUDA_VISIBLE_DEVICES=4`); the LUNA16 and liver runs ran as two processes side by side (~28 GB each) |

### Slice sweep (official procedure per key slice)

The repo has no whole-volume procedure (`inference.py` takes one folder of up to 9 slice images, uses the middle one as
key slice and all of them as U-Net context). Every axial slice z of the cached array is a key slice once:

1. HU -> 8-bit image with the dataset's window (`round((clip(HU, lo, hi) - lo) / (hi - lo) * 255)`): the PNG a user
   would hand to `inference.py`. Slices are shown as stored: rows = y (anterior at the top), columns = x (patient's
   right on the left), i.e. radiological convention, as in the Radiopaedia training images.
2. `inference.py` `preprocess_input`, fed from arrays instead of files: per-slice z-score `(img - mean) / (std + 1e-10)`;
   9-slice context = key slice with up to 4 neighbours on each side in cranio-caudal order (slices z+4 ... z+1 first,
   then z, then z-1 ... z-4; missing neighbours at the volume ends are zero slices, as the training datasets pad);
   official `pad_truncate_and_resize` (square padding, 512 x 512 bilinear: identity here, every eval slice is
   512 x 512); VLM image = official `get_pil_image` (min-max uint8) of the key slice in the official `format_data`
   message (512 x 512 pixels, system message "You are a helpful medical assistant.").
3. First round of `evaluate_refer_grounding.py` (batched, 32 key slices per batch, DataLoader workers run the
   processor in the collate function as there): generation with the base model's generation config, as the official
   code loads it (sampling at temperature 1e-6 with repetition penalty 1.05, i.e. greedy; the copy saved with the
   checkpoint, from the authors' older base snapshot, has temperature 0.1 / top_k 1, also greedy), max 1024 new
   tokens; last-layer hidden state of the first `<SEG>` -> U-Net (bf16
   autocast) -> sigmoid -> > 0.5 -> official `recover_single_slice` -> slice z of a 3D mask on the cached grid; the
   sigmoid map is kept the same way for the component score.
4. Fixed mask -> answer rule (`common.mask_to_answer`, identical to SAT-Pro / VISTA3D): 26-connected 3D components;
   point = centroid (cached [z, y, x]); size = equivalent-sphere diameter (6V/pi)^(1/3), V = voxels x sz*sy*sx;
   score = mean sigmoid probability over the component; answer = highest score (ties -> larger); top 10 kept as
   `candidates`; empty mask -> `present: false`.
5. The official two-round `generate_text_and_grounding` (inference.py) is run once per case on the answer's key
   slice (where the answer component is largest); its report is `raw.report`.

## Environment (`/raid/home/CAMCA/hj880/venvs/sota_omniabnorm`, full list in `pip_freeze.txt`, recipe in `setup_env.sh`)

Python 3.10.22, torch 2.0.0+cu118, torchvision 0.15.1+cu118, triton 2.0.0, transformers 4.49.0, tokenizers 0.21.1,
accelerate 1.2.1, peft 0.10.0, flash-attn 2.3.6 (prebuilt cu118/torch2.0 wheel), qwen-vl-utils 0.0.10,
huggingface-hub 0.30.2, safetensors 0.5.1, numpy 1.24.4, scipy 1.12.0, einops 0.8.0, opencv-python-headless
4.9.0.80, positional-encodings 6.0.3, monai 1.3.0, pillow 11.2.1, SimpleITK 2.3.1, nibabel 5.2.0, setuptools
69.5.1, and the customised dynamic-network-architectures 0.2 shipped in the repo (installed from a copy).
These are the README / requirements.txt pins. The repo's `model/modeling_qwen2_5_vl.py` equals transformers 4.49.0's
file apart from import paths (plus a training-only `_wSEG` subclass), which confirms the transformers version.

## Deviations from the official recipe

1. **Volume sweep.** The method is 2D; every axial slice is used as a key slice and the 2D masks are stacked
   (protocol above). Zero padding at the volume ends follows the training datasets. The context order
   (cranio-caudal, the usual order of axial Radiopaedia stacks) is an assumption: the slice order of the training
   stacks is not documented, and the processing script for the public datasets (`dataset/process_other_datasets.py`)
   is referenced in the README but not in the repo.
2. **Input from arrays, not image files.** Windowed uint8 slices are passed in memory: identical to lossless PNG
   input, no JPEG compression.
3. **Tokenizer.** `inference.py` points `tokenizer_path` at `checkpoints/ominiabnorm-ct-7b/extended_tokenizer`,
   which the HF release does not contain (the tokenizer files in `ominiabnorm-ct-7b/` lack `<SEG>`). The official
   `tokenizer_path=None` branch is used: it adds `<SEG>` as a special token, id 151665, which matches the 151666-row
   `lm_head` / `embed_tokens` in the checkpoint. Checked at load: both tensors loaded, 56/56 LoRA-B tensors
   non-zero, all 272 U-Net tensors present (no missing / unexpected keys).
4. **Second generation round skipped in the sweep** (it only writes the report; the mask is fixed before it). It is
   run once per case on the answer's key slice for `raw.report`.
5. **No `torch.compile`.** `inference.py` wraps the VLM in `torch.compile`, but then calls `model.generate`, which
   the compiled wrapper forwards to the uncompiled module, so it has no effect.
6. **Batching.** 32 key slices per batch (the official evaluation uses 8; all prompts have the same length, so there
   is no padding). bf16 kernels depend slightly on the batch shape: re-running the coordinate check with a different
   batch composition changed the rank-1 score in the 4th decimal and the number of small false-positive components
   (50 vs 40 over 25 slices); the rank-1 components were the same.
7. **Window choice.** The repo has no inference-time windowing for HU volumes; the windows come from
   `dataset/config/other_datasets.json` (training windows of the public lesion datasets; training picks one at random
   with +-20 % jitter). Chosen before any run (PROMPT.txt), not tuned.
8. **Environment.** torch 2.0.0 as the CUDA 11.8 build (H100 needs sm_90); pillow 11.2.1 because the pinned 11.2.0 was
   never published; opencv headless (no libGL); setuptools added (triton 2.0 imports it). CPU preprocessing runs with
   one thread per worker (the shared box is oversubscribed; outputs are bit-identical across thread counts).

## Coordinate check (README rule 3): PASS

`coord_check.py` (log `logs/coord_check.log`, result `coord_check.json`) on pool scan `…100953483028192176989979435275`
(spacing 2.5 x 0.703 x 0.703 mm, shape 133 x 512 x 512): 10 mm sphere of +300 HU at cached index (83, 240, 352), the
deepest lung voxel (same scan and point as SAT-Pro / VISTA3D).

* I/O path: original and synthetic arrays through the run's preprocessing for key slices 75-91; the difference of
  the 9-channel U-Net inputs, every channel c of key slice z mapped back with `recover_single_slice` to cached slice
  z + 4 - c, reduced with the run's rule: recovered (83.00, 240.00, 352.00), error 0 voxels. The VLM image of the key
  slice: recovered (83.00, 240.00, 352.00), error 0. The per-channel differences confirm the context order.
* End to end (VLM `<SEG>` -> U-Net -> recover -> stack, key slices 71-95): the model segments the synthetic sphere as
  "lung nodule": rank-1 component at (82.76, 239.56, 352.78), 0.87 mm from the known index, 99.3 % of the sphere's
  voxels in the mask, equivalent diameter 14.9 mm. In the original scan there is no component there (nearest 23.7 mm
  away); its rank 1 is the real 10.4 mm nodule (1.3 mm from the annotated centre).

## Smoke tests (training-pool cases only, the same cases as SAT-Pro / VISTA3D)

`smoke/` (`smoke_luna_check.txt`, `smoke_liver_check.txt`, masks in `smoke/masks/`):

* LUNA16 `…979435275`, `…764325545`, `…502185026`: rank 1 hits in 1 of 3 (the 10.3 mm nodule of `…502185026`; ranks 3
  and 4 there hit the 6.1 and 9.2 mm nodules). In `…979435275` the 10.4 mm nodule is rank 3 behind two false
  positives; in `…764325545` (548 slices, 637 components) the 12.7 mm nodule is not in the top 10.
* Liver `liver_0`, `liver_5`, `liver_14`: rank 1 lies inside a 12 mm tumour in liver_0; liver_5 has no tumour
  >= 10 mm (rank 1 outside the liver); in liver_14 rank 1 is liver parenchyma and the 13 mm tumour is rank 4.
* Nothing was changed after the smoke tests.

### Behaviour that shapes the result

* **The VLM emits `<SEG>` on every slice** (all key slices of all smoke cases; also for unrelated queries such as
  "intracranial hemorrhage" on a chest slice, checked on pool slices). This is how the model was trained:
  `format_two_round_data` always has `<SEG>` as the first assistant turn, and negative referring queries are
  supervised only through an empty mask and the second-round answer "I don't see any relevant abnormalities on the
  image." So the per-slice presence decision is the U-Net's, and the official evaluation scores that mask
  regardless of the second-round text.
* The U-Net returns non-empty masks on most slices (e.g. 89 of 133 slices of the first pool scan, 146 components),
  so almost every scan gets an answer and the mean-probability ranking has many false positives to choose from.
* Sizes are overestimated: 14.9 mm for the 10 mm sphere, 13.1 mm for the 10.4 mm nodule.

## Runtime

Eval run (`run_eval.sh`, 05:32-06:22 on 2026-10-16, 50 min wall clock; the two processes in parallel on GPU 4; the
CPU box was shared and oversubscribed, load average 115-310 on 224 cores):

* LUNA16: 200 scans = 50,725 key slices in 3009 s including a 60 s model load: 14.7 s per scan, 0.058 s per key slice
  (the liver process shared the GPU for the first 27 min).
* Liver: 53 cases = 23,849 key slices in 1624 s including a 60 s load: 29.5 s per case, 0.066 s per key slice.
* The once-per-case two-round report plus component labelling: median 1.7 s (LUNA16) / 2.7 s (liver) per case.
* Throughput comes from batching (32 key slices, 8 single-threaded DataLoader workers). The paper reports 1.01 s per
  sample for text-guided grounded report generation (one key slice, both rounds, A100).
* Coordinate check about 3 min; smoke tests 209 s (3 LUNA16 scans) and 300 s (3 liver cases), each including the
  model load.

## Failures

* No crashes in the eval run; every eval case has a line; every axial slice of every case was a key slice
  (74,574 in total); `validate_results.py`: FORMAT OK, no centroid outside the cached array.
* No empty answers: `present` for 200/200 LUNA16 scans and 53/53 liver cases. `<SEG>` on 100 % of key slices; the
  U-Net mask is non-empty on 82.9 % of LUNA16 key slices (per scan median 81 %, range 59-98 %) and on 48.1 % of liver
  key slices (median 49 %, range 30-97 %). Components per scan: median 538 (range 155-5127) for LUNA16, 416
  (109-1440) for the liver.
* Answer sizes: equivalent diameter median 12.6 mm (4.5-81.2) for LUNA16, 28.1 mm (5.8-149.7) for the liver
  (some answers are large components spanning many slices).
* Answers outside the target organ, judging from the model's own report on the answer's key slice: 178 of 200 LUNA16
  reports use lung terms and 8 describe abdominal organs (e.g. "a well-defined hypodense mass lesion ... within the
  pancreatic tail"); 40 of 53 liver reports mention the liver and 10 describe other organs only (e.g. "a mass in the
  pancreatic head"). No report says "I don't see any relevant abnormalities"; the re-run of the key slice emitted
  `<SEG>` in all 253 cases.
* Development only (not the eval run): in the first smoke attempt a DataLoader worker segfaulted on the second case
  (workers were forked per case after the main process had used the tokenizer and CPU threads); fixed by one
  persistent worker set for all cases before any eval run.

## Training-data overlap

From the paper (Sec. 5.3, App. D.1-D.2, Table 9) and `dataset/config/other_datasets.json`:

* OmniAbnorm-CT is trained on OmniAbnorm-CT-14K (Radiopaedia images, no overlap with our data) **plus public 2D
  lesion-segmentation data** at a 6:4 ratio in the grounding task, and the U-Net is first pre-trained on CT lesion
  segmentation. The list (Table 9; "all the datasets in the official ULS-23 challenge and 10 additional ones") and the
  config's per-dataset windows include **LIDC-IDRI** (750 "images", nodule masks) and **MSD Liver** (131 = every
  labelled MSD Task03 volume, so the column counts scans), as well as IRCADB3D, MSD Lung, LNDb, DeepLesion and others.
* **LUNA16** is a subset of LIDC-IDRI. The 750 LIDC-IDRI scans match the ULS23 challenge's LIDC-IDRI set
  (de Grauw et al., arXiv 2406.05231, Table 1: 750 scans, 2236 lesions, built on the LUNA16 reference of Jacobs et
  al. 2016), so our 200 eval scans, each with a >= 6 mm consensus nodule, were very likely seen with nodule masks.
* **MSD Task03**: all 53 of our liver eval cases are among the 131 labelled cases listed. 3D-IRCADb-01, also listed,
  is part of LiTS/MSD Task03 as well.
* Which scans were used, and any held-out split, are not published: the processing script is missing from the repo,
  the HF dataset covers only OmniAbnorm-CT-14K, and the checkpoint files hold no data lists. So the overlap cannot be
  counted per case. The public data were used as generic "segment all abnormalities" grounding samples; the
  text-guided task itself was trained on Radiopaedia queries only (`_get_other_data_item` is not implemented for it).

## External liver test set (HCC-TACE-Seg, 93 cases)

* **Protocol: exactly the MSD liver run.** Same query, window [-140, 260], every-slice sweep, mask -> answer rule
  and batch size; `run_omniabnorm.py --ds liver_ext` maps `liver_ext` to the liver query and window (two dictionary
  entries, no other change). Nothing was tuned or inspected on these cases. Invocation `run_eval_liver_ext.sh`, log
  `logs/eval_liver_ext.log`, output `results/sota/omniabnorm_ct_liver_ext.jsonl` (93 lines); GPU 3 only,
  `OMP_NUM_THREADS=8`, 28.5 GB on the GPU.
* **Coordinate check: PASS** (`coord_check.py --ds liver_ext --case HCC_002`, `coord_check_liver_ext.json`, log
  `logs/coord_check_liver_ext.log`). HCC_002 is the first case with the median 2.5 mm slice spacing; HCC_001 has 5 mm
  slices, which a 10 mm sphere barely spans. Every cached case is an eval case, so the sphere (10 mm, +300 HU) was
  placed from the image alone: middle slice, the body's centre row, 30 % of the body width from the patient's right,
  i.e. (54, 241, 161), at 111 HU (3 x 3 x 3 mean 87 HU, soft tissue). I/O path: recovered (53.95, 240.94, 161.09),
  error (-0.05, -0.06, +0.09) voxels, for the U-Net input (all channels) and the VLM image. The residual comes from the
  saturated window over uneven liver tissue; the lung check gave exactly 0. End to end (key slices 42-66) the model
  does not segment the sphere as its own "liver tumor": 40 % of the sphere's voxels fall inside a 77 mm component
  (rank 5, centroid 29.8 mm away), against 3.7 % of the same voxels in the original. Rank 1 is the same 11.7 mm
  component with and without the sphere. A bright +300 HU ball does not look like a portal-venous HCC, and the
  liver query fires on large regions (see the answer sizes below).
* **Runtime:** 93 cases = 8,352 key slices in 484 s including a 30 s model load: 4.9 s per case, 0.054 s per key slice.
* **Failures:** none. Every case has a line, every slice was swept, and `validate_results.py` reports FORMAT OK.
  `present` for 93/93; `<SEG>` on 100 % of key slices; non-empty masks on 75.1 % of key slices (per case median 76 %,
  range 38-100 %); components per case median 190 (22-1245); answer equivalent diameter median 33.0 mm
  (9.6-180.9). 72 of 93 reports on the answer's key slice mention the liver and 9 describe other organs only.
* **Training-data overlap: none documented.** HCC-TACE-Seg (TCIA, MD Anderson) is not in the paper's training data:
  not in Table 9 / `other_datasets.json`, not among the ULS23 datasets the paper says it covers, and not in CCC-18.
  CCC-18's source collections (TCIA page) are CPTAC-GBM/CCRCC/PDA/HNSCC/UCEC, TCGA-BLCA/COAD/HNSC/UCEC/LUSC, NSCLC
  Radiogenomics and Anti-PD-1 Melanoma/Lung, with no HCC-TACE-Seg or TCGA-LIHC. OmniAbnorm-CT-14K is Radiopaedia
  material.

## Score

One run of `score_sota.py omniabnorm_ct` (log `logs/score.log`), after all outputs and these notes were final:

    omniabnorm_ct            fresh1 hit 0.42 largest 0.34 diamMAE 6.4 answered 100 | fresh2 hit 0.41 largest 0.32 diamMAE 7.4 answered 100
    omniabnorm_ct            liver hit 0.60 (n=53, answered 53)

Neither number is a clean held-out result: LIDC-IDRI (the source of LUNA16) and all labelled MSD Task03 cases are in
the model's lesion-segmentation training data (see above). The large diameter errors follow from the oversized 2D
masks (coordinate check: 14.9 mm for a 10 mm sphere).

External liver set: one run of `score_sota.py omniabnorm_ct` after the liver_ext output and its section above were
final (log `logs/score_liver_ext.log`; the LUNA16 and MSD lines it reprints are unchanged):

    omniabnorm_ct            liver_ext hit 0.45 (n=93, answered 93)

This is the clean held-out liver number (no documented training overlap). The MSD liver hit of 0.60 includes cases
that are in the training data.
