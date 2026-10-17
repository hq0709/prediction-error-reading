# LesionLocator on the disease-course task (autoPET IV Longitudinal-CT)

LesionLocator (Rokuss et al., "LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body
Imaging", CVPR 2025, MIC-DKFZ), run in its documented longitudinal mode: segment each baseline lesion from a box
prompt, then track it into the follow-up.

Task. Each of the 64 evaluation pairs (`sota/eval_course.txt`) gets one follow-up location where a lesion is new or
has grown. Targets are new lesions, or matched lesions grown to x1.728 volume, with follow-up equivalent diameter
>= 10 mm. A hit is a point within 3 mm of a target. A pair-level progression score feeds an AUROC.

## What was run

| item | value |
|---|---|
| code | `ext_src/LesionLocator` = github.com/MIC-DKFZ/LesionLocator commit `ef073ba` (2026-09-21, "Faster inference & AGENTS.md"), unmodified, `pip install -e` |
| weights | Zenodo record 15174217 (open access, CC BY-NC-SA 4.0, no login, single version, published 2025-04-08): `LesionLocatorModel.zip`, 14,073,990,099 bytes, md5 `1f9b8a64703346df9bcc532a0a32a07b` (matches the record), extracted to `ext_src/LesionLocator_weights/LesionLocatorCheckpoint` |
| models used | `LesionLocatorSeg/bbox_optimized` (stage 1, because `-t box`) and `LesionLocatorTrack` (stage 2), all 5 folds each, mirroring TTA (always on in both stages) |
| mode | the CLI body of `LesionLocator_track -bl <BL img> -fu <FU img> -p <BL mask> -t box -m LesionLocatorCheckpoint` (README "Lesion Segmentation & Tracking"): baseline lesions are segmented from boxes that LesionLocator derives from the baseline instance mask; each segmentation is tracked into the follow-up (UniGradICON registration of the whole scans inside TrackNet, warp, U-Net refinement on a crop) |
| inputs | original NIfTI files: `inputsTr/<pid>_BL_img_QQ.nii.gz` + `<pid>_BL_mask_QQ.nii.gz` (baseline), `inputsTr/<pid>_FU_img_NN.nii.gz` (follow-up). LesionLocator registers them itself. Never used: our registered baseline, FU masks, CSV FU fields, organiser JSON points, targets |
| runner | `run_lesionlocator.py` (CLI body in a loop, models loaded once; `--shard` or `--claim` to share GPU 1 across processes), `make_plan.py` (inputs per pair), `answers.py` (fixed answer rule), `score_course.py`, `run_sanity.sh` + `sanity_train.py`, `run_eval.sh` then `run_eval_claim.sh` (the one evaluation run, see deviation 2), `progress.py`, `diag_tracking.py` (label-free), `diag_labels.py` (post-run interpretation) |
| outputs | `results/sota/lesionlocator_course.jsonl` (64 lines); per-lesion masks and `done.json` records in `sota/lesionlocator/_work/eval/<pair>/blQQ/` |
| GPU | GPU 1 only (`CUDA_VISIBLE_DEVICES=1`), `OMP_NUM_THREADS=8` |

## Inputs per pair

A pair `<pid>_NN` is one follow-up piece on its native grid. The baseline side can consist of several series
("pieces"). Each baseline piece that goes with this FU piece is one `LesionLocator_track` call (BL piece -> FU piece),
with that piece's lesion mask as the prompt. The pieces used for a pair are the ones in `meta["bl_order"]`. These
are the BL series that our image-only registration matched to this FU series. No labels are involved, and they are
exactly the series whose annotated lesions make up our own method's prior findings, so both methods start from the
same baseline lesions. For 61 of the 64 pairs this means all BL pieces. Three pairs leave out a BL piece whose field
of view does not match the FU series:

* `85d8ce590a_01` drops BL piece 2, a 135 mm head series with 0.45 mm pixels (1 lesion), against a 540 mm trunk FU;
* `85d8ce590a_02` drops BL pieces 0 and 1, both trunk series (14 lesions), against a 144 mm head FU;
* `af032fbcb0_00` drops BL piece 1, a 160 mm head series (3 lesions), against a 795 mm whole-body FU.

TrackNet registers whole scans after resampling both to 175^3, so pairing a head series with a trunk scan would only
feed it mismatched images. In all, 763 baseline lesions in 73 BL pieces are prompted. In 4 pairs the BL mask holds
only label 1 (`7ef605fc8d_00`, `bd4c9ab730_00`, `1d7f7abc18_00`, `a597e50502_00`). LesionLocator then treats it as a
binary mask and splits it into connected components (documented behaviour), which makes no difference when the lesion
is one component.

## Answer rule (fixed before the evaluation run, `answers.py`)

* Every baseline lesion that LesionLocator segmented (stage 1) and whose tracked follow-up mask (stage 2) is non-empty
  is a candidate.
* growth = follow-up mask volume / baseline mask volume. Both masks are LesionLocator's own, so the same model
  measures both time points, as RECIST requires. The annotated baseline volume is stored next to it (`bl_ml_annot`).
* The location is the centroid of the follow-up mask, as a `[z, y, x]` voxel index of the FU image. This is our
  native FU grid: identity direction in LPS, same shape (asserted for every pair in `make_plan.py`). The export
  writes masks with the input image's geometry.
* The answer is the candidate with the largest growth (ties: BL piece order, then lesion order). The progression
  score is that growth. With no candidate, the pair is answered `present: false`, `zyx: null`, score 0.
* `candidates` lists every tracked lesion, largest growth first. `untracked` lists the rest with the reason.
* Bookkeeping correction after scoring. The first answer file counted LesionLocator's empty per-ID prompt slots
  (lesion IDs absent from a BL piece's mask; 120 of them) in `n_bl_lesions` and `untracked`. `answers.py` now skips
  them, and the file was regenerated. `case`, `present`, `zyx`, `score` and `candidates` were verified
  byte-identical, so the scores are unaffected.

LesionLocator can only follow baseline lesions, so it cannot find new lesions. That is expected.

## Environment (`/raid/home/CAMCA/hj880/venvs/sota_lesionlocator`, recipe `setup_env.sh`, full list `pip_freeze.txt`)

Python 3.12.15 (the README recommends 3.12), torch 2.7.1+cu128, numpy 2.5.4, lesionlocator 1.0.0 (editable),
acvl-utils 0.2.6, dynamic-network-architectures 0.3.1, batchgenerators 0.25.5, batchgeneratorsv2 0.3.6,
unigradicon 1.0.6, icon-registration 1.1.9, SimpleITK 2.5.6, scikit-image 0.26.0, scipy 1.18.1, nibabel 5.4.2,
connected-components-3d 4.1.0. The pins in `pyproject.toml` are respected. The paper used PyTorch 2.3.1.

Checkpoint safety: the checkpoints are loaded with `weights_only=False`, so their pickles were scanned first. The only
globals they reference are `collections.OrderedDict`, `numpy.dtype` / `numpy._core.multiarray.scalar`,
`_codecs.encode`, torch storages, `torch._utils._rebuild_tensor_v2` and `torch.device`.

## Training-data overlap with autoPET (paper + checkpoint fingerprints + TCIA metadata)

**Verdict: the model never saw autoPET IV Longitudinal-CT. Its segmentation stage did train on autoPET I/II FDG-PET/CT,
which comes from the same hospital (University Hospital Tübingen) and partly the same disease. We found no shared scan,
but patient-level overlap cannot be ruled out.**

1. **No autoPET IV longitudinal data.**
   - The checkpoint record was published 2025-04-08 and has a single version.
   - The longitudinal CT release (fdat DOI 10.57754/FDAT.qwsry-7t837, record created 2025-05-07) was announced on
     the challenge site on May 8, 2025, after the checkpoint.
   - The paper (arXiv 2502.20985, CVPR 2025) lists no autoPET IV data. The tracking stage was trained on synthetic
     follow-ups and then fine-tuned on a private intra-institutional (Heidelberg) melanoma set: 60 patients,
     159 scans.
   - The checkpoint confirms this. `LesionLocatorTrack/dataset_fingerprint.json` lists exactly 159 training cases.
   - The README mentions that LesionLocator was "featured in the winning solution" of autoPET IV Task 2 (9/25). That
     fine-tuned model is not this checkpoint and was not used.
2. **autoPET I/II in the segmentation stage.**
   - Paper Tab. 4 (pretraining): "AutoPet2, 1014 PET/CT".
   - Paper Tab. 5 (lesion fine-tuning): "autoPET, 500, Melanoma".
   - The released Seg fingerprint (`LesionLocatorSeg/*/dataset_fingerprint.json`, 5227 cases = Tab. 5) holds exactly
     500 cases at 400x400 px, 2.036 x 2.036 x 3 mm. That is the autoPET `CTres` CT resampled to the PET grid. These
     are most likely autoPET's ~500 lesion-positive studies (melanoma, lymphoma, lung cancer), not melanoma only.
   - autoPET I/II (FDG-PET-CT-Lesions) was acquired at UKT Tübingen in 2014-2018 and includes 188 melanoma patients.
     The autoPET IV longitudinal training release (which contains all 64 of our eval pairs) is "from a single site
     (UKT)", melanoma, 300 patients. So shared patients are possible.
3. **Checks for shared scans.** All three found nothing:
   - **Patient IDs.** None of the 300 longitudinal pseudonyms appears among the 900 FDG-PET-CT-Lesions patient IDs
     (TCIA API). The two releases may use different pseudonyms, so this is weak evidence.
   - **Slice count + sex.** Matched against the 1014 TCIA FDG CT series. The only eval scan with a unique match was
     `f2fc990265_BL_img_00` (2471 slices, Biograph128, female) vs `PETCT_642d6c78d6`. Pixel check on two DICOM
     slices: different pixel spacing (0.863 vs 0.922 mm), and the best correlation over all slice offsets was 0.66
     and 0.57, at inconsistent offsets. Not the same scan.
   - **Fingerprints.** Exact (shape, spacing) matches between the 670 autoPET IV scans and the 5227 + 159
     fingerprint entries: 8 hits.
     - All 8 have standard geometry: 3 or 5 mm slices; in-plane spacing from a round FOV, e.g. 400 mm / 512 px =
       0.78125 mm.
     - One is an eval scan, `85d8ce590a_FU_img_00`, and it matches the Heidelberg tracking set.
     - Base rate: 5.7 % of autoPET IV scans share an exact (shape, spacing) with another patient's scan, so the
       8 hits are consistent with coincidence.
     - The 500 autoPET entries are at PET resolution and cannot be compared this way.
   - Eval scans acquired on the Tübingen PET/CT scanner (Biograph128): 10 BL and 5 FU.
4. **What overlap could affect.** If any eval patient is in autoPET I/II, the segmentation network may have seen that
   patient's anatomy, from a different exam, with PET-derived lesion masks. That touches stage 1 (baseline
   segmentation) only. It does not leak follow-up appearance, lesion matching or growth.

## Deviations from the official recipe

1. **Bug fix in the tracker's prompt; the only change to LesionLocator's computation.**
   - In `LesionLocatorSegTracker.track_single_lesion`, the list-prompt path used by `-t box`/`point` runs the
     preprocessor on each per-lesion baseline mask and passes the result to TrackNet unchanged. nnU-Net's
     `crop_to_nonzero` has set seg voxels outside the image's hole-filled nonzero mask to −1. The segmenter clears
     these before deriving prompts (`get_prompt_from_inst_or_bin_seg`: `seg[seg < 0] = 0`); the tracker does not.
   - TrackNet then centres its U-Net crop on `torch.nonzero(warped prompt).mean()`, which counts the −1 voxels.
   - 64 of the 67 distinct evaluation BL scans have 5–12,764 exterior voxels of exactly 0 HU. Those few hundred −1
     voxels, warped to the image border, pull the crop away from small lesions.
   - Evidence on training pair 02522a2b27_00, without the fix (`_work/sanity_nofix`, `logs/sanity_nofix.log`):
     - 3 of 4 lesions came back empty, and the fourth had Dice 0.06.
     - The diagnostic replay showed the warped prompt landing within about 10 mm of the CSV centroid. But a lesion
       warped to z ≈ 194 got a crop over z 31–127.
   - The fix: `run_lesionlocator.py: FixedSegTracker` wraps the preprocessor passed to `track_single_lesion` so that
     negative seg values become 0, the same rule the segmenter applies. `--no_fix` restores the original.
   - Nothing else changes: same networks, folds, TTA, crop rule and export.
2. **Batch driver.** The body of `segment_and_track()` runs in a loop with models loaded once.
   - The run started at 10:25 as 3 static shards on GPU 1. Another user's 34 GB training job had started on GPU 1
     shortly before, which capped the count at 3.
   - At 11:26 the shards were stopped and the same run continued as a work queue of 4 processes
     (`run_eval_claim.sh`, `--claim`: atomic per-piece claim files, largest pieces first). The 9 finished BL pieces
     were kept; the 3 partial ones were deleted and redone.
   - Configuration and code are identical in both phases; phase-1 logs are in `logs/phase1/`.
   - Same classes and constructor arguments as the CLI: `predict_from_files(..., overwrite=True,
     num_processes_preprocessing=1, num_processes_segmentation_export=3)`, `track(...)`, all 5 folds,
     `torch.set_num_threads(1)`.
   - The default skimage resampling was kept; the torch resampling patch from `documentation/inference_speed.md`
     was not applied.
3. **Pairing for multi-series studies** (see "Inputs per pair"). The CLI takes one baseline and one follow-up image.
   Each matched BL piece is one call into the FU piece. Three pairs leave out an off-field-of-view BL piece.
4. **Growth uses LesionLocator's own baseline segmentation** (box-prompted stage 1) as the denominator, not the
   annotated baseline volume. This is a reading of "growth = FU volume / BL volume" for a tracked lesion; the
   annotated volume is recorded in every candidate.
5. Environment: Python 3.12 and torch 2.7.1 instead of the paper's PyTorch 2.3.1. No registered or resampled inputs:
   LesionLocator reads the original NIfTI files and registers them itself (the README has no pre-registered-input
   mode, so our registered baseline was not used).

## Coordinate sanity check (training pairs; `run_sanity.sh`, `sanity_train.py`, `logs/sanity.log`)

Three single-piece training pairs (`perr.autopet.pairs('train')`) were run through exactly the evaluation pipeline,
including the fix. For each baseline lesion that the CSV links one-to-one to a follow-up lesion, we compared
LesionLocator's follow-up centroid with the CSV follow-up centroid, both as `[z, y, x]` on the native FU grid.

| pair | lesions tracked | centroid error (mm), per lesion | Dice with FU annotation |
|---|---|---|---|
| 02522a2b27_00 (hardest: our registration QA 12 mm) | 4/4 | 0.9, 25.2, 3.4, 1.0 | 0.90, 0.00, 0.63, 0.53 |
| 914b62192c_00 | 5/5 | 0.9, 0.9, 0.5, 0.4, 1.0 | 0.76, 0.84, 0.92, 0.84, 0.86 |
| c24cd76e1c_00 | 4/4 | 0.2, 0.1, 0.4, 0.2 | 0.97, 0.95, 0.93, 0.95 |

* 12 of 13 matched lesions land within 3.4 mm (median 0.9 mm).
* The miss (25 mm) is a 0.13 ml baseline node that LesionLocator followed onto an adjacent node; it matches CSV
  lesion 6, a new node. That is a tracking error, not a coordinate error.
* Every output mask has exactly the FU image's size, spacing, origin and direction.
* Volumes track the annotation: LesionLocator's growth ratios 3.34, 6.98 and 4.33 vs CSV 3.59, 7.35 and 3.84 for the
  three grown lesions. The one large over-estimate is a 0.42 ml baseline lesion that LesionLocator segmented at
  0.19 ml (4.88 vs 2.09).
* Without the fix (`logs/sanity_nofix.log`), pair 02522a2b27_00 tracked 1/4 (Dice 0.06). The other two pairs gave
  the same result with or without it.

## Results (one run, scored once: `score_course.py`, `logs/score_course.log`)

| metric | value |
|---|---|
| **pair hit rate** (pairs with a target) | **0.750** (15/20) |
| **progression AUROC** (all 64 pairs) | **0.798** (20 pairs with a target vs 44 without) |
| answered / present | 64 / 60 |
| misses on target pairs | answer on another lesion 3, on no lesion 2, absent 0 |
| target pairs containing a grown matched lesion (reachable by tracking at all) | 19/20 |

How the five misses on target pairs arise. This was evaluation-time analysis after scoring; nothing was changed
because of it.

* `1534b76d32_00`: the only target is a new lesion, which tracking cannot find.
* `3988c7f88e_00`:
  - one target came back with an empty baseline segmentation;
  - the other target, a merging lesion, got an empty follow-up mask after TrackNet misregistered the scans by
    about 220 mm (see below).
* `f2fc990265_00`: all three targets were "tracked", but the follow-up masks are nearly empty (growth ≤ 0.02).
* `496e05e1ae_00` and `bff82d1886_00`: the targets were tracked and grew (1.84x; 5.43x and 1.40–2.57x), but a
  non-target lesion was measured as growing faster (2.87x and 7.79x).

## Tracking behaviour

Label-free diagnostics (`diag_tracking.py`, `logs/diag_tracking.log`):

* 344 of 763 prompted baseline lesions were tracked (45 %).
  - 73 got an empty box-prompted baseline segmentation, mostly tiny lesions (median annotated volume 0.09 ml,
    about 5.6 mm; 44 are under 0.1 ml).
  - 346 got an empty follow-up mask.
* Agreement with our own image-only registration, for tracked lesions:
  - median distance between LesionLocator's follow-up centroid and the place our registration maps the same
    baseline lesion: 8.8 mm;
  - 81 % within 25 mm, 6 % more than 100 mm away (gross misregistration).
* TrackNet's z-window search (BL and FU z-extents differ by more than 60 mm) was active for 33 of 73 BL pieces.
  - Example, `3988c7f88e_00` BL piece 0: a 639 mm lower-body series that our registration places partly below the
    FU's field of view (NCC 0.85, 66 % coverage).
  - TrackNet's MSE window search assumes the baseline lies inside the follow-up. Here it put the warped prompts
    210–240 mm away, almost entirely along z. Only 2 of its 25 lesions were tracked; 2 of the 23 misses also had
    empty baseline segmentations.
  - In a single-fold, no-mirroring replay, the U-Net still segmented something at three of these wrong places. The
    5-fold x 8-flip average erased it.

With evaluation labels, post-run interpretation only (`diag_labels.py`, `logs/diag_labels.log`):

| CSV class of the baseline lesion | tracked |
|---|---|
| matched in this FU piece (UNCHANGED) | 276/471 = 0.59 |
| merging in this FU piece | 14/19 = 0.74 |
| matched in another FU piece | 10/36 = 0.28 |
| disappeared | 41/231 = 0.18 (false tracks) |

Tracked matched lesions land a median 1.4 mm from the CSV follow-up centroid, and 86 % are within 25 mm.
CPM@25 over all 471 matched lesions is 0.51. The paper reports CPM@25 ≈ 0.84–0.86 with 5-fold CV on its own in-house
melanoma data, so tracking transfers much worse to this data. The hit rate is still 15 of the 19 reachable target
pairs, presumably because the answer only needs the fastest-growing tracked lesion, and clearly growing lesions tend
to be large and are tracked more often.

## Runtime and failures

* Download and setup:
  - checkpoint 14 GB from Zenodo (about 20 min with parallel range requests);
  - venv about 5 min;
  - sanity runs 15 min without the fix and 18 min with it (3 pairs, 13 lesions per run).
* Evaluation: wall clock 10:25:24 to 15:49:59 on 2026-10-16 = **5 h 25 min** on one H100 (GPU 1).
  - The GPU was shared with another user's 34 GB training job until about 13:20.
  - Worker count: 3 static shards until 11:26, then a 4-worker queue; down to 3 after an OOM crash at 13:05; 5 from
    13:27, once the other job had ended.
  - Per-piece process time summed over `done.json`: 75,859 s = 21.1 process-hours, about 99 s per baseline lesion.
  - Most of it is single-threaded CPU resampling: LesionLocator re-preprocesses the whole baseline scan for every
    lesion and exports whole-volume logits.
  - Pure GPU time per lesion is roughly 10–20 s (an estimate from the sanity run): 40 TrackNet passes, each with a
    UniGradICON registration, plus 40 segmenter passes.
* One worker crashed with CUDA OOM (13:05, GPU 1 full).
  - The official segmenter's fallback moved its result arrays to the CPU, but the network forward pass then ran
    out of memory as well.
  - The crashed piece `a8f15eda80_00/bl00` was deleted and redone from scratch by another worker.
  - The same fallback absorbed three more OOM events, for 3 lesions of `a8f15eda80_01`: the network ran on the GPU
    and only the result arrays were accumulated on the CPU, so the computation is the same.
* Pairs answered absent, because no lesion was tracked: `85d8ce590a_02` (3 lesions), `cfa0860e83_00` (9),
  `07e1cd7dca_00` (2), `1d7f7abc18_00` (1).
* 4 of 60 answers have a multi-component follow-up mask; the centroid is over all components.

## Suspicious or noteworthy

1. **Upstream bug in the tracker** (deviation 1).
   - Without the fix, nearly every evaluation pair would have been affected, and small lesions in particular would
     have been mis-cropped.
   - Results with the unfixed tracker would mostly measure the bug.
2. **Tracking transfers poorly to autoPET IV**: CPM@25 0.51 here vs about 0.85 in the paper.
   - Main causes: tiny baseline lesions with empty box segmentations; registration failures on
     partial-field-of-view pairs (z-window search); and the 40-pass ensemble suppressing uncertain masks.
   - We give LesionLocator the original scans, as documented. A variant fed our pre-registered baseline might do
     better on partial-overlap pairs, but it was not run (single-run protocol).
3. **The growth score is noisy for small lesions.** The denominator is LesionLocator's own baseline segmentation, so
   a tiny baseline mask inflates the ratio (maximum score 104x). This affects the AUROC ranking more than the hit
   rate.
4. **Overlap**: autoPET I/II data from the same hospital is in the segmentation training set (see above). No shared
   scan was found.
5. 7ef605fc8d and ebf11048b1 (both evaluation patients) are on the challenge's list of 7 "corrupted cases to be
   corrected". The local copy is `Longitudinal_CT_v3.zip`, so presumably corrected, but this was not verified.
