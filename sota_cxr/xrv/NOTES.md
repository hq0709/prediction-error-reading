# torchxrayvision pretrained DenseNet-121s (Cohen et al. 2020 / 2022)

Outputs: `results/sota_cxr/xrv_mimic_ch.csv`, `xrv_mimic_nb.csv` and `xrv_all.csv`, 3,403 rows each.

Code:
- `run_xrv.py` (inference);
- `leakage.py` (reconstructs the candidate xrv training splits);
- `leakage_gap.py` (memorisation check);
- environment: `setup_env.sh`.

## Results (official test split, frontal images; `python sota_cxr/score.py ...`)

The models do not predict **Pleural Other** or **Support Devices**: those labels are dropped when the library relabels to its 18 default pathologies. All means below are therefore over the **11 findings they predict**. The 5-competition mean is complete.

| model | mean, 11 predicted findings | mean, 5 competition findings |
|---|---|---|
| `densenet121-res224-mimic_ch` | 0.619 | 0.700 |
| `densenet121-res224-mimic_nb` | 0.624 | 0.707 |
| `densenet121-res224-all` | 0.658 | 0.747 |
| reference: our DenseNet-121 (official train split), same 11 findings | 0.769 | 0.807 |

| finding | xrv_mimic_ch | xrv_mimic_nb | xrv_all | our DenseNet-121 |
|---|---|---|---|---|
| Atelectasis | 0.661 | 0.647 | 0.689 | 0.760 |
| Cardiomegaly | 0.672 | 0.680 | 0.733 | 0.783 |
| Consolidation | 0.647 | 0.664 | 0.681 | 0.744 |
| Edema | 0.735 | 0.767 | 0.809 | 0.858 |
| Enlarged Cardiomediastinum | 0.555 | 0.579 | 0.597 | 0.729 |
| Fracture | 0.504 | 0.454 | 0.530 | 0.739 |
| Lung Lesion | 0.495 | 0.498 | 0.595 | 0.705 |
| Lung Opacity | 0.596 | 0.604 | 0.633 | 0.687 |
| Pleural Effusion | 0.785 | 0.777 | 0.822 | 0.888 |
| Pleural Other | not predicted | not predicted | not predicted | 0.860 |
| Pneumonia | 0.542 | 0.573 | 0.560 | 0.734 |
| Pneumothorax | 0.612 | 0.622 | 0.586 | 0.834 |
| Support Devices | not predicted | not predicted | not predicted | 0.900 |

**Why these numbers are low (diagnostic, not a pipeline fault).**
- The MIMIC models were trained on PA images only, one per patient (see the leakage section), at 224 px. 71 % of the official test images are AP (2,405 AP, 998 PA).
- Split by view, the 11-finding mean for mimic_ch is **0.685 on PA and 0.584 on AP**; our DenseNet gets 0.808 and 0.744.
- A label choice also works against this evaluation. The xrv training labels mask "not mentioned" as unknown (except in "No Finding" studies). The convention here counts it as negative, which adds many hard negatives from sick patients.
- Preprocessing was checked against the documented pipeline. On sampled test images the input ranges and outputs are as expected.

## Versions
- torchxrayvision 1.5.5 (PyPI). The GitHub repo is cloned at commit 4f9c775 in `/raid/home/CAMCA/hj880/ext_src/torchxrayvision`, for reading the training code only.
- Weights were downloaded by the library from the GitHub "v1" release into `/raid/home/CAMCA/hj880/hf_home/xrv` (sha256 prefixes):
  - `densenet121-res224-mimic_ch`: mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt (23b13a04459684ff)
  - `densenet121-res224-mimic_nb`: mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt (a15dc8e9d0880d7b)
  - `densenet121-res224-all`: nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt (56524913dd16a906)
- venv `/raid/home/CAMCA/hj880/venvs/sota_cxr_xrv`: Python 3.12, torch 2.8.0+cu128, scikit-image 0.26.0, numpy 2.5.4, Pillow 12.3.0.

## Pipeline as run (README / training-script preprocessing)
1. Read the original 8-bit MIMIC-CXR-JPG file. PIL gives arrays identical to `skimage.io.imread`; this was checked on 8 images.
2. `xrv.datasets.normalize(img, 255)` maps pixels to [-1024, 1024]; the image is kept as a single channel.
3. `XRayCenterCrop()` takes a square crop of the longer side, then `XRayResizer(224)` resizes with skimage.
4. Output = sigmoid of the logits for the model's `targets`; "Effusion" is mapped to Pleural Effusion. The library's `op_norm` calibration is switched off so the CSV holds plain probabilities. op_norm is monotonic per class, so the AUROCs are unchanged.
5. Nothing was tuned.

## Deviations
None from the documented inference. Two choices: images are decoded with PIL (identical arrays), and op_norm is off (it does not affect AUROC).

## Training-data overlap with the MIMIC-CXR test split: **yes, leakage**

**The official MIMIC-CXR split was not used.** MIMIC-CXR is in the training data of all three models; `all` = NIH + PadChest + CheXpert + MIMIC-CXR (CheXpert labels) + Google + OpenI + RSNA. Each dataset was split at random into the model's own train and test parts.

What the code shows:
- **When the weights appeared.** The mimic_ch / mimic_nb weight URLs were added on 2020-03-30 (commit 13358b1). Before that, the mimic entries pointed to the CheXpert model. The `all` URL is the same since 2020-03-16. The filenames are unchanged today.
- **The loader at that time.** As of 2020-03-30, `MIMIC_Dataset` kept **PA views only** and, by default (`unique_patients=True`), **one image per patient** (`groupby(subject_id).first()`). This matches the paper's "filtering for one AP or PA view per patient".
- **The split in the first public script.** `train_model.py` first appeared 2020-05-20. It split every dataset **50/50 at random** with `torch.manual_seed(0)` + `random_split`.
- **The split in later scripts.** Since 2020-12 the script uses `GroupShuffleSplit(train 0.8, test 0.2, seed 0)` by patient, on PA + AP and every image. Because the weights predate this, it is unlikely to describe them.

What this means for our test set:
- 289 official test patients. 203 of them have a PA image in a labeled study; the other 86 (802 images) were never in the xrv MIMIC data.
- **Code as of the weight release (H2).** An expected ~half of those 203 patients had their first PA image in xrv training. The exact reconstruction gives 109 patients and 109 test images that were training images themselves; those patients have 1,365 test images in total.
- **Current script (H1).** 228 patients (2,691 images) would be in xrv training.

**Memorisation check** (`leakage_gap.py densenet121 xrv_mimic_ch xrv_mimic_nb xrv_all`; output saved in `leakage_gap.txt`):
- For each reconstruction, the test images are split by whether their patient fell in the reconstructed xrv training part or its held-out part.
- If a reconstruction were right and the models had memorised their training patients, the xrv models would score clearly better on the training-part group than on the held-out group, by more than our DenseNet does. Our DenseNet never saw any test patient, so it is the reference.
- Entries below: mean of the per-finding AUROCs over the 11 findings, and the pooled AUROC over all (image, finding) pairs after per-finding percentile ranking.

| group | images | our DenseNet | xrv_mimic_ch | xrv_mimic_nb | xrv_all |
|---|---|---|---|---|---|
| H2, patients in xrv train | 1365 | 0.796 / 0.774 | 0.640 / 0.657 | 0.651 / 0.670 | 0.678 / 0.703 |
| H2, patients held out | 1236 | 0.766 / 0.756 | 0.623 / 0.649 | 0.627 / 0.658 | 0.678 / 0.686 |
| H2, the 109 images used for training | 109 | 0.807 / 0.789 | 0.642 / 0.678 | 0.663 / 0.676 | 0.705 / 0.718 |
| H2, same image slot, held-out patients | 94 | 0.804 / 0.801 | 0.680 / 0.693 | 0.686 / 0.696 | 0.745 / 0.750 |
| H1, patients in xrv train | 2691 | 0.771 / 0.750 | 0.625 / 0.644 | 0.633 / 0.652 | 0.666 / 0.682 |
| H1, patients held out | 712 | 0.763 / 0.756 | 0.589 / 0.633 | 0.583 / 0.633 | 0.620 / 0.676 |

What the check shows:
- **H2.** There is no memorisation signature. The 109 images that would have been training images score *lower* than the matched held-out images for every xrv model. The patient-level gaps are no larger than our DenseNet's.
- **H1.** The xrv gap in mean per-finding AUROC (+0.04 to +0.05) is larger than our DenseNet's (+0.01). The pooled gaps are small (+0.006 to +0.019 against -0.006), and the held-out group is only 61 patients. This is not conclusive; the weights also predate this script.

**Verdict.**
- The xrv MIMIC models are **not clean** on the official test split. They come from a random split of all of MIMIC-CXR.
- Under the code at the time of the weight release, about 109 official test images (3 %), one per patient, were training images, and about half of the test patients with PA images contributed one.
- The exact split cannot be verified, and neither reconstruction shows a clear memorisation effect. So the leakage probably inflates these AUROCs little; they are far below the official-split DenseNet anyway.
- **Flag them as "trained on a random MIMIC split (test patients partly seen)" in any table.**
- The `all` model has the same MIMIC exposure, plus CheXpert and five other datasets.

## Runtime
About 70 s per model on GPU 3 (shared with other users' jobs), including weight download and model load. 16 CPU workers handle the full-resolution JPEG decoding and skimage resizing of the 3,403 images.
