# Supervised DenseNet-121, trained on the official MIMIC-CXR-JPG train split

Outputs:
- `results/sota_cxr/densenet121.csv`: official test split, 3,403 frontal images.
- `results/sota_cxr/densenet121_validate.csv`: official validate split, 1,959 frontal images.

Code:
- `train.py` (DDP training with epoch selection on validate), launched by `run_train.sh`;
- `predict.py` (test predictions at the chosen epoch);
- environment: `setup_env.sh`.

Checkpoints are in `ckpt/epoch{0..5}.pt`, with `ckpt/history.json` and `ckpt/best.json`. Logs are in `logs/`.

## Result (official test split, frontal images; `python sota_cxr/score.py densenet121`)
Mean AUROC over the 13 findings: **0.786**. Mean over the 5 competition findings: **0.807**. Mean over the 11 findings that torchxrayvision predicts: 0.769.

| finding | test positives / n | AUROC |
|---|---|---|
| Atelectasis | 764 / 3207 | 0.760 |
| Cardiomegaly | 896 / 3284 | 0.783 |
| Consolidation | 211 / 3307 | 0.744 |
| Edema | 726 / 3093 | 0.858 |
| Enlarged Cardiomediastinum | 143 / 3186 | 0.729 |
| Fracture | 87 / 3397 | 0.739 |
| Lung Lesion | 121 / 3388 | 0.705 |
| Lung Opacity | 1098 / 3324 | 0.687 |
| Pleural Effusion | 1095 / 3270 | 0.888 |
| Pleural Other | 63 / 3388 | 0.860 |
| Pneumonia | 342 / 3037 | 0.734 |
| Pneumothorax | 108 / 3384 | 0.834 |
| Support Devices | 1211 / 3399 | 0.900 |

By view: the 13-finding mean is 0.814 on the 998 PA images and 0.762 on the 2,405 AP images.

## Epoch selection (official validate split, 1,959 frontal images, mean AUROC over the 13 findings)

| epoch | 0 | 1 | **2** | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| validate mean AUROC | 0.7817 | 0.7985 | **0.8055** | 0.8055 | 0.8052 | 0.8048 |

Epoch 2 is chosen (0.80555 against 0.80548 for epoch 3). Only the chosen epoch was run on the test split.

## Recipe
- **Data.**
  - The official train split, every frontal (PA / AP) image from `perr.cxr.mimic_frontal()`: 237,972 images.
  - 131,302 come from `cache/cxr/<key>.npy`, the 512 px letterbox. The other 106,670 are read from the original JPEG on the fly. Each is letterboxed to 512 exactly as `perr.cxr.letterbox`, after a JPEG draft (DCT) downscale that keeps at least 512 px; the mean |difference| from the cached arrays is 0.5/255.
  - No image was unreadable. No data were copied.
- **Labels (fixed convention).**
  - Per finding, y = 1 if the CheXpert label is 1.0, y = 0 if it is 0.0 or missing.
  - Uncertain (-1.0) entries are masked out of the loss for that finding.
  - The 2,562 train images whose study has no CheXpert row are negative for every finding.
  - Train positives range from 2,005 (Pleural Other) to 71,448 (Support Devices).
- **Model.** torchvision `densenet121`, ImageNet init (`IMAGENET1K_V1`), with the classifier replaced by Linear(1024, 13). The grey image is replicated to 3 channels and given ImageNet normalisation.
- **Input.** The 512 px letterbox is resized to **320 x 320** (bilinear, antialiased), so the aspect ratio is kept with zero padding.
- **Augmentation** (training only, on the GPU):
  - brightness ±5 % and contrast ±10 %;
  - random affine: rotation ±10°, scale 0.9-1.1, translation ±5 %, zero padding;
  - no horizontal flip, to keep heart laterality.
- **Optimisation.**
  - Multi-label BCE with logits, averaged over the unmasked entries.
  - Adam (lr 1e-4, betas 0.9 / 0.999, no weight decay), 300-step linear warm-up, then cosine decay to 0 over 6 epochs.
  - Global batch 64: DDP over 2 GPUs x 32.
  - bf16 autocast, channels_last, seed 0, single run.
- **Selection and test.** Validation after each epoch. The epoch with the best mean validate AUROC over the 13 findings was used for the test predictions (sigmoid outputs, no test-time augmentation). The test split was read once, after training.

## Deviations
None from the request. The choices within it were 320 px, no flip and 6 epochs with cosine decay. Validate AUROC plateaued from epoch 2 (0.8055 → 0.8048), so more epochs would not change the selection.

## Training-data overlap with the MIMIC-CXR test split
**None.** The official split is patient-disjoint, and only train-split images and labels were used. The validate split was used only to choose the epoch. ImageNet pre-training contains no chest X-rays from MIMIC.

## Versions
- venv `/raid/home/CAMCA/hj880/venvs/sota_cxr_densenet121`: Python 3.12, torch 2.8.0+cu128, torchvision 0.23.0+cu128, Pillow 12.3.0, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1 (not used for scoring).
- ImageNet weights: `TORCH_HOME=/raid/home/CAMCA/hj880/hf_home/torch`.

## Runtime
- **Training:** 49 min of wall clock (15:28 to 16:17, 2026-10-16) on GPUs 3 and 4. Both cards were shared with other jobs at 100 % utilisation, including other users' training and the user's reader jobs.
  - Epochs took 11.1, 8.5, 8.8, 8.6, 6.4 and 5.6 min, including validation.
  - Throughput was 300-540 images/s after warm-up (mean of the progress prints 494).
  - 16 loader workers per process, `OMP_NUM_THREADS=8`.
- **Test and validate prediction:** about 1 min on GPU 4.
- A 6,400-image smoke test of the full pipeline took 3 min beforehand.
