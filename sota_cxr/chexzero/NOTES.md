# CheXzero (Tiu et al., Nat. Biomed. Eng. 2022), zero-shot

Outputs:
- `results/sota_cxr/chexzero.csv`: the official ensemble of 10 checkpoints, as in the paper. **This is the reported row.**
- `results/sota_cxr/chexzero_single.csv`: the single checkpoint `best_64_5e-05_original_22000_0.864.pt`, for reference.
- `results/sota_cxr/chexzero_parts/*.npy`: per-checkpoint probabilities.

Code: `run_chexzero.py`; environment: `setup_env.sh`.

## Result (official test split, frontal images; `python sota_cxr/score.py chexzero chexzero_single`)

| | mean, 13 findings | mean, 5 competition findings |
|---|---|---|
| **ensemble (10 checkpoints)** | **0.678** | **0.777** |
| single checkpoint (best_64_5e-05_..._0.864) | 0.656 | 0.763 |

Per-finding AUROC of the ensemble:

| finding | AUROC | finding | AUROC |
|---|---|---|---|
| Atelectasis | 0.714 | Lung Opacity | 0.652 |
| Cardiomegaly | 0.761 | Pleural Effusion | 0.860 |
| Consolidation | 0.717 | Pleural Other | 0.552 |
| Edema | 0.833 | Pneumonia | 0.629 |
| Enlarged Cardiomediastinum | 0.534 | Pneumothorax | 0.786 |
| Fracture | 0.438 | Support Devices | 0.801 |
| Lung Lesion | 0.539 | | |

## Versions
- **Code.** rajpurkarlab/CheXzero at commit 5c341db (2023-08-27), cloned to `/raid/home/CAMCA/hj880/ext_src/CheXzero`. Its `zero_shot.py` (`load_clip`, the zeroshot-classifier logic, `run_softmax_eval`), its `clip.py` / `model.py` / `simple_tokenizer.py` and `data_process.preprocess` are imported unchanged.
- **Checkpoints.** The 10 files in the README's Google Drive folder (downloaded with gdown 2026-10-16) are in `/raid/home/CAMCA/hj880/hf_home/chexzero_weights`. They are loaded with torch's `weights_only=True` default. sha256 prefixes:
  - best_128_0.0002_original_15000_0.859 620ee11d836f6510
  - best_128_0.0002_original_8000_0.857 c1ec95b173e8dcd0
  - best_128_5e-05_original_22000_0.855 1e3ae7be8d92caf0
  - best_64_0.0001_original_16000_0.861 8f5f2679d205476b
  - best_64_0.0001_original_17000_0.863 ba3f39d6981c2521
  - best_64_0.0001_original_35000_0.864 40ad37ea1933b6ae
  - best_64_0.0002_original_23000_0.854 060d9743c5580b35
  - best_64_5e-05_original_16000_0.858 6af517141430a61a
  - best_64_5e-05_original_18000_0.862 0bc03704af2c5db9
  - best_64_5e-05_original_22000_0.864 567a2efbabc6a023
- **ViT-B/32 skeleton.** `load_clip(pretrained=True)` builds it from OpenAI's ViT-B-32.pt, which is SHA-checked by `clip._download` and saved to `hf_home/clip`. Every parameter is then overwritten by the CheXzero state dict (strict load).
- **venv** `/raid/home/CAMCA/hj880/venvs/sota_cxr_chexzero`: Python 3.12, torch 2.8.0+cu128, torchvision 0.23.0, opencv-python-headless 5.0.0.93, Pillow 12.3.0, numpy 2.5.4.

## Pipeline as run (the recipe of `notebooks/zero_shot.ipynb`)
1. **Images.**
   - `data_process.img_to_hdf5`, step by step: `cv2.imread` the original JPEG, BGR to RGB, then PIL `preprocess(desired_size=320)`.
   - That resizes the longer side to 320 with ANTIALIAS and pastes the result centred on a black 320 x 320 'L' canvas, kept as float.
   - `CXRTestDataset` repeats the image to 3 channels. `zero_shot.make` applies `Normalize(101.48761, 83.43944)` and then `Resize(224, bicubic)` (pretrained=True).
2. **Prompts.** The notebook's pair `("{}", "no {}")` with the CheXpert label names, e.g. "Pleural Effusion" / "no Pleural Effusion". The CLIP tokenizer lowercases them.
3. **Score.** As in `run_softmax_eval`, on the raw cosine similarities (no logit scale): p = exp(pos) / (exp(pos) + exp(neg)). The ensemble score is the mean of the 10 checkpoints' p (`ensemble_models`).

## Deviations (none change the arithmetic)
- **No .h5 file.** Images are preprocessed in memory with the repo's own `preprocess`, so no copy of the data is written. The repo's .h5 route would hold the same float arrays.
- **GPU batches.** Image and text embeddings are computed in batches of 128 on the GPU in float32, with TF32 off, instead of one image at a time on the CPU. CLIP has no batch-dependent layers.
- **Library versions.**
  - `Image.ANTIALIAS` (removed in Pillow 10) is aliased to `Image.LANCZOS`, the same filter.
  - torchvision 0.11, the repo's pin, resized tensors without antialiasing. The resize therefore uses `antialias=False` explicitly, because newer torchvision defaults to True.
- The repo pins torch 1.10; torch 2.8 was used. The model code is plain CLIP.

## Training-data overlap with the MIMIC-CXR test split: **yes, leakage**
- CheXzero was trained on **all of MIMIC-CXR**. The paper says: "Each of the 377,110 chest X-rays in the MIMIC-CXR dataset were re-sized to 224 × 224 ...". The preprocessing script (`run_preprocess.py` → `get_cxr_path_csv`) walks every `.jpg` under `files/` with no split filter.
- So **the official test images and the impression sections of their reports were in the contrastive training data** (for multi-image studies, the frontal image was used).
- The CheXpert labels used here are extracted from those same reports. CheXzero's MIMIC-CXR test AUROCs are therefore not a clean held-out measurement and are likely optimistic.
- Model selection and the ensemble ranking used the CheXpert validation set, not MIMIC.

## Runtime
113 s in total on GPU 3, which was shared with other jobs:
- image preprocessing: 20 s with 32 processes;
- the 10 checkpoints: about 9 s each, including the ViT-B/32 skeleton download on the first one.
