# BiomedCLIP (Microsoft; Zhang et al. 2023), zero-shot

Output: `results/sota_cxr/biomedclip.csv` (3,403 rows, all 13 findings). Code: `run_biomedclip.py`; environment: `setup_env.sh`.

## Result (official test split, frontal images; `python sota_cxr/score.py biomedclip`)
Mean AUROC over the 13 findings: **0.552**. Mean over the 5 competition findings: **0.568**.

| finding | AUROC | finding | AUROC |
|---|---|---|---|
| Atelectasis | 0.432 | Lung Opacity | 0.601 |
| Cardiomegaly | 0.626 | Pleural Effusion | 0.759 |
| Consolidation | 0.543 | Pleural Other | 0.702 |
| Edema | 0.482 | Pneumonia | 0.528 |
| Enlarged Cardiomediastinum | 0.623 | Pneumothorax | 0.555 |
| Fracture | 0.449 | Support Devices | 0.358 |
| Lung Lesion | 0.521 | | |

Several findings are below 0.5. Bare label / "no label" prompts do not suit a model trained on PubMed figure captions, where negated captions are rare. The run is still faithful to the requested protocol (see below).

## Versions
- Weights: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, HF snapshot 9f341de24bfb. The tokenizer comes from `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (snapshot d673b8835373). Both are cached in `HF_HOME=/raid/home/CAMCA/hj880/hf_home`. They were downloaded with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, because the stored HF token is invalid.
- venv `/raid/home/CAMCA/hj880/venvs/sota_cxr_biomedclip`:
  - Python 3.12, torch 2.8.0+cu128;
  - open_clip_torch 2.23.0 and transformers 4.35.2 (the versions pinned on the model card);
  - timm 1.0.30, tokenizers 0.15.2, Pillow 12.3.0.

## Pipeline as run
- **Input.** Each original MIMIC-CXR-JPG file goes through the model's own open_clip `preprocess`: RGB, shorter side resized to 224 (bicubic), centre crop to 224 x 224, CLIP mean and std.
- **Prompts.** As for CheXzero, one positive / negative pair per finding: `"{label}"` and `"no {label}"`, using the CheXpert label names. There is no extra template. The pair was fixed before the first run and nothing was tried on test images.
- **Score.** `logit_scale * image_features @ text_features.T` with L2-normalised features, as on the model card. The probability is the softmax over the two prompts of the pair, positive entry. AUROC depends only on the positive-minus-negative similarity.
- **Weight check.** With the model card's own label set ("chest X-ray", "brain MRI", "bone X-ray", ...) and its template "this is a photo of", all 16 sampled test images were classified as "chest X-ray" (mean p = 0.9995). The weights are loaded correctly.

## Deviations
None from the model-card usage. The prompting follows the CheXzero pair protocol, as requested, not the card's "this is a photo of" template.

## Training-data overlap with the MIMIC-CXR test split
- BiomedCLIP was trained on PMC-15M: 15M figure-caption pairs from PubMed Central open-access articles.
- No MIMIC-CXR labels or reports were used.
- MIMIC images reproduced in published figures could appear incidentally. This cannot be checked and is negligible for this evaluation.

## Runtime
50 s for the 3,403 test images on GPU 3 (shared with other jobs), including model loading, with 16 loader workers.
