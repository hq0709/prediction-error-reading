# MedGemma 1.5 4B (Google, 2026), yes / no prompting

Output: `results/sota_cxr/medgemma.csv` (3,403 rows, all 13 findings). It is merged by `merge_parts.py` from `medgemma_part0of2.csv` (GPU 3) and `medgemma_part1of2.csv` (GPU 4).

Code: `run_medgemma.py`; environment: `setup_env.sh`.

## Result (official test split, frontal images; `python sota_cxr/score.py medgemma`)
Mean AUROC over the 13 findings: **0.673**. Mean over the 5 competition findings: **0.729**.

| finding | AUROC | finding | AUROC |
|---|---|---|---|
| Atelectasis | 0.606 | Lung Opacity | 0.630 |
| Cardiomegaly | 0.729 | Pleural Effusion | 0.833 |
| Consolidation | 0.680 | Pleural Other | 0.615 |
| Edema | 0.798 | Pneumonia | 0.634 |
| Enlarged Cardiomediastinum | 0.462 | Pneumothorax | 0.618 |
| Fracture | 0.666 | Support Devices | 0.816 |
| Lung Lesion | 0.656 | | |

## Versions
- **Weights.** `unsloth/medgemma-1.5-4b-it`, HF snapshot de0155051f54, an ungated mirror of `google/medgemma-1.5-4b-it` (the gated Google repository is not accessible with our HF token). Gemma3ForConditionalGeneration, bf16, SDPA attention.
- **venv** `/raid/home/CAMCA/hj880/venvs/sota_cxr_medgemma`: Python 3.12, torch 2.8.0+cu128, transformers 4.57.3 (the version recorded in the checkpoint), accelerate 1.10.1, Pillow 12.3.0.
- **Tokenizer warning.** transformers prints a "fix_mistral_regex" warning for this tokenizer. It is the known false positive for configs saved with 4.57.3 (see `sota/medgemma15/NOTES.md`). "Yes", "yes", "No" and "no" are each single tokens.

## Pipeline as run
- **Messages.** The official quick-start notebook's non-thinking format:
  - system: "You are an expert radiologist.";
  - user: [question text, image];
  - `apply_chat_template(add_generation_prompt=True)`;
  - the default Gemma3 processor (RGB, 896 x 896, mean and std 0.5) on the original JPEG.
- **Question**, one per finding: `Does this chest X-ray show {phrase}? Answer with only one word: yes or no.` The phrases (`PHRASE` in the code) are the lowercased label names, written out where the bare label is not English:
  - Edema: "pulmonary edema";
  - Lung Lesion: "a lung lesion (nodule or mass)";
  - Pleural Other: "a pleural abnormality other than effusion (such as pleural thickening)";
  - Support Devices: "support devices (such as lines, tubes or pacemakers)".
- **Score.** P(yes) at the first answer token, renormalised over the yes / no tokens: (p(Yes) + p(yes)) / (p(Yes) + p(yes) + p(No) + p(no)).
- **Format check (validate images only).**
  - With the first wording ("... Answer yes or no."), the model began about half of its answers with "Based on ...": the yes / no tokens carried a median 0.65 of the first-token mass on 3 validate images.
  - With the one-word instruction the median was 0.999 (minimum 0.987) on 20 validate images, so that wording was used.
  - On the test run the yes / no mass had median 0.999 and minimum 0.978; no prompt fell below 0.5.
  - The criterion was format compliance only. No accuracy was looked at, and nothing was run on test images before the final run.
- **Shared vision pass.** The 13 prompts of an image share one pass of the vision tower. The features are scattered into the token embeddings exactly as `Gemma3Model.forward` does, and `token_type_ids` keep the bidirectional image attention.
  - On 40 validate images, `--standard` (plain `model(**processor_outputs)`) and the fast path gave **identical** P(yes) (max |difference| < 5e-5).
  - The fast path is about 2.3x faster.

## Deviations
None from the model's documented usage. The prompt wording is ours; the user asked for yes / no prompting. Thinking mode was not enabled.

## Training-data overlap with the MIMIC-CXR test split
- The model card lists MIMIC-CXR (2.1.0) **images and reports** as training data, and Chest ImaGenome (boxes on MIMIC-CXR). It does not say which split.
- Its MIMIC-CXR classification results use "radiologist adjudicated labels, described in Yang (2024) Section A.1.1". That Med-Gemini paper (arXiv 2405.03162) states: "We used the MIMIC-CXR training set (237,912 images) to fine-tune Gemini ... We further employed the test cases of MIMIC-CXR as a benchmark".
- This suggests the official test split was held out. It cannot be verified for MedGemma's own data mixture.
- The model has certainly **seen MIMIC-CXR train images and reports**, so this is not zero-shot on MIMIC.

## Runtime
- Test run: two shards in parallel (GPU 3 and GPU 4, both shared with other jobs at 100% utilisation): 667 s and 625 s. That is about 11 min of wall clock for 3,403 images x 13 prompts, or 0.37 s per image per GPU.
- Format check and path-equivalence runs on validate images: about 3 min in total.
