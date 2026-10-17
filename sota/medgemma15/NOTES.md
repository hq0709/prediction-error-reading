# MedGemma 1.5 4B (Google, 2026)

Results: `results/sota/medgemma15_luna.jsonl` (200 rows) and `results/sota/medgemma15_liver.jsonl` (53 rows).
The rows are merged from `parts/` by `run_medgemma.py merge`; the eval was split over GPUs 2 and 3.
Code: `run_medgemma.py` (check / smoke / eval / merge), `common.py` and `validate_results.py` (format check before scoring).
The prompts are in `PROMPT.txt`.

## Versions
- Official code: google-health/medgemma at commit a60a660 (`/raid/home/CAMCA/hj880/ext_src/medgemma`). The recipe comes from `notebooks/high_dimensional_ct_hugging_face.ipynb` (CT input) and `notebooks/cxr_anatomy_localization_with_hugging_face.ipynb` (MedGemma's localization prompt format).
- Weights: `unsloth/medgemma-1.5-4b-it`, an ungated mirror of `google/medgemma-1.5-4b-it`, HF snapshot de0155051f54 (Gemma3ForConditionalGeneration), run in bf16. The gated Google repository is not accessible with our HF token. The mirror's `chat_template.jinja` carries an "Unsloth template fixes" header and behaves like the Gemma-3 template: text items are trimmed and concatenated.
- venv `/raid/home/CAMCA/hj880/venvs/sota_medgemma15` (`setup_env.sh`):
  - Python 3.12.15, torch 2.8.0+cu128, transformers 4.57.3 (the version recorded in the checkpoint), accelerate 1.10.1;
  - numpy 2.5.4, Pillow 12.3.0, SimpleITK 2.5.6, nibabel 5.4.2.
- Hardware: H100 GPUs 2 and 3. GPU 3 was also running another user's 36 GB job during the eval.
- transformers 4.57.3 prints a "fix_mistral_regex" tokenizer warning for this checkpoint. This is a known false positive for configs saved with 4.57.3; it only warns, and the tokenizer is not modified unless `fix_mistral_regex=True` is passed, which I did not do.

## Pipeline as run (CT notebook, verbatim where it applies)
1. **Slices.** Cached LPS slices ordered superior to inferior, as a DICOM series is commonly numbered. If there are more than 85, the notebook's rule picks `[s[int(round(i / 85 * (N - 1)))] for i in 1..85]`. Every LUNA16 scan therefore gets 85 slices, and liver cases with N < 85 keep all N.
2. **Windowing.** RGB = (-1024, 1024), (-135, 215) and (0, 80) HU, each clipped, scaled to 0-255, rounded and cast to uint8. Each slice is then JPEG-encoded (PIL default quality) into a data URL.
3. **Orientation.** The cached slice `vol[z]` is exactly the DICOM pixel layout (rows run anterior to posterior, columns run patient right to left), so no reorientation is needed. The processor resizes each slice to 896x896.
4. **Message.** The instruction, then for each slice an image followed by the text "SLICE k", then the query, sent as a single user turn. Generation uses `apply_chat_template(add_generation_prompt=True, continue_final_message=False)`, bf16, greedy (`do_sample=False`) and `max_new_tokens=2000`.
5. **Location format.** MedGemma's own localization convention: x and y normalised to [0, 1000] across the image. The inverse is z = fed_z[SLICE-1], x = x/1000*512 - 0.5 and y = y/1000*512 - 0.5, in cached LPS index.
6. **Parsing** (`common.parse_json_answer`, fixed before the eval run):
   - drop any thinking block and LaTeX brace escapes, then take the first JSON object after the last "Final Answer";
   - if there is no JSON, record coordinates only if "slice", "x=" and "y=" are all stated, and take presence as stated;
   - if present is false or coordinates are missing or out of range, zyx = null;
   - `diameter_mm` is used as stated.

## Deviations and choices
- **Slice order.** I chose superior to inferior because the notebook orders slices by DICOM InstanceNumber, and our arrays carry no instance numbers.
- **The query had to be developed on training-pool scans.** The CT notebook's own query style left most answers unparseable. In a long multi-slice CT context the model falls into report generation ("FINDINGS: ... IMPRESSION: ...") or into greedy repetition loops that run to 2000 tokens. I compared prompt variants on training-pool scans only (3 + 2 LUNA16 pool scans, 3 + 2 liver pool cases) and chose by the rate of machine-readable answers, with the rule set before seeing the last variant.

| variant | where the format instructions sit | pool scans | JSON parsed | with coordinates | 2000-token loops |
|---|---|---|---|---|---|
| draft0 | CT-notebook style query after the slices | 2 (synthetic) | 0 | 0 | 1 |
| A | localization prompt (Instructions + Query) after the slices | 6 | 3 | 1 | 2 |
| B | as A, "Do not write a report. Output only the final answer" | 6 | 0 | 0 | 5 |
| C | as B, notebook's "instructor teaching medical students" framing | 6 | 0 | 0 | 5 |
| D | Instructions before the slices, Query after | 10 | 1 | 5 (text) | 1 |
| F | task-first framing before the slices, A's query after | 6 | 3 | 1 | 2 |
| **G (used)** | as D, with the JSON requirement restated in the query | 10 | 8 | 8 | 2 |

  G keeps MedGemma's localization wording ("Instructions: ... You MUST output a single parseable json ... Now answer to the user query." / "Query: ... Don't give a final answer without reasoning. Output the final answer in the format "Final Answer: X" ... Answer:").

  The task asks for 2 to 3 pool scans for smoke tests; for this format choice I used 5 per dataset.

- **Coordinates and units on pool scans.** With G, x and y are mostly 500 (image centre), and `diameter_mm` often has a units error: the text says "3.5 cm" and the JSON says 3500. I left these as the model stated them, with no correction.

## Coordinate check (`coord_check.json`, pool scans only)
I put a 10 mm, +300 HU sphere deep in the lung or liver, centred on a fed slice. I located it in the actual `pixel_values` (85x3x896x896, in content order) by differencing against the scan without the sphere, then mapped the image index and centroid back.

| scan | fed slices | sphere found on SLICE (expected) | error back in cached voxels (z, y, x) |
|---|---|---|---|
| LUNA ...870245028 | 85 of 253 | 40 (40) | 0.00, -0.04, 0.11 |
| liver_3 | 85 of 534 | 18 (18) | 0.00, -0.51, -0.15 |

Both are within 1 voxel. The JPEG round trip and the 512-to-896 resize are included in this path.

With the final prompt (G), the model was also run end to end on the synthetic volumes (`logs/check_model_promptG.log`; the draft-prompt run is in `coord_check_draft0_prompt.json`).
- LUNA16: it named "left lower lobe, SLICE 39". The sphere is in the left lung on SLICE 40, so the slice is 3 cached slices off. But it gave x = y = 500 (image centre), 76 mm from the sphere.
- Liver: it named SLICE 18, which is correct, then gave "x=50%, y=50%" and wrote x = y = 50 in the JSON, 150 mm from the sphere.

So the model finds the right slice but does not localise in-plane.

## Runtime
The eval was split over two GPUs and took 68 min of wall clock (03:25 to 04:33).
- LUNA16: 200 scans, 4984 s summed over cases (24.9 s per scan).
  - Part 0/2 (GPU 2): 1427 s.
  - Part 1/2 (GPU 3, shared with another user's 36 GB job at 100% utilisation): 3557 s.
- Liver: 53 cases, 2634 s summed (49.7 s per case). The original NIfTI is read and checked against the cache for every case, under heavy CPU contention.
- Each prompt is about 22,800 input tokens: 85 x 256 image tokens plus text.
- The 46 answers that looped to 2000 tokens (32 LUNA16, 14 liver) took 60 to 90 s each.

Part 1/2 of the liver ran on GPU 2 after part 0/2 finished. The duplicate liver stage chained after LUNA16 part 1/2 on GPU 3 was stopped before it loaded the model, and no case was run twice (checked at merge).

## Failures
**LUNA16 (200 scans):**
- 137 answers had parseable JSON: 132 with coordinates and 5 "present": false.
- 63 had no JSON:
  - 29 stated "no nodule" in report-style text, giving present = false;
  - 33 described nodules in report-style text without coordinates, giving present = true with zyx null; most of these are the 2000-token repetition loops;
  - 1 was unparseable.
- In total, 132/200 answers carry a location.

**Liver (53 cases):**
- 28 had parseable JSON: 20 with coordinates and 8 absent.
- 25 had no JSON: 9 stated absent, 1 gave coordinates in text ("slice ..., x=..., y=...") and 15 were present without coordinates.
- In total, 21/53 answers carry a location.

**Prior-driven coordinates.** x = 500 in 122 of the 132 LUNA16 answers with coordinates and y = 500 in 129 of them; the liver shows the same pattern, with x = 500 in 12 of 21.

**Units in `diameter_mm`.** Often wrong: "3.5 cm" becomes 3500. The median liver `diameter_mm` is 1768, and 10 of 20 values exceed 300. I used these as stated.

There were no crashes.

## Known training-data overlap
Data are from the model card (README of the mirror, "Data ownership and documentation").
- **No LIDC-IDRI, LUNA16, MSD or LiTS dataset is listed.**
- The 3D CT training data are CT-RATE (chest CT, Turkey) and an internal US "CT dataset 1".
- **Indirect 2D overlap is possible through SLAKE**, which is in the training data and whose CT images are slices taken from the Medical Segmentation Decathlon, including liver and tumour cases. PMC-OA figures could also contain public-dataset images.
- Whether any of our eval cases are among those slices is unknown.

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
- **Unchanged.** Same pipeline (85-slice rule, three windows, JPEG), the same final prompt (variant G, MSD liver wording) and the same parser. Nothing was tuned on these cases.
  - Code: `run_medgemma.py check|eval --ds liver_ext --part 0/1`, then `merge --ds liver_ext`.
  - 42 of the 93 scans have at most 85 slices and are fed whole.
- **Coordinate check** (`coord_check_liver_ext.json`, `check_frame_liver_ext.png`) on HCC_002: 85 of 109 slices fed.
  - The sphere was placed at an image-only location (no labels) on a fed slice.
  - It was found on SLICE 42 (expected 42). Error back in cached voxels: (0.00, 0.58, 0.16), within 1 voxel.
  - On the synthetic volume the model looped through a report until 2000 tokens.
- **Runtime** (GPU 5 alone, `OMP_NUM_THREADS=8`): 93 cases in 3948 s (42 s per case). 71/93 answers ran to the 2000-token limit.
- **Failures.**
  - JSON in only 4/93 answers: 2 with coordinates and 2 absent.
  - 89/93 were report-style text ("FINDINGS: ... mass in the right hepatic lobe ... cm"): 79 present without coordinates and 10 stated absent.
  - So 2/93 answers carry a location. Format adherence is much worse than on MSD (JSON in 28/53), although the prompt is the same.
- **Training-data overlap:** HCC-TACE-Seg (TCIA, MD Anderson) is not in the model card's data list. No known overlap.

### Score including liver_ext (second and final scoring run, after the three liver_ext eval runs; the LUNA16/MSD lines are unchanged)
```
m3d_lamed                liver_ext hit 0.14 (n=93, answered 93)
hulumed14b               liver_ext hit 0.04 (n=93, answered 93)
medgemma15               liver_ext hit 0.00 (n=93, answered 93)
```
