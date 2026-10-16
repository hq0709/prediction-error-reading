# Prediction-error reading with open models: results (2026-10-15)

Single runs, seed 0. The selection rule (pick the candidate with the highest reader P(yes) among the world
model's top 10) was fixed on the LUNA16 training pool before the trained readers were scored; the reader
configuration (LoRA on the language model only vs language + vision) was chosen by validation AUROC.

## Lung nodules (LUNA16, nodule >= 6 mm; fresh1 = cases 60-159, fresh2 = 160-259, never used for design)

| Method | fresh1 hit | fresh2 hit | largest (f1/f2) | diameter MAE mm |
|---|---|---|---|---|
| One-shot VLM (32B / 72B) | 0.00 | 0.00 | – | – |
| Agent harness, no world model (72B) | – | 0.00 | – | – |
| Agent harness + world model (32B / 72B) | 0.36 | 0.45 | 0.29 / 0.38 | 1.1 / 1.3 |
| World model alone (top-1) | 0.47 | 0.40 | 0.34 / 0.34 | 1.1 / 1.1 |
| PER, zero-shot reader Qwen2.5-VL-72B (K=10) | 0.44 | 0.36 | | |
| PER, zero-shot reader MedGemma-27B (K=10) | 0.39 | 0.38 | | |
| PER, zero-shot reader Qwen3-VL-32B (K=10) | 0.51 | 0.43 | | |
| PER, zero-shot reader Qwen3-VL-8B (K=10) | 0.57 | 0.50 | 0.40 / 0.40 | 1.2 / 1.3 |
| PER, trained reader Qwen3-VL-8B (K=10) | **0.71** | **0.73** | 0.59 / 0.61 | 1.2 / 1.5 |
| Oracle: a target among the top 10 | 0.74 | 0.77 | | |

Candidate AUROC (nodule >= 6 mm vs not): zero-shot 8B 0.885 / 0.904; trained 8B 0.987 / 0.984.

### Ablations (trained Qwen3-VL-8B unless noted, K=10, fresh1 / fresh2)

| Variant | hit | candidate AUROC |
|---|---|---|
| Final (LoRA LM + vision, three planes, measurement, gt strips) | 0.71 / 0.73 | 0.987 / 0.984 |
| LoRA on the language model only | 0.70 / 0.66 | 0.983 / 0.967 |
| Larger reader: Qwen3-VL-32B, LoRA on the language model only | 0.71 / 0.68 | 0.982 / 0.969 |
| Axial view only | 0.70 / 0.66 | 0.972 / 0.962 |
| No measurement text | 0.68 / 0.69 | 0.981 / 0.972 |
| Trained on world-model candidates only (no gt strips) | 0.69 / 0.71 | 0.985 / 0.967 |
| No training (zero-shot) | 0.57 / 0.50 | 0.885 / 0.904 |
| No reader (world model top-1) | 0.47 / 0.40 | – |
| No world model (densest lung points, same reader) | 0.04 / 0.07 | – |

Recall of the top-K addresses (fresh1 + fresh2, 200 scans): shape world model 0.43 / 0.69 / 0.76 / 0.83 /
0.92 at K = 1 / 5 / 10 / 20 / 50; densest lung points 0.01 / 0.04 / 0.06 / 0.12 / 0.40.

Reading budget (final reader): K = 5 / 10 / 20 / 30 gives 0.64 / 0.71 / 0.78 / 0.77 (fresh1) and
0.69 / 0.73 / 0.75 / 0.76 (fresh2).

Selective answering (final reader, 200 scans): accuracy 0.98 at 50% coverage, 0.95 at 70%, 0.72 at 100%;
P(yes) of the chosen candidate separates right from wrong answers with AUROC 0.95.

Error analysis (final reader, 200 scans): 144 hits; 49 scans have no target among the top 10 (world-model
recall); 7 wrong picks, all but one with P(yes) <= 0.29.

## Liver tumours (MSD Task03, tumour >= 10 mm; 53 held-out scans)

| Method | hit |
|---|---|
| World model alone (top-1) | 0.26 |
| PER, zero-shot reader Qwen3-VL-8B | 0.51 |
| PER, trained reader Qwen3-VL-8B, LM-only LoRA | 0.75 |
| PER, trained reader Qwen3-VL-8B, LM + vision LoRA (val-selected) | **0.85** |
| Oracle | 0.89 |

The liver world model (local-mean hypodense deficit) and the darkest liver points have similar recall
(0.88 vs 0.85 at K=10): near-uniform parenchyma makes a constant expectation almost as good.

## One reader for both organs

A single Qwen3-VL-8B reader trained on the union of the lung and liver training pools (LoRA on the language
model): lung 0.70 / 0.70 (separate lung reader with the same configuration 0.70 / 0.66), liver 0.77 (separate
0.75). With the final recipe (LM + vision LoRA) the joint reader gives lung 0.69 / 0.73 (separate 0.71 / 0.73)
and liver 0.81 (separate 0.85).

## Brain MRI follow-up (LUMIERE, RANO progression; 60 pilot pairs)

Not working yet. Pair AUROC: intensity "no change" world model 0.67; tumour-space world model 0.64; zero-shot
8B reader 0.61-0.73; trained 8B reader (109 training pairs; LM-only LoRA, val-selected over LM + vision) 0.71;
automated enhancing-volume change 0.77.

## Learned anatomy world model with precision weighting (2026-10-16)

The world model is a 3D U-Net that predicts a 64 mm cube's hidden central 24 mm from its surroundings, with its own
log-variance (beta-NLL), trained on nodule-free cubes of 568 training-pool scans (`perr/context_wm.py`,
`runs_wm/ctx_nll`). Candidate addresses are peaks of the smoothed error divided by the smoothed predicted s.d.
Configuration (sigma 2 mm, this weighting, no fusion with the shape prior) fixed on 60 held-out pool scans.

Recall of the top-K addresses (200 evaluation scans, nodule >= 6 mm):

| World model | K=1 | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|---|
| Densest lung points (no world model) | 0.01 | 0.04 | 0.06 | 0.12 | 0.40 |
| Shape prior (mechanical) | 0.43 | 0.69 | 0.76 | 0.83 | 0.92 |
| Learned, raw error | 0.43 | 0.62 | 0.69 | 0.79 | 0.90 |
| **Learned, precision-weighted** | **0.69** | **0.91** | **0.95** | **0.97** | **0.98** |

Reader retrained on the new candidates (Qwen3-VL-8B, LoRA language + vision, `runs_reader/q8_ctx_vis`, val AUROC
0.970). Selection rule re-chosen on the training pool, both with zero-shot scores (WM x reader 0.744 vs max-p 0.648)
and with cross-fitted trained-reader scores on 162 pool scans (`choose_rule_cv.py`: WM x reader 0.821, max-p 0.796,
largest-confirmed <= 0.790): **WM x reader over the top 10**.

| Pipeline (K=10) | fresh1 hit | fresh2 hit | largest | diameter MAE mm |
|---|---|---|---|---|
| Shape prior + reader (previous final) | 0.71 | 0.73 | 0.59 / 0.61 | 1.2 / 1.5 |
| Learned WM + zero-shot 8B reader (WM x reader) | 0.69 | 0.70 | | |
| **Learned WM + trained reader (WM x reader)** | **0.81** | **0.79** | 0.60 / 0.65 | 1.4 / 1.9 |
| (same, max-p rule) | 0.88 | 0.77 | 0.70 / 0.66 | 1.4 / 1.4 |
| Oracle: a target among the top 10 | 0.94 | 0.96 | | |

Candidate AUROC 0.970 / 0.956. Selective answering (WM x reader): accuracy 0.99 at 50% coverage, 0.93 at 70%,
0.84 at 90%; confidence AUROC 0.87.

World-model ablations with the final reader fixed (K=10, max-p): shape prior 0.71 / 0.73; learned without precision
weighting 0.63 / 0.62; densest points 0.04 / 0.06.

Liver: learned liver model (`runs_wm/liver_nll`, 63 training cases) recall at K=1/5/10 0.42 / 0.72 / 0.77 vs local-mean
prior 0.26 / 0.74 / 0.89; the two fused by reciprocal rank (chosen on the pool) 0.64 / 0.81 / 0.85. Reader retrained on
fused candidates: 0.83 (previous 0.85; oracle 0.85 vs 0.89). Not an improvement.

## Baselines reproduced on our benchmark (published code and weights, run as released)

| Method | fresh1 | fresh2 | liver | note |
|---|---|---|---|---|
| SAT-Pro | 0.66 | 0.64 | 0.94* | *45/53 liver eval cases' tumour masks in its training |
| VISTA3D | 0.42 | 0.46 | 0.94* | *44/53 liver eval cases in training/validation; "lung tumor" class |
| M3D-LaMed-Phi-3-4B | 0.00 | 0.00 | 0.08 | organ-sized boxes |
| Hulu-Med-14B | 0.00 | 0.00 | 0.00 | prior-driven answers |
| MedGemma-1.5-4B | 0.00 | 0.00 | 0.00 | centre-of-image answers |

## Final lung configuration (2026-10-16, chosen on the training pool)

Reader input: the three-plane strip without the measurement text (pool cross-fitting, 162 scans, K=10 hit 0.852 vs
0.821 with the text; the harness measurement is still what the report states). Reader `runs_reader/q8_ctx_nomeas`.
Rule: WM x reader over the top 10.

| Final pipeline (K=10, WM x reader) | fresh1 | fresh2 |
|---|---|---|
| hit (nodule >= 6 mm) | **0.84** | **0.85** |
| largest nodule | 0.64 | 0.69 |
| diameter MAE (mm) | 1.5 | 1.8 |
| side | 0.88 | 0.86 |
| candidate AUROC | 0.968 | 0.968 |

Selective answering: accuracy 0.98 at 50% coverage, 0.95 at 70%, 0.89 at 90%, 0.84 at 100%.
Reading budget K = 10 / 20 / 30: 0.84 / 0.84 / 0.83 and 0.85 / 0.85 / 0.82.

Ablations (one component removed; final reader fixed for world-model rows, retrained for reader rows):

| Variant | fresh1 | fresh2 | candidate AUROC |
|---|---|---|---|
| Final | 0.84 | 0.85 | 0.968 / 0.968 |
| Learned world model without precision weighting | 0.61 | 0.63 | 0.977 / 0.977 |
| Mechanical world model (shape prior) | 0.69 | 0.69 | 0.975 / 0.976 |
| No world model (densest lung points) | 0.04 | 0.06 | – |
| No reader training (zero-shot Qwen3-VL-8B) | 0.70 | 0.71 | 0.854 / 0.801 |
| No reader (world model top-1) | 0.68 | 0.69 | – |
| Axial view only | 0.82 | 0.85 | 0.951 / 0.952 |
| LoRA on the language model only | 0.80 | 0.79 | 0.948 / 0.942 |
| Measurement text added | 0.81 | 0.79 | 0.970 / 0.956 |
| Larger reader, Qwen3-VL-32B (with measurement text) | 0.83 | 0.81 | 0.966 / 0.959 |

Baselines on the external liver set (HCC-TACE-Seg, 93 cases, clean): SAT-Pro 0.88, VISTA3D 0.85, OmniAbnorm-CT 0.45;
ours (MSD-trained, fused world model) 0.73. OmniAbnorm-CT on LUNA16: 0.42 / 0.41 (trained on ULS23-LIDC, i.e. LUNA16).

## Disease-course world: autoPET IV Longitudinal-CT (2026-10-16)

Pairs: baseline CT registered onto the follow-up (images only; `perr/autopet.py`, QA median 6.2 mm). Targets: new
lesions, or lesions grown to x1.728 volume (+20% diameter, RECIST-like), >= 10 mm. Evaluation: the official validation
and test patients, 64 pairs, 20 with a target (113 target lesions). Training: 272 pairs of 240 patients.

World model (`perr/course_wm.py`, `runs_wm/course_nll`): a 96 mm cube of follow-up CT with its central 36 mm hidden,
plus the registered baseline cube, predicts the hidden follow-up and its log-variance; trained on cubes that hide no
target. Validation MSE 0.0013 vs 0.0096 for "follow-up = registered baseline". Configuration (sigma 3 voxels,
precision-weighted, positive sign, top 10) fixed on 33 dev pairs of training patients. Candidates: every prior finding
(the baseline's annotated lesions, registered: the patient's history, re-read for growth) plus the world model's top 10.
Reader: Qwen3-VL-8B LoRA (language + vision) on baseline / follow-up strips (`runs_reader/q8_course_vis`, val AUROC
0.939). Rule: max P(yes), chosen on 27 validation pairs of training patients (0.60 vs 0.40 for WM x reader).

| Variant (evaluation pairs) | pair hit | oracle | lesion recall | progression AUROC |
|---|---|---|---|---|
| **Prior findings + learned world model + trained reader** | **0.80** | 0.90 | 0.58 | 0.90 |
| Prior findings only + trained reader | 0.80 | 0.85 | 0.53 | 0.91 |
| Learned world model only (no history) + trained reader | 0.35 | 0.50 | 0.12 | 0.62 |
| Mechanical "no change" (follow-up minus baseline) | 0.00 | 0.00 | – | – |
| Prior findings + learned world model, zero-shot reader | 0.15 | 0.90 | – | 0.52 |

Pair hit: the chosen candidate lies in a target (pairs with a target). Lesion recall: targets with a candidate of P(yes)
>= 0.5 in them. Progression AUROC: pairs with vs without a target, scored by the largest P(yes).

## Liver, final (2026-10-16)

Extra training data: TCIA CRLM (194 cases, colorectal metastases). Liver world model retrained on MSD pool + CRLM
(`runs_wm/liver_crlm_nll`, tag lcrlm), fused with the local-mean prior by reciprocal rank. Reader input: two rows of
views (60 mm and 150 mm), chosen on MSD validation + 40 held-out CRLM cases (hit 1.000 vs 0.974; AUROC 0.985 both); rule
max P(yes) (0.974-1.000 vs 0.949-0.974 for WM x reader there). Final reader `runs_reader/q8_liver_final` on all data.

| Liver (K=10) | MSD eval (53) | HCC-TACE-Seg external (93) |
|---|---|---|
| World model top-1 | 0.68 | 0.56 |
| **Final pipeline** | **0.85** | **0.73** |
| Oracle | 0.87 | 0.83 |
| Candidate AUROC | 0.973 | 0.936 |
| SAT-Pro | 0.94* | 0.88 |
| VISTA3D | 0.94* | 0.85 |
| OmniAbnorm-CT | 0.60* | 0.45 |
| M3D-LaMed / Hulu-Med-14B / MedGemma-1.5 | 0.08 / 0.00 / 0.00 | 0.14 / 0.04 / 0.00 |

*trained on most of the MSD evaluation cases.
