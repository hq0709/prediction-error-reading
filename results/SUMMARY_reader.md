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
| PatchCore, WideResNet-50 (not predictive; 2-D axial cells of 4 mm, bank of 500k normal cells from 60 training scans) | 0.10 | 0.22 | 0.32 | 0.39 | 0.50 |
| PatchCore, DINOv2-base (same, 7 mm cells) | 0.10 | 0.20 | 0.24 | 0.27 | 0.41 |

PatchCore (`ct_patchcore.py`) is the strongest non-predictive model of normal on radiographs (below), but in lung CT
its top addresses sit on pleural and hilar boundaries (64 % within 5 mm of the lung edge): on a slice a vessel in
cross-section looks like a nodule, and only the 3-D context separates them. (A first bank drawn from interior cells
only put 98 % of addresses on the boundary, recall 0.12; the bank now holds the same cells that are scored.)

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

Non-predictive baseline on the same cases (`ct_patchcore.py --organ liver`; bank of 500k normal-liver cells from 60
world-model training cases; hit = an address inside a target tumour; recall at K = 1 / 5 / 10):

| Addresses | MSD eval (53) | HCC-TACE-Seg (93) |
|---|---|---|
| Liver world model (final candidates) | 0.68 / 0.83 / 0.87 | 0.56 / 0.81 / 0.83 |
| PatchCore, DINOv2-base | 0.38 / 0.51 / 0.64 | 0.45 / 0.71 / 0.83 (K=20 0.88) |
| PatchCore, WideResNet-50 | 0.06 / 0.15 / 0.28 | 0.11 / 0.28 / 0.41 |

In the near-uniform liver a memory of normal patches is a serious competitor (as the darkest-points baseline was):
the world model leads at small K and ties at K=10 on the external set. In the lung it leads by 0.63 at K=10.

## One reader for all three tasks (final reader, 2026-10-16)

`runs_reader/q8_joint3`: Qwen3-VL-8B, LoRA on language + vision, trained once on the union of the lung (final
candidates, no measurement text), liver (MSD pool + CRLM, two-row strips) and disease-course training sets. Chosen over
the per-task readers on each task's own validation rows (AUROC lung 0.986 vs 0.964, liver 0.998 vs 0.998, course 0.935
vs 0.939). Per-task rules as chosen before: lung WM x reader, liver and course max P(yes).

| Task (evaluation) | per-task reader | **one reader** | oracle |
|---|---|---|---|
| Lung nodules >= 6 mm, fresh1 / fresh2 | 0.84 / 0.85 | **0.87 / 0.88** | 0.94 / 0.96 |
| Liver tumours >= 10 mm, MSD | 0.85 | 0.83 | 0.87 |
| Liver tumours, HCC-TACE-Seg (external) | 0.73 | 0.75 | 0.83 |
| New or grown lesions, autoPET pairs | 0.80 | 0.80 | 0.90 |

One reader, lung: largest nodule 0.66 / 0.73, diameter MAE 1.5 / 1.9 mm, side 0.88 / 0.91, candidate AUROC 0.975 /
0.977. Disease course: lesion recall 0.53, progression AUROC 0.89 (prior findings only 0.80, world model only 0.20).

Disease course, three-fold cross-validation over the 240 training patients (each fold's world model and reader trained
without the fold's patients, on 2/3 of the data): 272 pairs, 105 with a target: hit 0.59 (folds 0.74 / 0.53 / 0.53),
oracle 0.85, progression AUROC 0.84 (0.90 / 0.78 / 0.86). The official-split 0.80 rests on 20 pairs and is optimistic.

Prior-finding outlines on the disease-course strips (cyan contour of the registered baseline lesions on both rows):
better on the 27 validation pairs (hit 0.70 vs 0.60, candidate AUROC 0.956 vs 0.943) but worse on the official
evaluation pairs (0.70 vs 0.80, AUROC 0.87 vs 0.90). Decided, by a rule fixed beforehand, on the three-fold
cross-validation over training patients (105 pairs with a target): without outlines hit 0.590 and progression AUROC
0.836; with outlines 0.581 and 0.750. Not adopted; the final strips have no outlines.

Selective answering with the final reader (answer only the cases where the chosen candidate's P(yes) is highest):

| Task | accuracy at 50% / 70% / 100% coverage | confidence AUROC |
|---|---|---|
| Lung (200 scans) | 0.98 / 0.95 / 0.88 | 0.81 |
| Liver, MSD (53) | 1.00 / 0.97 / 0.83 | 0.95 |
| Liver, external HCC-TACE-Seg (93) | 0.96 / 0.94 / 0.75 | 0.90 |
| Disease course (20 pairs with a target) | 0.90 / 0.93 / 0.80 | 0.83 |

Disease-course baseline, LesionLocator (CVPR 2025; tracks each baseline lesion into the follow-up and segments it;
answer = the tracked lesion with the largest growth): pair hit 0.75 (15/20), progression AUROC 0.80 (one upstream
padding bug fixed; its segmenter was trained on autoPET I/II from the same hospital, shared patients not excluded).
Ours on the same pairs: 0.80 and 0.89.

## Chest radiographs, MIMIC-CXR (2026-10-16, in progress)

Radiograph world model (`runs_wm/cxr_nll`): 2-D U-Net, 128 px window with the central 48 px hidden, log-variance
head, trained on 79,886 MIMIC training frontal images labelled "No Finding" (val MSE 0.00163 vs 0.0405 for the
context mean).
- Unsupervised normal vs abnormal (MIMIC test, configuration chosen on validation): AUROC 0.61 (validation 0.72);
  per finding vs "No Finding" 0.57-0.66; left-right mirror 0.54.
- Pointing at NIH ChestX-ray14 boxes (984 boxes, external), top-10 addresses: precision-weighted 0.65, raw error 0.44,
  mirror 0.22, brightest 0.04 (chance per random point 0.08).

13-finding classification baselines on the official test split (3,403 frontal images; AUROC, uncertain labels left out):

| Method | mean 13 | mean 5 competition | note |
|---|---|---|---|
| DenseNet-121, official train split (all ~238k frontal) | 0.786 | 0.807 | no overlap |
| CheXzero (10-checkpoint ensemble) | 0.678 | 0.777 | trained on all MIMIC images and reports incl. test |
| torchxrayvision mimic_ch / mimic_nb / all | – | 0.700 / 0.707 / 0.747 | random-split training, test leakage |
| MedGemma-1.5-4B, P(yes) | 0.673 | 0.729 | MIMIC train in its data |
| BiomedCLIP | 0.552 | 0.568 | |

Readers (Qwen3-VL-8B LoRA, same recipe and 39k rows from 33,276 training images): global / world-model places /
fixed places — training.

Disease course on MIMIC pairs (each frontal study with the patient's previous frontal study, prior registered by a 2-D
affine, median NCC 0.88; report-derived "new finding" labels, not mentioned = negative; 2,752 test pairs). The
prior-conditioned radiograph world model (`runs_wm/cxr_course`, val MSE 0.0022 vs 0.047 for "current = registered
prior") does not separate pairs with a new finding as an image-level score: AUROC 0.56 (single-image world model 0.56,
|current - prior| 0.52; per finding 0.46-0.57). Strictly new findings (prior explicitly negative) are too few for a
benchmark (effusion 58, edema 39, others < 20); MS-CXR-T would be needed.

Latent radiograph world models (`perr/cxr_latent_wm.py`, `train_cxr_latent_wm.py`): a transformer predicts the patch
features of hidden 6 x 6 blocks of a frozen encoder from the features of the image with those blocks blanked in pixels
(no leakage through ViT attention), with a log-variance head, trained on "No Finding" images. MIMIC test, normal vs
abnormal (configuration chosen on validation):

| World model | validation | test |
|---|---|---|
| Pixel (2-D U-Net) | 0.721 | 0.607 |
| Latent, DINOv2-base (natural images only) | 0.753 | 0.643 |
| Latent, RAD-DINO (pretrained on 369k MIMIC images incl. test, unlabelled) | 0.788 | 0.687 |

RAD-DINO latent per finding vs "No Finding": pneumothorax 0.77, support devices 0.76, enlarged cardiomediastinum 0.69,
atelectasis 0.67, edema 0.67, effusion 0.65, lung lesion 0.52, fracture 0.49. In latent space the raw error beat the
precision-weighted one on validation (0.79 vs 0.57).

MIMIC classification, readers so far (official test split, mean AUROC over 13 / 5 findings):

| Reader | training images | mean 13 | mean 5 |
|---|---|---|---|
| DenseNet-121, image only (baseline recipe) | all ~238k | 0.786 | 0.807 |
| DenseNet-121, image only | 60k subset | 0.775-0.778 | 0.803-0.804 |
| DenseNet-121, image + registered prior radiograph | same 60k | 0.767 | 0.797 |
| Qwen3-VL-8B single-finding yes/no reader, image only | 33k | 0.659 | 0.702 |

The prior channel helped on validation (0.789 vs 0.780) but not on test, for nearly every finding. The single-question
VLM reader mostly learned per-question answer priors (pooled validation AUROC 0.83 but per-finding 0.66 on test).

World-model signal as an extra input, tested and not adopted (MIMIC test, mean AUROC 13 findings):
- DenseNet with the pixel world model's z-map as one input channel ([X, Z, X]): 60k labels 0.775 vs 0.778 image only;
  2k labels 0.638 vs 0.662 (validation 0.689 vs 0.708). The 6k / 20k runs were stopped.
- Late fusion of the 60k DenseNet with the latent (RAD-DINO) world model's image score, weights fitted on validation:
  0.7731 vs 0.7729.
- Single-finding VLM reader (Qwen3-VL-8B, 33k training images) with the world model's six places circled and enlarged
  from the original image: test mean AUROC 0.687 vs 0.659 for the global image (5 competition findings 0.724 vs
  0.702), 11 of 13 findings higher (effusion 0.825 vs 0.777, devices 0.847 vs 0.808, pneumonia 0.621 vs 0.566); pooled
  validation AUROC did not show it (0.831 vs 0.833; it is dominated by per-question answer rates). The fixed-place
  control did not train (loss at ln 2 from its first epoch; a re-run with the same seed went the same way and was stopped). These single-question readers barely learn (training loss 0.67-0.68 vs ln 2 = 0.69 for every design), so the multi-label reader below is the test of the evidence views.

Where to enlarge (NIH ChestX-ray14 boxes, 984 boxes; a box counts as covered when its centre lies inside one of the
0.2-side views):

| Views | n | covered | point in box |
|---|---|---|---|
| world model's top addresses (sigma 8, both signs) | 6 | 0.32 | 0.46 |
| six fixed lung-field places | 6 | 0.65 | 0.64 |
| world-model choice among a 4 x 6 lattice / random choice | 6 | 0.48 / 0.35 | |
| fixed 8 / fixed 12 places | 8 / 12 | 0.67 / 0.65 | |
| six fixed + world model's top addresses outside them | 8 / 10 / 12 | 0.68 / 0.71 / 0.74 | 0.68 / 0.70 / 0.71 |

The world model's addresses alone cover less than fixed lung places (its errors concentrate at mediastinal and
diaphragmatic contours); added to the fixed places they cover more than the same number of further fixed places,
most for masses (0.87 vs 0.72) and pneumonia (0.88 vs 0.83). Normalising the error map per location by "No Finding"
statistics made addresses worse (point in box 0.28 at 6). The latent (RAD-DINO) world model localises worse than the
pixel one: point in box at 10 addresses 0.31 raw / 0.39 precision-weighted vs 0.60.

Resolution and evidence views for the DenseNet reader (same 60k training images, recipe of the image-only baseline):
- image only at 512 instead of 320 px: test 0.779 / 0.808 vs 0.778 / 0.803 (validation 0.794 vs 0.789).
- two-stream (whole image + six 224 px views from the original JPEG, ~3.5x the detail of the 320 px input), test
  mean 13 / mean 5: fixed lung places 0.780 / 0.806 (validation 0.801), world-model addresses 0.777 / 0.803
  (validation 0.786), image only 0.778 / 0.803 (validation 0.789). Fixed views help the small findings (pneumothorax
  0.831 vs 0.803, lung lesion 0.715 vs 0.702) and cost a little elsewhere; the world model's addresses add nothing.
- previous radiograph as a second stream (siamese DenseNet: the previous study's frontal image read like the current one, late fusion, a learned term when there is none): test 0.779 / 0.803 (validation 0.798); lung lesion 0.730 vs 0.702, fracture 0.724 vs 0.706, pleural other 0.843 vs 0.870. The same history as an input channel (registered prior, [X, P, X]) had given 0.767.
- The official validation split overstates gains on the official test split (fixed views +0.013 on validation, +0.003
  on test; the prior channel +0.009 / -0.008). 90% of test images have a previous study against 70% in train and
  validation; the test patients are sicker and imaged more often.

Unsupervised normal vs abnormal and pointing, against a non-predictive normal model: PatchCore (memory bank of
600k RAD-DINO patch features from 3,000 "No Finding" training images, nearest-neighbour distance, top-200 mean):

| Model of the normal radiograph | MIMIC test AUROC (validation) | NIH pointing at 10 addresses |
|---|---|---|
| Pixel world model (2-D U-Net) | 0.607 (0.721) | 0.60-0.65 |
| Latent world model, RAD-DINO | 0.687 (0.788) | 0.31-0.39 |
| Latent, RAD-DINO, multi-scale (3 / 6 / 12-patch block lattices, 4 passes per scale; z-scored per patch by "No Finding" validation maps) | 0.698 (0.823) | 0.33-0.38 |
| PatchCore, RAD-DINO (not predictive) | 0.740 (0.834) | 0.64 |
| PatchCore + latent world model (weight on validation) | 0.742 (0.835) | |

A context-predictive model expects what the surrounding context implies; when the context is itself abnormal (diffuse
opacities, effusion, edema) the prediction follows it and the error stays small. A memory of normal patches has no
such dependence, and on radiographs, where most findings are diffuse, it is the stronger model of normal.

Multi-label VLM reader (Qwen3-VL-8B LoRA, one prompt per image, 13 yes/no answers; the same 60k training images as
the DenseNet runs, one epoch): image only, test 0.766 / 0.795 (validation on 600 images 0.770) vs DenseNet 0.778 /
0.803 on the same images. With the previous study's frontal image as a first image: test 0.770 / 0.798 (validation 0.789 vs 0.770). With the evidence view (radiograph with six places circled and enlarged, the single-finding reader's input): world-model places 0.754 / 0.789, fixed places 0.752 / 0.789 (validation 0.769 / 0.770). The composite costs the reader ~0.012 whichever places it shows, and the world model's places are no better than fixed ones; the single-finding readers' +0.028 does not hold for a reader that learns.

## Chest radiographs, second pass: where the radiograph world model fails and what changed (2026-10-17)

Search domain. The radiograph world model's addresses were read over the whole radiograph (letterbox padding removed),
so a third of its top-10 addresses fell on shoulders, neck, bowel gas, ECG leads and markers. The CT pipeline reads
inside the grown lung domain; the radiograph counterpart is now the thorax (`perr/cxr_anat.py`: the ChestX-Det PSPNet
released with torchxrayvision, `cxr_anatomy.py`; lungs, hila, heart, aorta and mediastinum, holes filled, grown 8 px;
the whole radiograph when the segmentation finds no thorax). 99% of NIH box centres lie inside it.

| NIH boxes, organs+8 domain (`cxr_point.py`) | K=1 | K=3 | K=10 | 6 views cover (`cxr_coverage.py`) |
|---|---|---|---|---|
| pixel world model, old protocol (whole radiograph) | 0.12 | 0.32 | 0.65 | 0.32 |
| pixel world model, thorax (sigma 8, brighter) | 0.25 | 0.51 | 0.80 | 0.57-0.59 |
| PatchCore, thorax | 0.32 | 0.55 | 0.78 | 0.65 |
| random thorax points | 0.17 | 0.36 | 0.63 | 0.52 |
| six fixed lung places | | | | 0.65 |

ChestX-Det test (553 images, 1,734 boxes, 13 findings) gives the same order (K=10: world model 0.68, PatchCore 0.67,
random 0.54). MIMIC normal vs abnormal barely moves with the domain (pixel world model 0.613, PatchCore 0.739, latent
0.712 on test; `cxr_ad.py`).

Two attempts that did not help:
- larger hidden regions: the same U-Net on the radiograph downscaled 2x and 4x (8 and 16 cm hidden, the 4x model
  sees the whole radiograph; `train_cxr_wm.py --scale`): MIMIC normal vs abnormal 0.609 / 0.584 (vs 0.613), pointing
  worse (NIH K=10 0.77 / 0.58), their maximum with the fine model worse still (0.65). Effusion and oedema stay
  undetected in pixel space even when the hidden region spans the lung base.
- position: PatchCore with each token's position in the thorax appended (`patchcore_pos.py`): 0.736-0.739 vs 0.739.

Per finding, the world model localises focal findings better than PatchCore and diffuse ones worse (NIH, K=3: mass
0.62 vs 0.56, nodule 0.38-0.48 vs 0.25, effusion 0.39 vs 0.54, atelectasis 0.38 vs 0.45). With the configuration for
focal findings chosen on ChestX-Det train (sigma 2, brighter than expected, 16 px suppression), recall of focal boxes
among the top-K addresses on the held-out sets:

| Focal findings, top-10 addresses | world model | PatchCore | random |
|---|---|---|---|
| ChestX-Det test (nodule, mass, calcification, fracture; 386 boxes) | 0.45 | 0.35 | 0.13 |
| ... nodule / calcification / fracture / mass | 0.41 / 0.45 / 0.41 / 0.82 | 0.31 / 0.21 / 0.35 / 0.82 | 0.06 / 0.01 / 0.17 / 0.56 |
| NIH (nodule, mass; 164 boxes) | 0.81 | 0.60 | 0.30 |
| ... nodule | 0.76 | 0.44 | 0.16 |

Enlarging the focal addresses does not help the MIMIC classifier: two-stream DenseNet with ten ~native-detail views
(0.1-side crops; `make_cxr_crops.py focal`, `train_cxr_evidence.py --crops focal`), same 60k images: test mean 13 0.773
(lung lesion 0.704, fracture 0.685, pleural other 0.801) vs ten random thorax views 0.777 (0.715, 0.689, 0.852) vs image
only 0.778 (0.702, 0.706, 0.870).
