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
