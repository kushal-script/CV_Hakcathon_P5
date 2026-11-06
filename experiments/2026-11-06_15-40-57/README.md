# E13: temperature scaling and decision thresholds on val

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_15-40-57` |
| Type | calibration |
| Author | Kushal |
| Started | 2026-11-06 15:20 IST (folder created 15:40) |
| Status | done |
| Compute | Local CPU |

## Hypothesis
Temperature scaling lowers ECE substantially, and thresholds chosen on val to meet fixed accuracy targets give a reliable back-off (class → family → make → abstain) plus an out-of-catalogue gate.

## Setup
- **Logits:** the deployed model exactly: INT8 classifier on fine-tuned-detector crops, val split (n = 815).
- **Temperature:** single scalar T minimising val NLL (Guo et al. 2017).
- **Targets:** class-level answers ≥ 97% correct (τ_hi), family-level ≥ 97% (τ_fam), make-level ≥ 95% (τ_make). Each threshold is the smallest value meeting its target with at least 20 accepted images.
- **Out-of-catalogue gate:** τ_ood = 5th percentile of the maximum logit on val (95% of in-catalogue images pass). It can only be tuned on in-distribution data until a web-image dev set exists.
- **Detector thresholds:** τ_lo = 0.10, τ_det = 0.25 from E06 (val recall 99.14%, 0/100 COCO dev false alarms).
- **Uncertainty:** 500 bootstrap resamples of val, refitting T and every threshold.

## How to reproduce
```bash
python3 scripts/calibrate.py --logits runs/e11_int8_logits/logits_val_pred_int8.npz --out runs/e13_calibration --boot 500
```

## Results

| Metric | Value |
|---|---|
| Val Top-1 of the deployed model | 95.09 |
| Temperature T | **0.769** (the model is under-confident, a known effect of label smoothing) |
| ECE before → after | **11.10% → 2.02%** |
| MCE before → after | 51.7% → 74.0% (dominated by near-empty high-confidence bins) |
| AURC | 0.0237 |
| Accuracy at 80 / 90 / 95 / 100% coverage | 98.16 / 97.41 / 96.77 / 95.09 |

| Threshold | Value | 95% bootstrap CI |
|---|---|---|
| τ_hi (class) | 0.74 | [0.46, 0.96] |
| τ_fam (family) | 0.57 | [0.18, 0.94] |
| τ_make (make) | 0.05 (floor: make accuracy already ≥ 95% everywhere) | [0.05, 0.05] |
| τ_ood (max logit) | 2.475 | [1.95, 2.93] |

Decision levels on val: class 760 (97.0% correct), family 8 (87.5%), make 6 (100%), "possibly ⟨make⟩, not a known model" 41 (82.9% correct make).

## Observations
- Calibration is the cheapest large win: one scalar removes almost all miscalibration.
- The wide CIs on τ_hi and τ_fam come from the small val set (815 images, about 4 per class). They are reported rather than hidden.
- The out-of-catalogue gate costs about 5% of genuine catalogue cars, which receive a make-level answer instead of a model. This trade-off will be revisited with real out-of-catalogue photos.

## Artifacts
`runs/e13_calibration/thresholds.yaml` (frozen, packaged into `models/`), `runs/e13_calibration/calibration.json`.
