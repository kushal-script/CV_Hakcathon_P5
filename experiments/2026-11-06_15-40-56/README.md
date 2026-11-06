# E12: classical fusion and quality-gated enhancement on val

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_15-40-56` |
| Type | hybrid |
| Author | Kushal |
| Started | 2026-11-06 15:00 IST (folder created 15:40) |
| Status | done |
| Compute | Local Apple M3 CPU (ONNX Runtime, deployed INT8 classifier and detector) |

## Hypotheses
1. **Late fusion:** adding classical SVM log-probabilities to the deep logits (z + β·log p_cls) does not help on clean images. Pre-registered rule: ship fusion only if β > 0 lowers val NLL, raises Top-1 and McNemar p < 0.05.
2. **Enhancement:** the classical quality-gated enhancement (white balance, gamma, NLM denoise, CLAHE) helps on degraded images and must not hurt clean ones.

## Part A: late fusion (val, n = 815, detector crops, FP32 logits)

| β | NLL | Top-1 | Make Top-1 |
|---|---|---|---|
| 0 (deep only) | 0.3112 | 94.85 | 98.65 |
| 0.05 (best NLL) | 0.3107 | 94.97 | 98.77 |
| 0.1 | 0.3110 | 94.97 | 98.77 |
| 0.2 | 0.3136 | 95.09 | 98.90 |
| 0.3 | 0.3186 | 95.09 | 98.90 |
| 0.5 | 0.3382 | 94.60 | 98.40 |
| 1.0 | 0.4597 | 92.39 | 96.56 |

McNemar at β = 0.05 vs deep only: 1 image fixed, 0 broken, p = 1.0. **Fusion is rejected** and ships with β = 0.

## Part B: enhancement selection (val subset n = 200, seed 7, corruptions generated on the fly)
Cells: end-to-end Top-1 / share of images with a detection ≥ 0.10.

| Condition | Raw | Gamma | Gamma + CLAHE | Gamma + light NLM + CLAHE | Full enhancement | **Detector-selected (deployed)** |
|---|---|---|---|---|---|---|
| Clean | 92.5 / 100% | 92.5 / 100% | 92.5 / 100% | 92.5 / 100% | 92.5 / 100% | **92.5 / 100%** |
| Low light s1 | 81.0 / 92% | 50.5 / 60% | 61.0 / 71% | 76.5 / 88% | 75.0 / 86% | **85.5 / 97%** |
| Low light s3 | 16.5 / 28% | 0.5 / 0% | 0.0 / 0% | 0.5 / 0% | 1.5 / 4% | **17.0 / 28%** |
| Low light s5 | 0.0 / 0% | 0.0 / 0% | 0.0 / 0% | 0.0 / 0% | 0.0 / 0% | 0.0 / 0% |
| Gaussian noise s3 | 0.0 / 0% | 0.0 / 0% | 0.0 / 0% | 1.0 / 1% | 62.0 / 80% | **62.0 / 80%** |
| Contrast s3 | 90.0 / 98% | 90.0 / 98% | 91.0 / 99% | 79.5 / 99% | 90.5 / 99% | **91.0 / 99%** |
| Contrast s5 | 43.0 / 55% | 41.5 / 52% | 75.5 / 90% | 2.0 / 15% | 83.5 / 94% | **83.5 / 94%** |

**Deployed rule:** run the classical enhancement when the quality flags fire, run the detector on both the original and the enhanced image, and keep whichever gives the more confident detection. The crop is classified from that same image.

## Observations
- The deep detector is extremely noise-sensitive: at Gaussian noise severity 3 it finds nothing on the raw image. Classical NLM denoising restores 80% detection and 62% Top-1.
- CLAHE rescues severe low contrast (43.0 → 83.5 Top-1).
- On low light, brightening alone exposes the sensor noise and hurts the detector. The detector-confidence selection avoids this and even beats both inputs at severity 1.
- The selection rule is never worse than raw, so the classical stage can only help. This, not score fusion, is where classical CV earns its place in the hybrid.
- Low light at severity ≥ 3 remains unsolved and is reported as a failure case.

## Artifacts
`runs/e12_fusion_fp32/fusion.json`, `runs/e12_enhance/enhance_tuning.json` (copied to `logs/`). Script: `scripts/tune_enhance.py`, `scripts/fusion.py`.
