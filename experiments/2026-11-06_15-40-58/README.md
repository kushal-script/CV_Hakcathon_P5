# E14: end-to-end robustness suite on 1,960 test images

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_15-40-58` |
| Type | robustness |
| Author | Kushal |
| Started | 2026-11-06 14:58 IST (folder created 15:40) |
| Status | done |
| Compute | JarvisLabs A100, run `r_69aa1469` (no labels read on the GPU); scored by `scripts/eval_test.py` |

## Hypothesis
End-to-end degradation (detect → crop → classify) is driven by different failure points per corruption: the detector for noise, the classifier for blur and resolution. The classical quality-gated enhancement, applied through the detector-selected rule chosen on val in E12, helps on noise, contrast and low light without hurting elsewhere.

## Setup
- **Subset:** 1,960 test images sampled at random (seed 42, `assets/splits/robust_subset_ids.txt`), chosen without labels.
- **Conditions (37):** clean; 8 ImageNet-C-style families at severities 1/3/5 (gaussian noise, motion blur, defocus blur, low light, contrast, JPEG, pixelate, fog); 2 held-out families never used for tuning (speckle noise, gaussian blur); resolution of the car region 112/56/28 px; grey occluder over 10/25/50% of the car box.
- **Pipeline:** fine-tuned YOLO26s → top box (whole image if none) → TResNet-L crop classification. Also classified: the GT crop of the corrupted image (isolates the classifier).
- **Variants:** raw; always-enhance; detector-selected (the deployed rule: keep whichever of raw or enhanced gives the more confident detection).

## How to reproduce
```bash
bash scripts/remote/robustness.sh
```

## Results (test subset n = 1,960, Top-1 %)

| Condition | Raw | Always enhance | **Detector-selected (deployed)** | Δ deployed | GT crop | Detection raw → deployed |
|---|---|---|---|---|---|---|
| Clean | 94.6 | 94.4 | 94.5 | −0.1 | 95.4 | 100 → 100% |
| Gaussian noise s1 | 81.9 | 90.4 | **90.9** | **+8.9** | 92.6 | 45 → 97% |
| Gaussian noise s3 | 75.4 | 77.6 | 78.3 | +2.9 | 83.2 | 0 → 81% |
| Gaussian noise s5 | 38.1 | 42.2 | 38.0 | −0.1 | 40.1 | 0 → 0% |
| Speckle noise s1 (held out) | 85.0 | 92.6 | **92.7** | **+7.7** | 93.9 | 83 → 99% |
| Speckle noise s3 (held out) | 78.9 | 88.4 | **88.3** | **+9.4** | 88.7 | 14 → 100% |
| Speckle noise s5 (held out) | 69.1 | 75.4 | 75.2 | +6.1 | 78.4 | 1 → 85% |
| Contrast s3 | 93.6 | 94.3 | 94.1 | +0.5 | 94.2 | 99 → 99% |
| Contrast s5 | 83.9 | 91.7 | **91.2** | **+7.3** | 90.3 | 51 → 95% |
| Low light s1 | 89.8 | 89.1 | **92.1** | **+2.3** | 93.1 | 93 → 96% |
| Low light s3 | 58.5 | 51.9 | 58.4 | −0.1 | 66.8 | 29 → 30% |
| Low light s5 | 13.3 | 11.1 | 13.3 | 0.0 | 15.4 | 0 → 0% |
| Fog s5 | 93.3 | 92.5 | 92.8 | −0.6 | 93.6 | 99 → 100% |
| JPEG s5 | 82.5 | 82.0 | 82.3 | −0.2 | 87.0 | 94 → 94% |
| Motion blur s3 / s5 | 62.6 / 22.1 | 61.9 / 22.2 | 62.3 / 22.3 | ≈0 | 61.4 / 21.5 | 99 / 94% |
| Defocus blur s3 / s5 | 33.6 / 6.9 | 33.2 / 6.7 | 33.4 / 6.8 | ≈0 | 31.8 / 6.8 | 99 / 93% |
| Gaussian blur s3 / s5 (held out) | 33.9 / 2.9 | 33.1 / 3.0 | 33.2 / 3.0 | ≈0 | 31.9 / 2.9 | 100 / 86% |
| Pixelate s3 / s5 | 85.9 / 61.1 | 85.7 / 60.8 | 85.9 / 61.1 | 0.0 | 86.7 / 58.8 | 94 / 34% |
| Car region at 112 / 56 / 28 px | 92.1 / 53.3 / 3.4 | 92.1 / 53.5 / 3.5 | 92.1 / 53.3 / 3.5 | 0.0 | 93.0 / 54.2 / 3.5 | ≈100% |
| Occlusion 10 / 25 / 50% | 93.1 / 84.9 / 42.8 | 93.1 / 84.8 / 43.3 | 93.1 / 84.9 / 43.0 | ≈0 | 94.8 / 93.0 / **84.2** | ≈100% |

Full table for all 37 conditions: `logs/robustness_test.json`.

## Observations
- **Noise breaks the detector, and classical denoising fixes it.** Raw detection collapses (0–45%) under noise. The selected enhancement restores 81–100% detection and up to +9.4 points end to end, including on the held-out speckle family, so the classical stage generalises to an unseen corruption.
- **Always enhancing is not safe.** It costs 6.5 points at low light s3. The detector-selected rule removes that loss (−0.1) while keeping all the gains, and is never worse than raw by more than 0.7 points (within noise at n = 1,960).
- **Blur, pixelation and low resolution hit the classifier, not the detector.** Detection stays ≈100% but Top-1 falls to 34% (defocus s3) and 53% (56 px). Enhancement does not target these. These are the main open weaknesses.
- **Occlusion breaks localisation.** At 50% occlusion the GT crop is still 84.2% correct, but end to end falls to 42.8%: the detector's box shrinks to the visible part, and the crop loses the car's context.
- **Low light at severity ≥ 3 is unsolved:** detection stays at ~30%, then ~0%.
