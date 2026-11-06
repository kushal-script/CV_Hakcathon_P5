USN1,USN2,USN3,USN4

# eval_P5_TeamID: Intelligent Fine-Grained Vehicle Recognition and Analysis System

All numbers are on the Stanford Cars **test split (n = 8,041)** unless marked val. Models, thresholds and the enhancement rule were frozen on train/validation before test labels were read; test labels are read only by `scripts/eval_test.py` and every call is logged in `experiments/test_log.csv`.

**Deployed system:** YOLO26s (fine-tuned, 1 class) → crop → TResNet-L@368 (ImageNet-1k pretrained, fine-tuned, INT8) → temperature-scaled decision policy with family/make back-off; classical quality analysis, enhancement with detector-confidence selection, GrabCut/Canny/contour ROI, geometry and a colour/texture/shape SVM for the intermediate panels and the comparison.

## 1. Classification performance: Top-1 and Top-5

| Approach | Input | Top-1 | Top-5 | Make Top-1 | Family Top-1 |
|---|---|---|---|---|---|
| Classical only (HOG, colour, LBP, edges, geometry → LinearSVC) | saliency ROI | 2.96 | 8.73 | 7.66 | 3.01 |
| Classical on detector crop | YOLO crop | 14.74 | 33.27 | 23.38 | 15.04 |
| Deep only, no detection | full image | 91.88 | 97.66 | 95.88 | 92.34 |
| Deep on detector crop (FP32) | YOLO crop | 94.53 | 98.67 | 98.22 | 94.96 |
| Deep on detector crop (INT8) | YOLO crop | 94.49 | 98.68 | 98.22 | 94.93 |
| **Hybrid: deployed system (INT8)** | **raw upload** | **94.69** | 98.94 | 98.37 | 95.15 |
| Deep on GT crop (oracle localisation, diagnostic) | GT crop | 95.21 | 99.22 | 98.77 | 95.63 |
| I-JEPA ViT-H/14, fine-tuned (research row) | YOLO crop | 92.53 | 98.53 | 97.15 | 93.11 |
| I-JEPA ViT-H/14, frozen linear probe (research row) | GT crop | 66.24 | 87.49 | 76.88 | 66.92 |

Visually similar classes (deployed system): year-twin confusion 6.4%, rebadge-twin confusion 10.0%. Make-level accuracy 98.37%.

## 2. Object localisation: IoU

| Detector | Mean IoU | IoU ≥ 0.5 | IoU ≥ 0.75 |
|---|---|---|---|
| YOLO26s, COCO zero-shot (car, bus, truck) | 0.9124 | 95.51 | 94.71 |
| **YOLO26s, fine-tuned (deployed)** | 0.9493 | 99.03 | 98.45 |

## 3. Object detection: mAP

| Detector | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 |
|---|---|---|---|
| YOLO26s zero-shot | 0.9321 | 0.9234 | 0.8696 |
| **YOLO26s fine-tuned (deployed)** | 0.9758 | 0.9727 | 0.9261 |

No-vehicle images: 99/100 held-out COCO test images without vehicles correctly return "no vehicle". Test images with no detection: 0.34%.

## 4. Model efficiency: size and inference time

| Classifier | Size (MB) | Val Top-1 | Test Top-1 (detector crop) | Latency p50, 2 threads, AMD EPYC 7542 | Latency p50, 2 threads, Apple M3 |
|---|---|---|---|---|---|
| FP32 | 206.1 | 95.34 | 94.53 | 354 ms | 316 ms |
| **INT8 (static, QDQ, per-channel)** | **52.9** | 94.97 | 94.49 | 434 ms | 100 ms |

Detector: YOLO26s ONNX, 38 MB. INT8 is 3.9× smaller and 3.2× faster on CPUs with integer dot-product instructions (ARM, x86 VNNI); on the AMD EPYC 7542 (no VNNI) it is slower than FP32.

End-to-end latency of the full product pipeline (all panels and comparison rows), 2 threads, 40 test images, Apple M3:

| Stage | INT8 p50 (ms) | INT8 p95 (ms) | FP32 p50 (ms) | FP32 p95 (ms) |
|---|---|---|---|---|
| io | 1 | 27 | 1 | 27 |
| quality | 4 | 7 | 4 | 7 |
| enhance | 0 | 5 | 0 | 5 |
| detect | 126 | 251 | 125 | 251 |
| classify | 100 | 296 | 316 | 1178 |
| refine_classical | 160 | 337 | 158 | 311 |
| comparison | 145 | 494 | 136 | 338 |
| panels | 8 | 20 | 8 | 19 |
| total | 590 | 1097 | 787 | 1824 |

## 5. Robustness: performance under varying conditions

1,960 random test images, end-to-end Top-1 (%). "Deployed" applies the classical enhancement and keeps whichever of the original or enhanced image gives the more confident detection (rule chosen on validation). Speckle noise and Gaussian blur were never used for tuning.

| Condition | Raw | Always enhance | **Deployed** | GT crop | Detection rate raw → deployed |
|---|---|---|---|---|---|
| clean | 94.6 | 94.4 | **94.5** | 95.4 | 100% → 100% |
| contrast_s1 | 94.5 | 94.2 | **94.5** | 94.9 | 100% → 100% |
| contrast_s3 | 93.6 | 94.3 | **94.1** | 94.2 | 99% → 99% |
| contrast_s5 | 83.9 | 91.7 | **91.2** | 90.3 | 51% → 95% |
| defocus_blur_s1 | 85.9 | 85.7 | **86.0** | 86.2 | 100% → 100% |
| defocus_blur_s3 | 33.6 | 33.2 | **33.4** | 31.8 | 99% → 99% |
| defocus_blur_s5 | 6.9 | 6.7 | **6.8** | 6.8 | 93% → 94% |
| fog_s1 | 94.5 | 94.5 | **94.6** | 95.1 | 100% → 100% |
| fog_s3 | 94.3 | 94.3 | **94.2** | 94.8 | 100% → 100% |
| fog_s5 | 93.3 | 92.5 | **92.8** | 93.6 | 99% → 100% |
| gaussian_blur_s1 | 93.4 | 93.4 | **93.4** | 93.8 | 100% → 100% |
| gaussian_blur_s3 | 33.9 | 33.1 | **33.2** | 31.9 | 100% → 100% |
| gaussian_blur_s5 | 2.9 | 3.0 | **3.0** | 2.9 | 86% → 89% |
| gaussian_noise_s1 | 81.9 | 90.4 | **90.9** | 92.6 | 45% → 97% |
| gaussian_noise_s3 | 75.4 | 77.6 | **78.3** | 83.2 | 0% → 81% |
| gaussian_noise_s5 | 38.1 | 42.2 | **38.0** | 40.1 | 0% → 0% |
| jpeg_compression_s1 | 93.3 | 93.3 | **93.4** | 93.5 | 100% → 100% |
| jpeg_compression_s3 | 91.9 | 91.9 | **92.1** | 93.1 | 100% → 100% |
| jpeg_compression_s5 | 82.5 | 82.0 | **82.3** | 87.0 | 94% → 94% |
| lowlight_s1 | 89.8 | 89.1 | **92.1** | 93.1 | 93% → 96% |
| lowlight_s3 | 58.5 | 51.9 | **58.4** | 66.8 | 29% → 30% |
| lowlight_s5 | 13.3 | 11.1 | **13.3** | 15.4 | 0% → 0% |
| motion_blur_s1 | 93.5 | 93.4 | **93.5** | 94.1 | 100% → 100% |
| motion_blur_s3 | 62.6 | 61.9 | **62.3** | 61.4 | 99% → 99% |
| motion_blur_s5 | 22.1 | 22.2 | **22.3** | 21.5 | 94% → 94% |
| occlusion_10 | 93.1 | 93.1 | **93.1** | 94.8 | 100% → 100% |
| occlusion_25 | 84.9 | 84.8 | **84.9** | 93.0 | 100% → 100% |
| occlusion_50 | 42.8 | 43.3 | **43.0** | 84.2 | 98% → 98% |
| pixelate_s1 | 92.7 | 92.6 | **92.7** | 93.5 | 100% → 100% |
| pixelate_s3 | 85.9 | 85.7 | **85.9** | 86.7 | 94% → 94% |
| pixelate_s5 | 61.1 | 60.8 | **61.1** | 58.8 | 34% → 34% |
| resolution_112 | 92.1 | 92.1 | **92.1** | 93.0 | 100% → 100% |
| resolution_28 | 3.4 | 3.5 | **3.5** | 3.5 | 97% → 98% |
| resolution_56 | 53.3 | 53.5 | **53.3** | 54.2 | 100% → 100% |
| speckle_noise_s1 | 85.0 | 92.6 | **92.7** | 93.9 | 83% → 99% |
| speckle_noise_s3 | 78.9 | 88.4 | **88.3** | 88.7 | 14% → 100% |
| speckle_noise_s5 | 69.1 | 75.4 | **75.2** | 78.4 | 1% → 85% |

## 6. Confidence and threshold analysis

* Temperature T = 0.769, fitted on validation (deployed INT8 model on detector crops).
* Expected calibration error on test: 10.36% before, **1.13%** after temperature scaling.
* AURC 0.0200. Accuracy at 80 / 90 / 95 / 100% coverage: 98.21 / 97.68 / 96.79 / 94.47.
* Thresholds (validation, 95% bootstrap CI): tau_hi = 0.740 [0.46, 0.96], tau_fam = 0.570 [0.18, 0.94], tau_make = 0.050 [0.05, 0.05], tau_ood = 2.475 [1.95, 2.93]; detector τ_lo = 0.10, τ_det = 0.25.

| Decision level (deployed, test) | Share of images | Correct at that level |
|---|---|---|
| Make, model and year | 93.46% | 97.22% |
| Model family, year uncertain | 1.31% | 71.43% |
| Make only | 0.36% | 93.10% |
| Possibly ⟨make⟩, not one of the 196 models | 4.54% | 81.10% |
| No vehicle detected | 0.34% | — |

Across all levels, 95.81% of test images receive an answer that is correct at the level given.

## 7. Comparison: classical vs deep vs hybrid

| Approach | Test Top-1 | What it adds |
|---|---|---|
| Classical only (saliency ROI) | 2.96 | Fully hand-crafted pipeline, no learning beyond a linear SVM |
| Classical + deep detector | 14.74 | Deep localisation; recognition still hand-crafted |
| Deep only (full image) | 91.88 | Fine-tuned CNN without detection |
| Deep detector + deep classifier | 94.53 | Detection crop feeds the fine-grained classifier |
| **Hybrid deployed system** | **94.69** | + classical enhancement with detector selection, calibrated back-off, classical panels |

Robustness is where the hybrid differs most (section 5): Gaussian noise s1 81.9 → 90.9, speckle s3 78.9 → 88.3, contrast s5 83.9 → 91.2.

## 8. Ablation study

| Factor (one change at a time) | Split | Top-1 with | Top-1 without | Δ | Significance |
|---|---|---|---|---|---|
| Detection crop vs full image (TResNet-M, 15 epochs) | val | 93.25 | 91.04 | +2.21 | McNemar p = 0.013 (Holm) |
| Robustness augmentation bundle | val | 93.25 | 92.39 | +0.86 | p = 0.167 (n.s. on clean images) |
| Detection crop vs full image (TResNet-L, deployed model) | test | 94.53 | 91.88 | +2.65 | — |
| Fine-tuned vs zero-shot detector (mean IoU) | test | 0.9493 | 0.9124 | +0.0369 | — |
| Classical score fusion (β = 0.05) | val | 94.97 | 94.85 | +0.12 | McNemar p = 1.00 (rejected) |
| INT8 quantization | val | 94.97 | 95.34 | -0.37 | 3.9× smaller |
| Classical enhancement with detector selection, gaussian_noise_s1 | test subset | 90.9 | 81.9 | +8.9 | n = 1,960 |
| Classical enhancement with detector selection, speckle_noise_s3 | test subset | 88.3 | 78.9 | +9.4 | n = 1,960 |
| Classical enhancement with detector selection, contrast_s5 | test subset | 91.2 | 83.9 | +7.3 | n = 1,960 |
| Classical enhancement with detector selection, lowlight_s1 | test subset | 92.1 | 89.8 | +2.3 | n = 1,960 |
| Classical features: HOG block | val | 15.21 | 9.33 | +5.89 | shape carries the classical signal |
| Classical features: HSV colour block | val | 15.21 | 13.50 | +1.72 | |

## 9. Failure-case analysis

430 of 8041 test images are wrong at Top-1 (5.35%).

| Category | Images |
|---|---|
| same make, different model | 256 |
| different make | 70 |
| rebadged twin | 39 |
| same model, different year | 38 |
| no vehicle or rejected input | 27 |

Most frequent confusions:

| True class | Predicted | Count |
|---|---|---|
| Chevrolet Express Van 2007 | Chevrolet Express Cargo Van 2007 | 13 |
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 13 |
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 11 |
| Dodge Caliber Wagon 2007 | Dodge Caliber Wagon 2012 | 11 |
| Audi V8 Sedan 1994 | Audi 100 Sedan 1994 | 11 |
| Dodge Caliber Wagon 2012 | Dodge Caliber Wagon 2007 | 10 |
| Dodge Sprinter Cargo Van 2009 | Mercedes-Benz Sprinter Van 2012 | 8 |
| BMW 6 Series Convertible 2007 | BMW M6 Convertible 2010 | 8 |
| Audi TT Hatchback 2011 | Audi TTS Coupe 2012 | 8 |
| Bentley Continental GT Coupe 2012 | Bentley Continental GT Coupe 2007 | 7 |

Worked cases (figures in `experiments/2026-11-06_16-31-27/figures/`):

1. **Trim twin:** a windowless Express Cargo Van predicted as Express Van at 100%. The CAM sits on the identical front fascia; the discriminative cue (side windows) is ignored.
2. **Same-generation year twin:** Dodge Caliber 2007 predicted as 2012. Both years share one generation; the decision policy's family back-off is the correct output.
3. **Probable label error:** labelled HUMMER H3T, predicted GMC Canyon at 96%; the truck carries a Chevrolet Z71 badge (Colorado, same platform as Canyon and H3T).
4. **Missed detection in clutter:** a Lamborghini on a show stand behind a crowd; three small cars, none detected.
5. **Rejected input:** 6 test images smaller than 64 px were refused by the input guard (since lowered to 32 px; the numbers above keep the original guard).

