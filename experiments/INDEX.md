# Experiment Index

| Timestamp | Title | Type | Author | Status | Split | Top-1 | Notes |
|---|---|---|---|---|---|---|---|
| [2026-11-06_13-51-36](2026-11-06_13-51-36/README.md) | E05: TResNet-L@368 main classifier run | deep | Kushal | done | val | 95.34 | JarvisLabs A100 run r_80d05f09 |
| [2026-11-06_13-51-37](2026-11-06_13-51-37/README.md) | E03+E06: YOLO26s zero-shot and single-class fine-tune | detection | Kushal | done | val | mIoU 0.949, mAP50 0.981 | JarvisLabs A100 run r_b3a1db81 |
| [2026-11-06_13-51-38](2026-11-06_13-51-38/README.md) | E08: classical colour/texture/shape features + LinearSVC | classical | Kushal | done | val | 15.21 | local CPU, GT crops |
| [2026-11-06_15-40-53](2026-11-06_15-40-53/README.md) | E09+E10: crop and augmentation ablations (TResNet-M, 15 epochs) | ablation | Kushal | done | val | 93.25 / 91.04 / 92.39 | JarvisLabs A100 run r_eb8bcb31 |
| [2026-11-06_15-40-54](2026-11-06_15-40-54/README.md) | E11: ONNX export and static INT8 quantization | optimisation | Kushal | done | val | 94.97 (INT8) | instance CPU run r_72de10db |
| [2026-11-06_15-40-56](2026-11-06_15-40-56/README.md) | E12: classical fusion and quality-gated enhancement on val | hybrid | Kushal | done | val | fusion +0.1 (rejected); noise s3 0 → 62 | local CPU |
| [2026-11-06_15-40-57](2026-11-06_15-40-57/README.md) | E13: temperature scaling and decision thresholds on val | calibration | Kushal | done | val | ECE 11.1 → 2.0 | local CPU, INT8 detector-crop logits |
| [2026-11-06_15-40-58](2026-11-06_15-40-58/README.md) | E14: end-to-end robustness suite on 1,960 test images | robustness | Kushal | done | test | noise s1 81.9 → 90.9 | JarvisLabs A100 run r_69aa1469 |
| [2026-11-06_15-40-59](2026-11-06_15-40-59/README.md) | E17: I-JEPA ViT-H/14 linear probe and fine-tune | deep | Kushal | done | val | 65.40 probe, 92.64 fine-tune | JarvisLabs A100 run r_70bef59c |
| [2026-11-06_16-31-27](2026-11-06_16-31-27/README.md) | E15: final test evaluation of the deployed system (single run) | failure-analysis | Kushal | done | test | **94.69** deployed | local; reads test labels via scripts/eval_test.py only |
| [2026-11-06_19-13-27](2026-11-06_19-13-27/README.md) | E19: vehicle hunt match model from simulated second camera views | calibration | Kushal | done | val | AUC 0.981, TPR 89.7% at 5% FPR | local CPU |
