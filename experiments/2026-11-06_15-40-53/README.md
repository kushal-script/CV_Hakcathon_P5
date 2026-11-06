# E09+E10: crop and augmentation ablations (TResNet-M, 15 epochs)

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_15-40-53` |
| Type | ablation |
| Author | Kushal |
| Started | 2026-11-06 14:16 IST (folder created 15:40) |
| Status | done |
| Compute | JarvisLabs A100 40GB, run `r_eb8bcb31`, three runs of about 6 min each |

## Hypothesis
- **E09:** training and predicting on the detected vehicle region beats feeding the whole image (single factor: input region).
- **E10:** the robustness augmentation bundle (down-up scaling, CLAHE/gamma, RandomErasing) costs nothing on clean images (single factor: augmentation).

## Setup
Same recipe as E05 except: backbone `tresnet_m.miil_in21k_ft_in1k` at 368 px, 15 epochs, batch 48. One factor changes per arm.

| Arm | Input | Augmentation |
|---|---|---|
| Control | GT crop with jitter | full bundle |
| E09 full image | whole image resized to 368×368 | full bundle |
| E10 no augmentation | GT crop | none (no flip, RandAugment, down-up, CLAHE/gamma, erasing) |

## How to reproduce
```bash
bash scripts/remote/ablations.sh
```

## Results
Split: **val (n = 815)**. Each arm is evaluated on its own input (crop arms on GT crops, full-image arm on full images). McNemar against the control, Holm-corrected.

| Arm | Top-1 | Make Top-1 | Δ Top-1 | Fixed / broken vs control | p (Holm) |
|---|---|---|---|---|---|
| Control (crops + augmentation) | **93.25** | 98.40 | — | — | — |
| E09 full image | 91.04 | 96.81 | **−2.21** | 11 / 29 | **0.013** |
| E10 no augmentation | 92.39 | 97.91 | −0.86 | 6 / 13 | 0.167 |

## Observations
- Using the detected region is a significant gain (+2.2 points). This is the clearest evidence that the detection stage feeding the fine-grained classifier matters, and it matches the literature (Krause et al. 2015).
- The augmentation bundle is not significant on clean images (+0.9, p = 0.17). It is designed for robustness under corruption and low resolution, which clean validation cannot show. Its corrupted-image effect was not measured for this arm, which is a stated limitation.

## Artifacts
`runs/remote/work-runs/e09_control_crops/`, `e09_full_image/`, `e10_no_aug/` (checkpoints, val logits, histories). Statistics: `logs/ablation.json`.
