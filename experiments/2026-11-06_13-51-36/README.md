# E05: TResNet-L@368 main classifier run

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_13-51-36` |
| Type | deep |
| Author | Kushal |
| Started | 2026-11-06 13:51:36 IST |
| Git commit | `e7858a9-dirty` |
| Status | done |
| Compute | JarvisLabs A100 40GB (PCIe), run `r_80d05f09`, 40 epochs in about 25 min (shared GPU with E06 for most of it) |

## Hypothesis
A full fine-tune of an ImageNet-1k-only TResNet-L at 368 px on GT box crops (10% padding, box jitter) reaches 94–95% val Top-1 with a 40-epoch cross-entropy recipe, below the 96.0% the TResNet paper reports with 80 epochs, 1-cycle and a triplet loss.

## Setup
- **Data / split:** 7,329 train / 815 val (stratified 90/10 of the official train split, seed 42, `assets/splits/val_ids.txt`). Test untouched.
- **Model:** `timm` `tresnet_l.miil_in1k_448` (ImageNet-1k only), new 196-way linear head, input 368×368, normalisation from `pretrained_cfg` (mean 0, std 1).
- **Training:** AdamW, LR 2e-4 body / 2e-3 head, weight decay 0.05 (none on norms and biases), batch 32, 2-epoch warm-up then cosine to 1e-6, label smoothing 0.1, bf16 autocast, channels-last. Last epoch used (no early stopping).
- **Augmentation:** box jitter (scale ±10%, shift ±5%), horizontal flip, RandAugment (n=2, m=7), down-then-up scale (p=0.2), CLAHE or gamma (p=0.2), RandomErasing (p=0.25). No CutMix or Mixup.

## How to reproduce
```bash
bash scripts/remote/train_classifier.sh e05_tresnet_l_368 --backbone tresnet_l.miil_in1k_448 --size 368 --epochs 40 --batch 32
```

## Results
Split used for these numbers: **val (n = 815), GT crops**

| Metric | Single | Flip TTA |
|---|---|---|
| Top-1 (196-way) | **95.34** | 95.71 |
| Top-5 | 99.02 | 99.14 |
| Make-level Top-1 | 99.02 | 99.14 |
| Family Top-1 (year twins merged) | 95.83 | 96.20 |
| Year-twin confusion | 6.56 | 6.56 |
| Rebadge-twin confusion | 8.33 | 8.33 |
| ECE (before → after temperature scaling, T = 0.753) | 11.38 → 1.85 | |
| Accuracy at 80% / 90% coverage | 98.31 / 97.55 | |

Validation curve: 88.7 (epoch 5), 90.4 (10), 91.5 (15), 93.7 (20), 94.5 (25), 94.4 (30), 95.3 (35), 95.3 (40). Full history in `logs/history.csv`, metrics in `metrics.json`.

## Observations
- The model is under-confident (T < 1), a known effect of label smoothing. Temperature scaling fixes ECE almost entirely.
- Make-level accuracy (99.0%) is far above model-level accuracy. Most remaining errors stay inside the right make, which supports the make-level back-off in the decision policy.
- Year twins remain the hardest group: 6.6% of their images are predicted as the twin year.

## Failure cases seen
- To be filled from the E15 test analysis (confusion pairs, Grad-CAM/CAM evidence).

## Conclusion & next step
- Target met without the paper's longer recipe. This checkpoint is frozen as the deployed classifier. Next: INT8 export (E11), calibration on deployed-precision logits (E13), test evaluation once (E15).

## Artifacts
Local copies (git-ignored, too large for git): `runs/remote/work-runs/e05_tresnet_l_368/` (`model.pt` 217 MB, `last.pt` 649 MB with optimiser state, `export/classifier.onnx` 216 MB, `val_logits.npz`). Full GPU log: `runs/remote/jl-runs/r_80d05f09/output.log`.
