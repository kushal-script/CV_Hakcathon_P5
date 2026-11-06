# {{TITLE}}

| Field | Value |
|---|---|
| Experiment ID | `{{ID}}` |
| Type | {{TYPE}} |
| Author | {{AUTHOR}} |
| Started | {{STARTED}} |
| Git commit | `{{COMMIT}}` |
| Status | running |
| Compute | e.g. Kaggle T4 ×1 / Colab A100 / local CPU |

## Hypothesis
What question does this answer, and what result do we expect? (e.g. "Cropping to the GT box improves Top-1 by ≥2 pts over the full image at 384px.")

## Setup
- **Data / split:** train (90% of the 8,144) / val (10%) / test (8,041). Seed: …
- **Model:** backbone, pretrained weights, input resolution, heads
- **Training:** optimiser, LR, schedule, epochs, batch size, augmentation, loss
- **Classical components (if any):** preprocessing, features, classifier
- **Changes vs the previous experiment:** link to `../<timestamp>/`

## How to reproduce
```bash
# exact command(s)
```

## Results
Split used for these numbers: **val / test**

| Metric | Value |
|---|---|
| Top-1 (196-way) | |
| Top-5 | |
| Make-level Top-1 | |
| Mean IoU / IoU≥0.5 | |
| mAP@0.5 / mAP@0.5:0.95 | |
| Model size (MB) / params | |
| Latency CPU p50 / p95 (ms) | |
| ECE (before → after temperature scaling) | |
| Accuracy @ τ / coverage @ τ | |

Figures in `figures/`; full numbers in `metrics.json`.

## Observations
- …

## Failure cases seen
- …

## Conclusion & next step
- …
