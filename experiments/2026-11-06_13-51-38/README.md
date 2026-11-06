# E08: classical colour/texture/shape features + LinearSVC

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_13-51-38` |
| Type | classical |
| Author | Kushal |
| Started | 2026-11-06 13:51:38 IST |
| Git commit | `e7858a9-dirty` |
| Status | done |
| Compute | Local Apple M3 CPU, 8 worker processes, about 16 min (feature extraction 12 min, SVM fits 4 min) |

## Hypothesis
Global hand-crafted descriptors on GT crops with a linear SVM reach roughly 10–30% Top-1 on 196 classes. Shape (HOG) carries most of the signal and colour adds a little.

## Setup
- **Input:** GT box crop (10% padding) from the crop cache, resized to 128×256.
- **ROI:** box-initialised GrabCut (5 iterations at ≤ 320 px), morphological opening, largest contour; Canny with median-based thresholds.
- **Features (3,989 dims):** HSV histogram 8×4×4 on the GrabCut foreground (128), LAB colour moments (9), uniform LBP P=8,R=1 and P=16,R=2 (28), HOG 9 orientations, 16 px cells, 2×2 blocks (3,780), 8×4 edge-density grid plus overall edge density (33), geometry: aspect, extent, solidity, left-right symmetry, 7 log Hu moments (11).
- **Classifier:** StandardScaler → PCA (256, near-zero-variance components dropped) with whitening → LinearSVC; C chosen on val from {0.005, 0.02, 0.05, 0.2}; temperature fitted on val for probabilities.
- **Bug fixed during the run:** the first fit whitened exactly-zero-variance PCA directions (histogram blocks sum to 1), which drove some ablation arms to chance. Components below 1e-6 of the top variance are now dropped; numbers below are from the fixed run.

## How to reproduce
```bash
python3 scripts/train_classical.py --workers 8
```

## Results
Split used for these numbers: **val (n = 815), GT crops**

| Metric | Value |
|---|---|
| Top-1 (196-way) | **15.21** (chance 0.51) |
| Top-5 | 34.11 |
| Make-level Top-1 | 23.68 |
| Family Top-1 | 15.83 |
| Year-twin confusion | 8.20 |
| Rebadge-twin confusion | 8.33 |
| Best C | 0.005 (C grid: 15.21 / 14.48 / 12.52 / 10.43) |

Feature-block ablation (val Top-1):

| Block | Without the block | Only the block |
|---|---|---|
| HOG (shape) | 9.33 (−5.9) | 13.37 |
| HSV colour histogram | 13.50 (−1.7) | 3.56 |
| Edge-density grid | 13.99 (−1.2) | 3.31 |
| Geometry | 14.72 (−0.5) | 1.96 |
| LBP texture | 15.58 (+0.4) | 3.80 |
| Colour moments | 15.34 (+0.1) | 2.45 |

## Observations
- Shape dominates: HOG alone reaches 13.4% and removing it costs 5.9 points.
- Colour and edge layout add complementary signal, while LBP texture slightly hurts. Car bodies are smooth painted surfaces, so micro-texture is not discriminative between models.
- The classical branch is far below the deep model (95.3%). Its value in the system is interpretability, the intermediate panels, a fully classical comparison row and possible gains under corruption (tested in E12/E14), not clean accuracy.

## Conclusion & next step
- `classical_global.npz` is the live classical model (`runs/e08_classical/`). Next: features on detector crops and saliency ROIs for the C and C+Y comparison rows, then fusion tests in E12.
