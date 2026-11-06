# E19: vehicle hunt match model from simulated second camera views

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_19-13-27` |
| Type | calibration |
| Author | Kushal |
| Started | 2026-11-06 19:05 IST |
| Git commit | `152f649` |
| Status | done |
| Compute | Local CPU (Apple M3), about 1.5 minutes |

## Hypothesis
The hunt score (target photo against a vehicle track from a camera feed) was a hand-weighted sum of appearance, model and colour agreement with hand-set floors. A logistic model fitted on labelled pairs should replace those values with learned ones and give a threshold with a known false-positive rate.

## Setup
- **Proxy task:** there is no labelled set of the same vehicle seen by two Bengaluru cameras. A second camera view is simulated from each Stanford Cars validation crop with box jitter, flip, 20 to 60% downscale and upscale, gamma, per-channel colour gain, blur and heavy JPEG compression.
- **Pairs:** for each vehicle, one positive (reference crop with its own second camera view), one hard negative (another vehicle of the same model) and three negatives of other models.
- **Features:** appearance cosine of the pooled classifier feature map, Bhattacharyya overlap of the temperature-scaled class probabilities, colour agreement (1 same, 0.5 neighbouring names, 0 different).
- **Fit and evaluation:** vehicles are split in half. The model and threshold are fitted on one half and every number below is measured on the other half.

## How to reproduce
```bash
python3 scripts/calibrate_hunt.py --fpr 0.05 --out configs/hunt_model.json
```

## Results
Split used for these numbers: **val (held-out half of the validation vehicles)**

| Metric | Value |
|---|---|
| Pairs (fit / evaluation) | 1978 / 1989 |
| AUC, full model | 0.9808 |
| AUC, appearance only | 0.9822 |
| AUC, appearance and model overlap (no colour) | 0.9811 |
| TPR at 1% / 5% / 10% FPR | 58.6% / 89.7% / 96.3% |
| Operating point (threshold 0.516) | TPR 89.7%, FPR 5.1% |
| FPR by negative kind | other model 0.0%, same model other vehicle 22.4% |
| Colour name agreement on same-vehicle pairs | 67.2% |

Learned weights: appearance 6.91, model overlap 0.95, colour 1.16, bias -7.11.

## Analysis
- Appearance carries almost all of the signal. Model overlap and colour do not raise the held-out AUC (0.981 against 0.982 for appearance alone). They are kept because they make the score explainable to an officer, and their learned weights are small.
- The colour name changes in a third of same-vehicle pairs under simulated camera colour casts, so colour cannot be used as a hard filter.
- The hard case is another vehicle of the same model: at the 5% operating point about one in five such pairs is accepted. A hunt result is therefore a ranked list of leads for a person to confirm, not an identification.

## Decision
Ship the fitted model and the 5% FPR threshold (`configs/hunt_model.json`). The hand-set weights, floor, ceiling and minimum score were removed from `configs/investigation.yaml`.

## Limitations
- The second camera view is simulated. Viewpoint change between cameras (front against rear) is not simulated, so real-world recall will be lower.
- Stanford Cars images are mostly clean web photos. The negatives contain no number-plate or livery cues, which a real deployment could add.
