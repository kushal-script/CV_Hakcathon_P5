# E22: auto-rickshaw check and naming limits for video

| Field | Value |
|---|---|
| Experiment ID | `2026-11-09_15-19-19` |
| Type | calibration |
| Author | Kushal |
| Started | 2026-11-09 |
| Status | done |
| Compute | local CPU |

## Why

Running nine Bengaluru road clips through the live console showed confident wrong names. Every auto-rickshaw was matched to a car model, for example "Geo Metro Convertible 1993" at 96%, "Lamborghini Diablo" and "AM General Hummer". At night, blurred headlight crops were called "Isuzu D-Max" at 94 to 96%. The car catalogues have no three-wheelers, and the calibration only holds for the kind of photos the models were trained on. So a video track now has to pass three checks before it is named, and auto-rickshaws are recognised as such.

## Auto-rickshaw check

* **Data:** 836 free-licensed photos from Wikimedia Commons, `Category:Auto-rickshaws in India` and its subcategories (`scripts/build_autorickshaw_set.py`; licences, authors and pages in `assets/autorickshaw/sources.csv`). The COCO detector found a vehicle covering at least 5% of the frame in 555 of them, from 290 photographers. Negatives are 2,000 Indian catalogue crops (split by the E18 split) and 1,000 Stanford Cars crops (train images for training, validation images for validation and test).
* **Split:** auto-rickshaw photos are split 70/15/15 by photographer, so no photographer appears in two splits.
* **Model:** logistic regression on the embedding of the deployed Stanford Cars classifier, the same embedding the vehicle hunt uses. Every image is used twice, once cropped the way photos are and once the way video tracks are.
* **Threshold:** the lowest one that calls at most 0.5% of validation cars an auto-rickshaw: 0.884.

| Split | Auto-rickshaws found | Indian cars called auto-rickshaw | US cars called auto-rickshaw |
|---|---|---|---|
| Validation (124 auto, 600 Indian, 300 US crops) | 77.4% | 0.67% | 0.0% |
| Test (224 auto, 600 Indian, 300 US crops) | 76.8% | 1.0% | 0.0% |

About a quarter of auto-rickshaws are still missed. Those can still reach the car catalogues, where the checks below and the decision policy are the remaining safeguards.

## Naming limits for video

The thresholds come from the E14 robustness suite and from measuring crops of 300 (light) and 200 (blur) Stanford Cars validation images under the same corruptions. They were not tuned on the Bengaluru clips.

| Check | Measurement | Threshold |
|---|---|---|
| Size | E14: 92.1% top-1 with the car reduced to 112 px, 53.3% at 56 px | 112 px |
| Light | Median pixel brightness of the crop: 0.375 clean, 0.140 at low light severity 1 (92.1% in E14), 0.066 at severity 2, 0.027 at severity 3 (58.4%) | 0.08 |
| Blur | Laplacian variance at 224 px: medians 12 to 13 for Gaussian and defocus blur at severity 3 (33% in E14), 10th percentile 21 for defocus blur at severity 1 (86%) | 20 |

The median is used for brightness because headlights and street lamps lift the mean of a night crop without making the car any easier to see.

## Catalogue agreement in footage

In E18 the Stanford Cars model named the make correctly for only 29.0% of Indian photos whose make it knows. In footage from Bengaluru a name that only the US catalogue gives is therefore shown only when the Indian catalogue names the same make. Otherwise the vehicle is counted and tracked without a name, and the console says why.

## How to reproduce

```bash
python3 scripts/build_autorickshaw_set.py --out data/autorickshaw
python3 scripts/train_vehicle_type.py --report experiments/2026-11-09_15-19-19
```

The script writes `models/vehicle_type.json` and `metrics.json` in this folder. `autos_used.txt` lists the photos that passed detection.
