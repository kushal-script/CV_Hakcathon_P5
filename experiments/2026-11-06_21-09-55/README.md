# E18: Indian-market catalogue from free-licensed photos

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_21-09-55` |
| Type | deep |
| Author | Kushal |
| Started | 2026-11-06 18:00 IST (dataset build), final training 22:30 IST |
| Git commit | `152eea5` at the test evaluation (`experiments/test_log.csv`) |
| Status | done |
| Compute | JarvisLabs 2x A100 40GB, one arm per GPU, 28 minutes of training per arm |

## Hypothesis
Stanford Cars is almost entirely US-market. A second catalogue trained on photos of cars sold or made in India should recognise Indian traffic far better than the Stanford model, and starting from the Stanford-trained weights should beat starting from ImageNet, because the network has already learned which details separate car models.

## Data
No public dataset covers the Indian market at model level, so one was built (`scripts/build_india_dataset.py`).

* **Candidates:** 359 car models from Wikidata (manufactured in India, Indian origin, or an Indian manufacturer) and from Wikipedia articles on the Indian automotive industry and each manufacturer's Indian arm, kept only if Wikidata types them as car models.
* **Photos:** free-licensed Wikimedia Commons photos found by a category search, at most 150 per Commons category. Commons rate-limits downloads per network address, so the collection took most of an evening and some thumbnails were fetched at 500 px. Every photo was normalised to 500 px before cropping so that resolution cannot identify a class.
* **Cleaning:** 33,647 photos checked; kept 27,822. Rejected: no vehicle 1,849, vehicle too small 3,276, several large vehicles 568, listed under two models 1,902, near duplicates 132.
* **Classes:** 282 models with at least 15 photos (35 of them made in India according to Wikidata), 15 to 272 photos per model (median 113, the largest are models merged from several categories). 62 candidates were dropped (59 with too few free photos, 3 without enough independent photographers) and 7 sub-categories were merged into their parent model.
* **Split:** 70/15/15 with every photographer in one split only: 18,585 train, 4,499 validation, 4,298 test.

Two dataset rules were corrected before any test photo was scored. A rule that dropped classes the COCO detector mostly labels "truck" removed boxy cars such as the Mahindra Scorpio and Thar, so it was turned off (the Wikidata type already limits candidates to cars). The photographer split let one large photographer group overflow the validation share and dropped the Toyota Innova, so validation and test are now capped at 30% of a class. Earlier runs on the incomplete and the uncorrected datasets are kept under `runs/remote/` and were never scored on test.

## How to reproduce
```bash
python3 scripts/build_india_dataset.py --stage classes
python3 scripts/build_india_dataset.py --stage list
python3 scripts/build_india_dataset.py --stage collect --offline
python3 scripts/build_india_dataset.py --stage clean
python3 scripts/build_india_dataset.py --stage split --min-images 15 --min-holdout 2
PREFIX=e18v3 bash scripts/remote/india_train_arms.sh
python3 scripts/calibrate.py --logits <arm>/export_int8/val_logits_int8.npz --assets assets/india --out runs/e18_calibration/<arm>
python3 scripts/eval_india.py --runs <both arms> --thresholds <both thresholds.yaml> --us-models models --out runs/e18_test --tag "E18 final test evaluation"
```

## Results
Split used for these numbers: **test (photographer-disjoint, n = 4,298, 282 models)**, read once.

| Model (INT8, single view) | Top-1 | 95% CI | Top-5 | Macro Top-1 | Make Top-1 | ECE |
|---|---|---|---|---|---|---|
| Started from the Stanford Cars model (deployed) | **84.32** | 83.18–85.32 | 93.65 | 82.73 | 89.95 | 1.42 |
| Started from ImageNet | 83.18 | 82.11–84.34 | 92.35 | 82.00 | 89.11 | 1.28 |

* Paired comparison (McNemar): the Stanford start is right on 170 photos where the ImageNet start is wrong, and wrong on 125 where it is right, p = 0.01.
* FP32 against INT8 on test: 84.78 against 84.32. Flip test-time augmentation (FP32): 84.76.
* Validation (INT8): 84.37 and 82.66. The ImageNet arm's quantization ran out of host memory while both arms calibrated at once and was rerun alone with the same settings. Temperature 0.670 lowers validation ECE from 20.06 to 2.13.
* Decision levels on test with thresholds from validation: model named for 72.4% of photos and correct 97.5% of those times; make only 19.3%; not sure 3.8%; outside the catalogue 4.5%.
* Accuracy at 80% and 90% coverage (most confident answers): 95.38 and 90.93.

**The US catalogue on Indian cars.** Run on the same test photos, the deployed Stanford Cars model names a specific model for 8.2% of them and flags 91.3% as outside its catalogue. On the 3,043 photos whose make also exists in Stanford Cars it names the make correctly 29.0% of the time, against 90.1% for the Indian model.

Per-class test accuracy is in `artifacts/per_class_e18v3_india_e05init.csv`.

## Decision
Deploy the model started from Stanford Cars as the Indian-market catalogue (`models/india/`), with its own temperature, thresholds and out-of-catalogue gate. The app runs both catalogues on every vehicle and shows the more specific and more confident answer, with the other catalogue's answer beside it.

## Limitations
* Commons coverage of current Indian models is thin. 59 candidates, including recent Tata and Mahindra models, have fewer than 15 usable free photos and are not in the catalogue. 44 kept models have fewer than 40 photos, so their test accuracy rests on a handful of images.
* Most photos are enthusiast shots in daylight, often at motor shows, not CCTV frames. Expect lower accuracy on night-time traffic camera crops.
* Some Commons categories mix generations or body styles of a model, and 7 sub-categories were merged into parent models by photo overlap.
* Classes are model names without years, unlike Stanford Cars.
