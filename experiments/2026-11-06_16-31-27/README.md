# E15: final test evaluation of the deployed system (single run)

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_16-31-27` |
| Type | failure-analysis |
| Author | Kushal |
| Started | 2026-11-06 16:30 IST |
| Status | done |
| Compute | Deployed pipeline run on the instance CPU (run `r_24c3b8bd`, 40 processes, labels not read); scoring locally with `scripts/eval_test.py` |

## Protocol
- Everything (models, thresholds, enhancement rule, fusion off) was frozen on train/val before any test label was read.
- Test labels are read only by `scripts/eval_test.py`; every call is logged in `experiments/test_log.csv`. Three calls were made, all on the same frozen system: (1) the main evaluation; (2) adding the detector-selected column to the E14 table (a metric for a rule fixed in E12, no tuning); (3) failure-case extraction. The headline numbers are identical across calls.
- The **deployed row** is the real product pipeline run on every test image: input guard, quality analysis, enhancement with detector selection, YOLO26s, INT8 TResNet-L on the detected crop, calibrated decision policy.

## Headline results (Stanford Cars test, n = 8,041)

| Row | Input | Top-1 | Top-5 | Make Top-1 | Family Top-1 |
|---|---|---|---|---|---|
| C: classical only | saliency ROI | 2.96 | 8.73 | 7.66 | 3.01 |
| C+Y: classical on detector crop | YOLO crop | 14.74 | 33.27 | 23.38 | 15.04 |
| Classical on GT crop (diagnostic) | GT crop | 14.74 | 33.06 | 23.19 | 15.01 |
| D: deep, full image | full image | 91.88 | 97.66 | 95.88 | 92.34 |
| D: deep on detector crop (FP32) | YOLO crop | 94.53 | 98.67 | 98.22 | 94.96 |
| D: deep on GT crop (diagnostic) | GT crop | 95.21 | 99.22 | 98.77 | 95.63 |
| D: deep on detector crop (INT8) | YOLO crop | 94.49 | 98.68 | 98.22 | 94.93 |
| **H: deployed hybrid system (INT8)** | **raw upload** | **94.69** | **98.94** | **98.37** | **95.15** |
| I-JEPA ViT-H fine-tuned (E17) | YOLO crop | 92.53 | 98.53 | 97.15 | 93.11 |
| I-JEPA ViT-H linear probe (E17) | GT crop | 66.24 | 87.49 | 76.88 | 66.92 |

Year-twin confusion (deployed): 6.4%. Rebadge-twin confusion: 10.0%.

**Detection (test, single GT box per image):**

| Detector | Mean IoU | IoU ≥ 0.5 | IoU ≥ 0.75 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| YOLO26s zero-shot | 0.9124 | 95.51 | 94.71 | 0.9321 | 0.8696 |
| **YOLO26s fine-tuned** | **0.9493** | **99.03** | **98.45** | **0.9758** | **0.9261** |

No-vehicle images (100 held-out COCO test negatives): **99/100 correctly report "no vehicle"**.

**Confidence (deployed decision policy):**
- ECE 10.36% → **1.13%** after temperature scaling (T = 0.769 from val); AURC 0.0200.
- Accuracy at 80 / 90 / 95% coverage: 98.21 / 97.68 / 96.79.
- Decision levels: class 93.5% of images (**97.2% correct**), family 1.3% (71.4%), make 0.4% (93.1%), "possibly ⟨make⟩, not a known model" 4.5% (81.1% correct make), no vehicle 0.3%. Answered-and-correct rate across all levels: **95.81%**.

## Failure analysis (430 wrong Top-1 of 8,041 = 5.35%)

| Category | Count | Most frequent pairs |
|---|---|---|
| Same make, different model or trim | 256 | Express Van ↔ Express Cargo Van (13+), Audi TTS ↔ TT Hatchback (21), S5 ↔ A5 (11), V8 ↔ 100 (11), 6 Series ↔ M6 Convertible (8, same E64 body), Spyker C8 Coupe ↔ Convertible (6) |
| Different make | 70 | HUMMER H3T → GMC Canyon (platform twins) |
| Rebadged twins across makes | 39 | Dodge Sprinter → Mercedes-Benz Sprinter (8) |
| Same model, different year | 38 | Dodge Caliber 2007 ↔ 2012 (21, same generation), Bentley Continental GT 2007 ↔ 2012 (13) |
| No vehicle found or input rejected | 27 | 21 detector misses (show-floor scenes, unusual angles), 6 images under 64 px rejected by the input guard |

Worked cases (figures in `figures/`):
1. **Trim twin, confidently wrong** (`rebadge_twin_05725.jpg`): a windowless Express *Cargo* Van predicted as Express Van at 100%. The CAM sits on the grille and headlights, which are identical between the trims. The real cue, the windowless side panels, gets no attention. Cause: front-fascia shortcut; the two classes differ only in body panels.
2. **Same-generation year twin** (`year_twin_04160.jpg`): Dodge Caliber 2007 predicted as 2012. Both years are the same generation with a mild refresh; the CAM is on the unchanged grille. Cause: near-indistinguishable labels (Krause et al. defined years as representative appearances). The decision policy's family back-off ("Dodge Caliber Wagon (2007 or 2012)") is the right output here.
3. **Probable label error** (`confident_cross_make_06227.jpg`): labelled HUMMER H3T Crew Cab, predicted GMC Canyon at 96%. The truck carries a Chevrolet "Z71" badge: it is a Colorado, which shares the GMT355 platform with the Canyon and H3T. The prediction is closer to the truth than the label.
4. **Missed detection in a cluttered scene** (`no_detection_00760.jpg`): Lamborghini Aventador on a show stand behind a crowd; three small cars, none detected. Cause: small, partly occluded vehicles in a scene type absent from training.
5. **Input guard too strict**: 6 test images are smaller than 64 px (e.g. 78×58) and were rejected. The guard was lowered to 32 px after this evaluation. The numbers above keep the original guard and count these as failures.

## Observations
- The full product pipeline (94.69%) slightly beats the same INT8 model on cached detector crops (94.49%): the quality-gated enhancement with detector selection helps a little even on the "clean" test set (it fired on 24% of images).
- Detection crops cost 0.68 points against GT crops (94.53 vs 95.21), consistent with Krause et al. (0.6 points).
- Most errors stay inside the right make (98.4% make accuracy). Year twins, trim twins and rebadges dominate, which the decision policy turns into honest family- or make-level answers instead of a confident wrong model.

## Artifacts
`logs/test_report.json` (all numbers), `logs/failures.json`, `logs/test_log.csv`, `logs/pipeline_run_output.log`, `figures/*.jpg`. Per-image deployed outputs: `runs/remote/work-runs/e15_pipeline/test_pipeline.jsonl` (git-ignored).
