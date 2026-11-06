# E03+E06: YOLO26s zero-shot and single-class fine-tune

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_13-51-37` |
| Type | detection |
| Author | Kushal |
| Started | 2026-11-06 13:51:37 IST |
| Git commit | `e7858a9-dirty` |
| Status | done |
| Compute | JarvisLabs A100 40GB, run `r_b3a1db81`; 30 epochs, about 32 min of training while sharing the GPU with E05 |

## Hypothesis
- **E03:** COCO-pretrained YOLO26s, keeping only the car, bus and truck classes, already localises the Stanford Cars box well (mean IoU ≥ 0.85).
- **E06:** fine-tuning one "vehicle" class on Stanford Cars boxes improves IoU and mAP. Pseudo-labelling unlabelled background cars and adding COCO no-vehicle images keeps it from learning to ignore secondary cars or to fire on empty scenes.

## Setup
- **Images:** full images from the cache (longest side ≤ 512), letterboxed to 640.
- **Zero-shot (E03):** `yolo26s.pt`, classes {car, bus, truck}, conf 0.001, max 100 detections.
- **Pseudo-labels:** zero-shot boxes with score ≥ 0.5, IoU < 0.3 with the GT box and area ≥ 1% of the image are added as extra "vehicle" labels on training images only. Result: **3,373 extra boxes on 1,817 images**. Validation keeps GT boxes only (strict).
- **Negatives:** 500 COCO val2017 images with no car, bus, truck or motorcycle (`assets/splits/coco_neg_train.txt`). 100 dev and 100 test negatives are held out.
- **Fine-tune (E06):** YOLO26s from COCO weights, 1 class, 30 epochs, batch 32, image size 640, Ultralytics defaults otherwise, seed 42. The best checkpoint by Ultralytics fitness is kept.

## How to reproduce
```bash
bash scripts/remote/train_detector.sh --epochs 30 --batch 32
```

## Results
Split used for these numbers: **val (n = 815)**, top-scoring box per image against the single GT box; mAP over all boxes with COCO-style 101-point interpolation.

| Metric | E03 zero-shot | E06 fine-tuned |
|---|---|---|
| Mean IoU | 0.9137 | **0.9492** |
| IoU ≥ 0.5 (CorLoc) | 95.83 | **99.14** |
| IoU ≥ 0.75 | 94.85 | **98.53** |
| mAP@0.5 | 0.9352 | **0.9809** |
| mAP@0.75 | 0.9213 | **0.9741** |
| mAP@0.5:0.95 | 0.8679 | **0.9276** |
| Images with no box | 0 | 0 |

Threshold analysis (val recall at IoU ≥ 0.5 and false-alarm rate on 100 held-out COCO dev no-vehicle images):

| Score threshold | 0.05 | 0.10 | 0.25 | 0.50 |
|---|---|---|---|---|
| Fine-tuned val recall | 99.14 | 99.14 | 99.14 | 99.14 |
| Fine-tuned no-car false alarms | 2% | **0%** | **0%** | **0%** |
| Zero-shot no-car false alarms | 5% | 3% | 1% | 1% |

The lowest top-box score among val images is 0.657 (0.1% quantile), so τ_lo = 0.10 (weak) and τ_det = 0.25 (strong) are well inside the safe range.

## Observations
- Most zero-shot misses (IoU < 0.5) come from the top-scoring box landing on a different car or on part of a truck. Fine-tuning to the dataset's box style removes most of them.
- The negatives remove empty-scene false alarms entirely at the deployed thresholds.
- Ultralytics' own validation (mAP50 0.990, mAP50-95 0.919 at the last epoch, see `logs/results.csv`) differs slightly from ours because it uses a different interpolation and matching rule. The numbers above come from `p5.eval.detection` so zero-shot and fine-tuned are scored the same way.

## Conclusion & next step
- Deploy the fine-tuned detector (`detector.onnx`). Test-set IoU and mAP will be reported once, in E15, together with the 100 COCO test negatives.

## Artifacts
`runs/remote/work-runs/e06_detector/` (git-ignored): `detector.pt`, `detector.onnx`, prediction caches for train (zero-shot), val, test and COCO negatives, and the Ultralytics run folder `e06/` with curves and batch previews. Training curves and a prediction preview are copied to `figures/`.
