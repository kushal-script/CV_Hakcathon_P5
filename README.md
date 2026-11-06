# P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System

Computer Vision Hackathon 2026, Problem Statement P5. The system detects a vehicle in an uploaded image, identifies its make, model and year among the 196 Stanford Cars classes, and gives a calibrated confidence with every answer. Classical computer vision (quality analysis, quality-gated enhancement, GrabCut ROI refinement, Canny edges and contours, colour/texture/geometry features) works alongside deep learning (a fine-tuned YOLO26s detector and a fine-tuned TResNet-L classifier). A decision policy answers at the model, family or make level, flags cars outside the catalogue, and reports "no vehicle" when there is none.

## Results (Stanford Cars test, n = 8,041)

| Metric | Value |
|---|---|
| **Top-1 / Top-5, deployed system** | **94.69% / 98.94%** (make level 98.37%) |
| Classical only / classical on detector crop | 2.96% / 14.74% |
| Deep only, full image (no detection) | 91.88% |
| Detection mean IoU / IoU ≥ 0.5 | 0.949 / 99.03% |
| mAP@0.5 / mAP@0.5:0.95 | 0.976 / 0.926 |
| No-vehicle images correctly rejected | 99 / 100 |
| Calibration error (ECE) | 10.4% → 1.1% after temperature scaling |
| Classifier size, INT8 | 52.9 MB (3.9× smaller, −0.37 points on val) |
| Full pipeline latency, 2 CPU threads (Apple M3, INT8) | p50 0.59 s, p95 1.10 s |
| Robustness, Gaussian noise s1 / speckle s3 / contrast s5 | 81.9 → 90.9 / 78.9 → 88.3 / 83.9 → 91.2 with the classical enhancement |

Full tables, ablations and failure cases: [eval_P5_TeamID.md](eval_P5_TeamID.md). Per-experiment records: [experiments/INDEX.md](experiments/INDEX.md).

## Documentation

| Document | Contents |
|---|---|
| [eval_P5_TeamID.md](eval_P5_TeamID.md) | Comparative evaluation in the required metric order, ablation and failure-case analysis |
| [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) | What was built and every deviation from the plan |
| [docs/FINAL_APPROACH.md](docs/FINAL_APPROACH.md) | Execution plan: requirement coverage, components, experiments, timeline, risks, viva prep |
| [docs/architecture.md](docs/architecture.md) | Architecture diagrams, module responsibilities and interfaces |
| [docs/LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md) | Literature survey (178 references) |
| [docs/research/](docs/research/), [docs/pipeline/](docs/pipeline/) | Research reports and the synthesis, critique and revision of the approach |
| [experiments/](experiments/) | One timestamped folder per experiment with config, logs, metrics and write-up |

## Repository layout

```
src/p5/
  classical/       quality analysis, enhancement, saliency ROI, GrabCut refinement, geometry, features, SVM
  detection/       ONNX detector, box selection, YOLO training
  classification/  crops, dataset, model factory (timm and I-JEPA), training, export, ONNX classifier, CAM
  confidence/      temperature scaling, selective thresholds, decision policy
  hybrid/          late fusion
  eval/            classification, detection, robustness, statistics
  pipeline.py      end-to-end inference used by the API
app/server.py      FastAPI backend: /api/predict, /api/health, /api/meta
scripts/           data preparation, training launchers, evaluation, packaging
experiments/       experiment records
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-train.txt fastapi "uvicorn[standard]" python-multipart pillow-heif
python3 scripts/build_assets.py        # validates the dataset, writes taxonomy and the val split
python3 scripts/build_cache.py         # GT crops and downscaled full images
```

Run the API locally once `models/` is populated (see Reproduction):

```bash
PYTHONPATH=src .venv/bin/uvicorn app.server:app --port 8000
curl -F "file=@car.jpg" http://localhost:8000/api/predict
```

Docker:

```bash
docker build -t p5 . && docker run -p 8000:8000 p5
```

## Reproduction

GPU jobs are plain scripts launched on a JarvisLabs A100 (`scripts/remote/*.sh`); everything they produce is synced to `runs/remote/` (git-ignored).

| Step | Command |
|---|---|
| Classifier (E05) | `bash scripts/remote/train_classifier.sh e05_tresnet_l_368 --backbone tresnet_l.miil_in1k_448 --size 368 --epochs 40` |
| Detector (E03, E06) | `bash scripts/remote/train_detector.sh --epochs 30 --batch 32` |
| Ablations (E09, E10) | `bash scripts/remote/ablations.sh` |
| Logits, INT8, robustness | `bash scripts/remote/predict_logits.sh`, `bash scripts/remote/quantize.sh`, `bash scripts/remote/robustness.sh` |
| I-JEPA (E17) | `bash scripts/remote/jepa.sh` |
| Classical baseline (E08) | `python3 scripts/train_classical.py` |
| Calibration (E13) | `python3 scripts/calibrate.py --logits <INT8 val logits> --out runs/e13_calibration` |
| Package models | `python3 scripts/package_models.py --classifier <export_int8> --detector <detector.onnx> --classical <classical_global.npz> --thresholds <thresholds.yaml>` |
| Test evaluation (E15) | `python3 scripts/eval_test.py ...` (the only script that reads test labels; calls are logged) |
| Evaluation file | `python3 scripts/make_tables.py --team <TeamID> --usns <USN1,...>` |

## Dataset

Stanford Cars (Krause et al., 2013): 16,185 images, 196 classes, 8,144 train and 8,041 test, one bounding box per image. The official host is offline; we use the torchvision-layout Kaggle mirror [`rickyyyyyyy/torchvision-stanford-cars`](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars), which includes test labels. Extract to `data/stanford_cars/`. The `data/` folder is git-ignored.

## Limitations

- Stanford Cars covers mostly US-market cars up to 2012; newer models and many regional makes are outside the catalogue and can only receive a make-level or "not a known model" answer.
- Low light at high severity, strong blur and car regions under ~60 px remain weak points (see the robustness table).
- The out-of-catalogue threshold was set on in-distribution validation data only.
- The system identifies vehicle models only; it does not read number plates or identify people.
