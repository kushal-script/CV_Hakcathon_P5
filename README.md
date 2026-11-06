# P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System

Computer Vision Hackathon 2026, Problem Statement P5. The system detects a vehicle in an uploaded image, classifies its make, model and year across the 196 Stanford Cars classes, and shows a confidence score with every answer. It combines classical computer vision (quality analysis, conditional enhancement, GrabCut ROI refinement, edges and contours, colour/texture/geometry features) with deep learning (YOLO26 detection, a fine-tuned TResNet classifier). A calibrated decision policy backs off to family or make level, or abstains, and rejects cars outside the catalogue.

> Status: planning complete, implementation starting. Results, the live URL and reproduction commands will be added here as they land.

## Documentation

| Document | Contents |
|---|---|
| [docs/LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md) | Consolidated literature survey (178 references) |
| [docs/FINAL_APPROACH.md](docs/FINAL_APPROACH.md) | Authoritative execution plan: requirement coverage, components, experiments, evaluation, timeline, risks, demo and viva prep |
| [docs/architecture.md](docs/architecture.md) | Architecture diagrams, module responsibilities and interface contracts |
| [docs/research/](docs/research/) | Five verified research reports the survey is built from |
| [docs/pipeline/](docs/pipeline/) | How the approach was derived: synthesis, critique and revision |
| [experiments/](experiments/) | One timestamped folder per experiment, plus the index |

## Dataset

Stanford Cars (Krause et al., 2013): 16,185 images, 196 classes, 8,144 train and 8,041 test, one bounding box per image. The official host is offline, so we use the torchvision-layout Kaggle mirror [`rickyyyyyyy/torchvision-stanford-cars`](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars), which includes the test labels. Extract it so the layout is `data/stanford_cars/{cars_train,cars_test,devkit,cars_test_annos_withlabels.mat}`. The `data/` folder is git-ignored.

## Experiments

```bash
python3 scripts/new_experiment.py --title "short title" --type deep --author "name"
```

See [experiments/README.md](experiments/README.md) for the conventions.
