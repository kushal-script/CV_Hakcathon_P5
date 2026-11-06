# Experiments

Every experiment we run (training, evaluation, ablation, robustness test, optimisation, classical baseline, data audit) gets its **own folder named by its start timestamp**. Nothing is overwritten: a re-run is a new folder.

```
experiments/
├── README.md              ← this file (conventions)
├── INDEX.md               ← one row per experiment, newest at the bottom
├── _template/             ← copied into every new experiment
└── 2026-11-06_14-32-07/   ← one experiment (local time, YYYY-MM-DD_HH-MM-SS)
    ├── README.md          ← hypothesis, setup, results, observations, conclusion
    ├── config.yaml        ← exact config used (model, data, hyper-params, seed)
    ├── metrics.json       ← machine-readable results (feeds eval_P5_<TeamID>.md)
    ├── logs/              ← training / eval logs (text, kept small)
    ├── figures/           ← plots: curves, confusion matrix, reliability diagram, Grad-CAM
    └── artifacts/         ← checkpoints, ONNX files, predictions (git-ignored — too large)
```

## Creating an experiment

```bash
python3 scripts/new_experiment.py --title "ConvNeXt-S 384 on GT crops" --type deep --author "Kushal"
```

This creates the timestamped folder from `_template/`, fills in the header (title, type, author, git commit, start time), and appends a row to `INDEX.md`.

`--type` is one of: `data`, `classical`, `deep`, `detection`, `hybrid`, `ablation`, `robustness`, `optimisation`, `calibration`, `failure-analysis`.

## Rules

1. **One question per experiment.** State the hypothesis before running.
2. **Tune on val, never on test.** Test-set numbers are reported only for configs frozen on the validation split. Record which split each number comes from.
3. **Reproducible:** `config.yaml` + the git commit in the header + the exact command in the README must be enough to re-run.
4. **Fill `metrics.json`** with the same keys every time, so results can be aggregated into the comparison and ablation tables automatically.
5. **Update `INDEX.md`** when an experiment finishes: set its status (`running` → `done` / `failed` / `abandoned`) and its headline Top-1.
6. **Keep large files out of git.** Checkpoints and ONNX files go in `artifacts/` (ignored) and get uploaded to the HF Hub or GitHub Releases. Link them from the README.
7. **Failed experiments stay.** Write down why they failed; they feed the failure-case analysis.
