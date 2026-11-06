# Implementation notes: what was built, and where it differs from the plan

`docs/FINAL_APPROACH.md` is the plan written before implementation. This file records what was actually built on 2026-11-06 and every deviation from that plan, with the reason. Experiment records are in `experiments/` (see `experiments/INDEX.md`).

## Status

| Component | Status | Evidence |
|---|---|---|
| Data protocol (E01): taxonomy, 90/10 split, crop cache | done | `scripts/build_assets.py`, `assets/` |
| Detector: YOLO26s zero-shot (E03) and fine-tuned (E06) | done | `experiments/2026-11-06_13-51-37` |
| Classifier: TResNet-L@368 (E05) | done | `experiments/2026-11-06_13-51-36` |
| Classical branch: quality, enhancement, GrabCut, Canny, contours, geometry, features + SVM (E08) | done | `experiments/2026-11-06_13-51-38` |
| Ablations (E09, E10) | done | `experiments/2026-11-06_15-40-53` |
| ONNX export and INT8 quantization (E11) | done | `experiments/2026-11-06_15-40-54` |
| Fusion and enhancement tuning (E12) | done | `experiments/2026-11-06_15-40-56` |
| Calibration and thresholds (E13) | done | `experiments/2026-11-06_15-40-57` |
| Robustness suite (E14) | done | `experiments/2026-11-06_15-40-58` |
| Final test evaluation and failure analysis (E15) | done | `experiments/2026-11-06_16-31-27`, `eval_P5_TeamID.md` |
| I-JEPA research arm (E17, not in the original plan) | done | `experiments/2026-11-06_15-40-59` |
| Backend API (FastAPI) and Dockerfile | built, tested locally | `app/server.py`, `Dockerfile` |
| Web frontend and public hosting | not started | hosting choice still open |
| Web-image out-of-catalogue set (planned E13 dev/test split) | not collected | τ_ood tuned on in-distribution val only |

## Deviations from the plan

1. **Compute:** training ran on a JarvisLabs A100 40GB instead of free Kaggle T4s. Jobs are plain scripts in `scripts/remote/` launched with the `jl` CLI; every log and artifact was synced back into `runs/remote/` (git-ignored). Total GPU instance cost was about ₹293 (A100 on demand, about 3.5 hours including downloads).
2. **Serving stack:** a FastAPI JSON API (`app/server.py`: `/api/predict`, `/api/health`, `/api/meta`) replaces the planned Gradio app, so any web frontend can call it. The frontend and host are still to be decided.
3. **Enhancement policy:** the plan applied the classical enhancement whenever quality flags fired. E12 showed that always enhancing can hurt (brightening exposes sensor noise that defeats the detector). The pipeline now runs the detector on both the original and the enhanced image and keeps the more confident one. On the test robustness subset this is never worse than raw by more than 0.7 points and gains up to +9.4.
4. **Fusion:** classical score fusion was implemented and tested (E12) but rejected by its pre-registered rule (+0.12 points, p = 1.0). It ships with β = 0, so the classical SVM drives the comparison view and panels, not the decision.
5. **Ablation backbone:** E09/E10 used TResNet-M for 15 epochs (as planned) on a single A100 rather than split across Kaggle accounts.
6. **Quantization location:** ONNX Runtime's percentile calibration ran out of memory on the 16 GB laptop, so E11 ran on the instance CPU. INT8 is 3.3× faster on Apple M3 but slower on the instance's AMD EPYC 7542 (no VNNI). The deployed precision should be chosen by measuring on the actual host.
7. **Input guard:** the minimum image side was 64 px. Six Stanford Cars test images are smaller and were rejected during E15. The guard is now 32 px; E15 numbers keep the original guard.
8. **Added experiment E17:** I-JEPA ViT-H/14 (self-supervised on ImageNet-1k) as a research comparison: 66.2% frozen linear probe, 93.1% fine-tuned on test GT crops; no ensemble gain. Not deployable on CPU.
9. **Plan items not done:** the web-image out-of-catalogue set (so τ_ood is set at 95% in-distribution TPR without OOD tuning), the SSB open-set benchmark, and the `tests/test_architecture.py` diagram check.

## Test-label hygiene
`scripts/eval_test.py` is the only code that reads test labels. `experiments/test_log.csv` lists its three calls, all on the same frozen system: the main evaluation, a call that added the detector-selected column to the robustness table, and failure-case extraction.
