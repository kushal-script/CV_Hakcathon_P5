# E17: I-JEPA ViT-H/14 linear probe and fine-tune

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_15-40-59` |
| Type | deep |
| Author | Kushal |
| Started | 2026-11-06 15:24 IST (folder created 15:40) |
| Status | done |
| Compute | JarvisLabs A100 40GB, run `r_70bef59c`, about 40 min (probe 8 min, fine-tune 32 min, logits 4 min) |

## Hypothesis
I-JEPA (Assran et al., CVPR 2023), a joint-embedding predictive architecture pretrained without labels on ImageNet-1k only, is weak as a frozen linear probe (the paper's own caveat) but competitive after end-to-end fine-tuning. Its errors may differ from the supervised CNN's enough for an ensemble to help.

## Setup
- **Backbone:** `facebook/ijepa_vith14_1k` (ViT-H/14, 630M parameters, ImageNet-1k, self-supervised) via Hugging Face Transformers. Mean-pooled final tokens, LayerNorm, linear head. Same data and contamination profile as the TResNet (ImageNet-1k only).
- **Linear probe:** frozen features at 224 px on GT crops (train plus horizontal flips), linear head trained with AdamW for 300 epochs, weight decay chosen on val from {1e-4, 1e-2, 0.1, 0.5}.
- **Fine-tune:** 280 px (positional embeddings interpolated), 15 epochs, AdamW, LR 1e-4 at the top block with layer-wise decay 0.85 down the 32 blocks, head LR 1e-3, weight decay 0.05, label smoothing 0.1, gradient checkpointing, bf16, same crops and augmentation as E05.
- **Ensemble:** weighted average of temperature-scaled log-probabilities with TResNet, weight chosen on val.

## How to reproduce
```bash
bash scripts/remote/jepa.sh
```

## Results

| Model | Val Top-1 | Test Top-1 (GT crop) | Test Top-1 (detector crop) | Test make Top-1 |
|---|---|---|---|---|
| I-JEPA linear probe (frozen, 224 px) | 65.40 | 66.24 | — | 76.88 |
| **I-JEPA fine-tuned (280 px)** | 92.64 (93.01 flip) | **93.14** | **92.53** | 97.15 |
| TResNet-L (E05, 368 px) for reference | 95.34 | 95.21 | 94.53 | 98.22 |

Fine-tuning curve (val): 85.8 (epoch 3), 90.3 (6), 92.8 (9), 93.1 (12), 92.6 (15).

Ensemble on val (choosing the weight): GT crops +0.12 points at weight 0.3 (5 images fixed, 4 broken, McNemar p = 1.0); detector crops: best weight 0.0 (no gain). Agreement between the two models: 95%. Both wrong on the same image: 3.8–4.2% of val.

## Observations
- The linear probe confirms the paper's caveat: frozen I-JEPA features are far from linearly separable for fine-grained car models (66%).
- End-to-end fine-tuning recovers almost all of the gap (+27 points): a label-free ImageNet-1k encoder reaches 93.1%, within 2.1 points of the supervised TResNet, at lower resolution (280 vs 368 px).
- The ensemble does not help because the two very different models fail on the same images. That points to the residual errors being inherently ambiguous (trim and year twins, rebadges, likely label noise; see E15) rather than a model-capacity limit.
- Not deployable on a CPU host: a ViT-H forward pass takes seconds on 2 vCPUs. TResNet-L remains the deployed classifier.

## Artifacts
`runs/remote/work-runs/e17_ijepa_vith14/` (probe logits and metrics, fine-tuned `model.pt` 2.5 GB, `last.pt` with optimiser state, val and test logits). Logs and ensemble analysis in `logs/`.
