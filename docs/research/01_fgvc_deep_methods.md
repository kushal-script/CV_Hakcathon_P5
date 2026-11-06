# Deep FGVC Methods on Stanford Cars (2015–2026): Literature Survey

*Compiled 2026-11-06. Dataset: Stanford Cars, 196 classes, 8,144 train / 8,041 test.*

**How the numbers were checked**
- **[P]**: I read the number in the primary paper's own table or text, from the arXiv or CVF PDF.
- **[S]**: I read the number only in another paper's comparison table. The primary paper was not checked.
- **(unverified)**: I could not confirm the number from any paper table.
- "Res" is the training/test input size. "—" means the paper does not report Stanford Cars.

---

## 1. Part / attention localisation

| Method | Authors, venue | Key idea | Backbone @ res | Cars Top-1 |
|---|---|---|---|---|
| B-CNN | Lin et al., ICCV 2015 | Outer-product (bilinear) pooling of two CNN streams | VGG-M+VGG-D @448 | **91.3** [P] (VGG-D alone 90.6) |
| RA-CNN | Fu et al., CVPR 2017 | Recurrent attention-proposal zoom with an inter-scale ranking loss | VGG-19 @448 | **92.5** [P] |
| MA-CNN | Zheng et al., ICCV 2017 | Channel grouping into several part attentions | VGG-19 @448 | 92.8 [S: DCL, NTS-Net tables] |
| DFL-CNN | Wang et al., CVPR 2018 | Learnable 1×1 filter bank for discriminative patches | VGG-16 / ResNet-50 | 93.8 / 93.1 [S: MC-Loss, PMG tables] |
| NTS-Net | Yang et al., ECCV 2018 | Navigator–Teacher–Scrutinizer self-supervised region proposals | ResNet-50 @448 | **93.9** [P] |
| WS-DAN | Hu et al., arXiv 2019 | Attention-guided cropping and dropping as data augmentation | Inception-v3 @448 (res unverified) | **94.5** [P] |
| S3N | Ding et al., ICCV 2019 | Sparse resampling around class peak responses | ResNet-50 @448 | **94.7** [P] |
| MMAL-Net | Zhang et al., MMM 2021 | Multi-branch attention: raw image, object crop, part crops | ResNet-50 @448 | **95.0** [P] |
| CAL | Rao et al., ICCV 2021 | Counterfactual (causal) attention learning on top of WS-DAN | ResNet-101 @448 | **95.5** [P] |
| CAP | Behera et al., AAAI 2021 | Context-aware attentional pooling over regions, with an LSTM | Xception @224 | **95.7** [P] |
| SR-GNN | Bera et al., IEEE TIP 2022 | Relation-aware region graph neural network | Xception @224 | **96.1** [P] |
| CMAL-Net | Liu et al., Pattern Recognition 2023 | Shallow-to-deep CNN layers act as "experts" that exchange attention regions | TResNet-L (@368 per PMD) | 97.1 [S: PMD paper, Table V] |

## 2. Feature encoding / high-order pooling

| Method | Authors, venue | Key idea | Backbone @ res | Cars Top-1 |
|---|---|---|---|---|
| Compact Bilinear (CBP) | Gao et al., CVPR 2016 | Tensor-Sketch / Random-Maclaurin approximation of bilinear features | VGG-16 @448 | — in the original paper. 91.2 for VGG-D in the iSQRT-COV table [S] |
| Kernel Pooling | Cui et al., CVPR 2017 | Higher-order polynomial kernel approximation | VGG-16 | 92.4 [S: DCL, iSQRT tables] |
| iSQRT-COV | Li et al., CVPR 2018 | Covariance pooling with Newton–Schulz iterative matrix square root | ResNet-50 / ResNet-101 @448 | **92.8 / 93.3** [P] |
| HBP | Yu et al., ECCV 2018 | Hierarchical cross-layer bilinear pooling | VGG-16 | 93.7 [S: MMAL, CIN tables] |
| DBTNet | Zheng et al., NeurIPS 2019 | Deep bilinear transformation | ResNet-101 | 94.5 [S: TransFG table] |
| MC-Loss | Chang et al., IEEE TIP 2020 | Channel-level discriminality and diversity loss | ResNet-50 / B-CNN @448 | **93.7 / 94.4** [P] |

## 3. Training-strategy / regularisation methods

| Method | Authors, venue | Key idea | Backbone @ res | Cars Top-1 |
|---|---|---|---|---|
| DCL | Chen et al., CVPR 2019 | Destruction (region-confusion shuffle) plus construction (layout recovery) | ResNet-50 @448 | **94.5** [P] |
| Cross-X | Luo et al., ICCV 2019 | Cross-category and cross-layer regularisation of OSME features | ResNet-50 / SENet-50 | **94.6 / 94.5** [P] |
| API-Net | Zhuang et al., AAAI 2020 | Attentive pairwise interaction between confusable image pairs | R50 / R101 / DenseNet-161 @448 | **94.8 / 94.9 / 95.3** [P] |
| CIN | Gao et al., AAAI 2020 | Self- and contrastive channel interaction | ResNet-50 / ResNet-101 | **94.1 / 94.5** [P] |
| PMG | Du et al., ECCV 2020 | Progressive multi-granularity training on jigsaw patches | ResNet-50 @448 (550 resize) | **95.0** (95.1 combined) [P] |
| PMAL / PMD | D. Liu et al., arXiv 2401.14336 (2024; venue unverified) | Progressive multi-task anti-noise learning, then distillation back into a plain backbone (no extra inference cost) | TResNet-L @368 / ResNet-50 @448 | **97.3 / 97.3** (TResNet-L); **95.4 / 95.3** (R50) [P] |

### Strong-backbone transfer baselines (no FGVC-specific modules)

| Method | Venue | Backbone @ res | Pretraining | Cars Top-1 |
|---|---|---|---|---|
| Kornblith et al., "Do Better ImageNet Models Transfer Better?" | CVPR 2019 | Inception-v4 @299 | IN-1k | **93.3** fine-tuned; 92.7 from scratch [P] |
| EfficientNet | Tan & Le, ICML 2019 | B3 / B7 @600 (B7) | IN-1k | **93.6 / 94.7** [P] |
| GPipe | Huang et al., NeurIPS 2019 | AmoebaNet-B | IN-1k | 94.6 [S: MetaFormer, CAP tables] |
| DeiT | Touvron et al., ICML 2021 | DeiT-B @224 / ↑384 / distilled ↑384 | IN-1k | **92.1 / 93.3 / 93.9** [P] |
| TResNet | Ridnik et al., WACV 2021 | TResNet-L @368, CE + triplet loss | IN-1k | **96.0** [P] |
| Domain-Adaptive Transfer | Ngiam et al., arXiv 2018 | Inception-v3 / AmoebaNet-B, "JFT-Car" subset | JFT (extra data) | **96.0 / 96.2** [P] |
| SAM | Foret et al., ICLR 2021 | EffNet-L2, no SAM / with SAM | JFT/NoisyStudent (extra data) | **95.07 / 95.96** (errors 4.93 / 4.04) [P] |

## 4. Vision-Transformer FGVC

| Method | Authors, venue | Key idea | Backbone @ res | Cars Top-1 |
|---|---|---|---|---|
| TransFG | He et al., AAAI 2022 | Part-selection module using attention rollout, plus contrastive loss and overlapping patches | ViT-B/16 (IN-21k) @448 | **94.8** [P]. AF-Trans's re-run gives 94.1 |
| FFVT | Wang et al., BMVC 2021 | Fuses token features selected from multiple layers | ViT-B/16 | — (checked: no Cars result) |
| RAMS-Trans | Hu et al., ACM MM 2021 | Recurrent attention multi-scale crops | ViT-B/16 | — (checked: CUB, Dogs, iNat17 only) |
| AF-Trans | Zhang et al., ICASSP 2022 | Selective attention collection and multi-scale fusion | ViT-B/16 | **95.0** [P] |
| DCAL | Zhu et al., CVPR 2022 | Global-local and pair-wise cross-attention | R50-ViT-B | 95.3 (unverified: seen only in a search snippet of the arXiv text) |
| MetaFormer | Diao et al., arXiv 2022 | Hybrid conv+attention backbone with optional meta-information | MetaFormer-1 @384 | **94.9** (IN-1k) / **95.0** (IN-21k) / **95.0** (iNat21) [P] |
| CAMF | (venue unverified) | Swin-based multi-feature fusion | Swin-B @224 | 95.3 [S: SR-GNN table] |
| SIM-Trans | Sun et al., ACM MM 2022 | Structure-information and multi-level feature boosting modules | ViT-B/16 | Aggregators list no Cars result (unverified) |
| IELT | Xu et al., IEEE TMM 2023 | Internal ensemble: layers vote with dynamic token selection | ViT-B/16 | Cars not in the reported dataset list per search results (unverified) |
| PIM / HERBS | Chou et al., arXiv 2022 / 2023 | Plug-in background suppression with high-temperature refinement | Swin-L @384 | — (checked: no Cars result in either paper) |
| ACC-ViT | Zhang et al., 2024 | — | ViT-B | Primary paper not found. Cars number unverified |
| SCOPE | Xu et al., arXiv 2508.06959 (2025) | Spatial-domain extraction of subtle cues | — | **Withdrawn by the authors** (they cite inconsistent experiment records). Do not cite |

**Observation.** Plain ViT-B/16 FGVC methods top out at about 94–95% on Cars. This is below the best CNNs at 448 resolution and below TResNet-L. TransFG's authors attribute this to the "more regular and simpler" shapes of cars, which favour high-resolution CNNs.

## 5. Foundation models / large pretraining on Stanford Cars

### Zero-shot and frozen linear probe

| Model | Source | Protocol | Cars Top-1 |
|---|---|---|---|
| CLIP ViT-B/16 | Radford et al., ICML 2021 | Zero-shot / linear probe | **65.6 / 86.7** [P] |
| CLIP ViT-L/14@336 | Radford et al., ICML 2021 | Zero-shot / linear probe | **78.8 / 91.5** [P] |
| EVA-02-CLIP-L/14+ (336) | Sun et al., EVA-CLIP, arXiv 2023 | Zero-shot | **90.5** [P] |
| OpenCLIP ViT-G/14 (LAION-2B) | EVA-CLIP Tab. 2 | Zero-shot | **94.5** [P] |
| EVA-02-CLIP-E/14+ | EVA-CLIP Tab. 2 | Zero-shot | **94.6** [P] |
| DINOv2 ViT-S / B / L / g (/14) | Oquab et al., TMLR 2024, Tab. 8 | Linear (logistic regression) | **81.6 / 88.2 / 90.1 / 91.4** [P] |
| OpenCLIP ViT-G/14 | DINOv2 Tab. 8 | Linear | **96.1** [P] |
| DINOv3 7B/16 | Siméoni et al., arXiv 2508.10104, Tab. 22 | Linear ("Fine-S") | **95.0** [P] |
| DINOv2 g/14 (re-evaluated) | DINOv3 Tab. 22 | Linear | **94.6** [P] |
| SigLIP g/16 (labelled "SigLIP" in the table) | DINOv3 Tab. 22 | Linear | **96.5** [P] |
| PE-core G/14 | DINOv3 Tab. 22 | Linear | **96.5** [P] |
| AIMv2 3B / EVA-CLIP 18B / AM-RADIOv2.5 g | DINOv3 Tab. 22 | Linear | **95.6 / 95.4 / 95.4** [P] |
| Supervised IN-21k ViT-B/16 | CLIP Tab. 10 | Linear | 54.7 [P]. Supervised features transfer poorly to Cars |

Language-supervised (CLIP-family) models beat SSL models on Cars. DINOv2's authors note they trail OpenCLIP by 4.7 points on Cars. A controlled 2025 study (arXiv 2510.11835) reports CLIP 74.7 vs DINO 54.1 on Cars under identical data (unverified, search summary only). The likely reason is that car make/model names appear in web captions.

### Parameter-efficient fine-tuning (VTAB-style "FGVC" protocol)

This protocol uses ViT-B/16 (IN-21k) at **224**, with an internal 7,329/815 train/val split. The numbers are **not comparable** to the 448-resolution FGVC literature.

| Method | Venue | Cars Top-1 |
|---|---|---|
| Full fine-tune / Linear probe | VPT baselines | 84.5 / 51.3 [S: SSF Tab. 3] |
| Adapter / BitFit | VPT baselines | 68.6 / 79.4 [S] |
| VPT-shallow / VPT-deep | Jia et al., ECCV 2022 | 68.7 / 83.6 [S: SSF Tab. 3] |
| SSF (scale and shift features) | Lian et al., NeurIPS 2022 | **89.2** [P] |
| GPS (CVPR 2024), V²APT (arXiv 2503.17650, 2025), frequency-based PEFT (arXiv 2411.19297) | 2024–25 | Cars numbers unverified. V²APT reports that full FT still beats it on Cars |

**Takeaway.** Cars is the dataset in this benchmark where PEFT methods trail full fine-tuning most. Fine detail at low resolution needs backbone adaptation.

### Other 2024–2026 work worth knowing
- **Car-1000** (arXiv 2503.12385; CVPR 2024 FGVC workshop): a newer car dataset with 1,000 models. It is motivated by the fact that Stanford Cars covers only models released before 2013.
- **Training-free and LVLM-based FGVC** (FineR, ICLR 2024; Enriched-FineR, 2025): useful for discovering open vocabulary, but well below supervised accuracy (unverified numbers).
- **stanford_cars_refined** (GitHub): re-labels Cars by colour into 1,288 classes. It is not a label-error audit.

## 6. Current leaderboard snapshot (Stanford Cars Top-1)

paperswithcode.com no longer serves leaderboards. These rows come from mirrors (hyper.ai) and are cross-checked against the papers.

| Rank | Method | Top-1 | Extra data? | Check |
|---|---|---|---|---|
| 1 | TResNet-L + PMD / PMAL (Liu, 2024) | 97.3 | No (IN-1k) | [P] |
| 2 | CMAL-Net, TResNet-L (Liu, PR 2023) | 97.1 | Flagged "extra data" on PwC mirror; PMD table lists it as TResNet-L | [S] |
| — | SigLIP g/16 or PE-core G, **frozen linear probe** | 96.5 | Yes (web-scale) | [P] |
| — | Domain-Adaptive Transfer, AmoebaNet-B | 96.2 | Yes (JFT) | [P] |
| — | OpenCLIP ViT-G/14 linear probe | 96.1 | Yes (LAION-2B) | [P] |
| — | SR-GNN, Xception @224 | 96.1 | No | [P] |
| — | TResNet-L @368 | 96.0 | No | [P] |
| — | EffNet-L2 + SAM | 95.96 | Yes (JFT) | [P] |
| — | CAP, Xception | 95.7 | No | [P] |
| — | CAL, ResNet-101 | 95.5 | No | [P] |
| — | TResNet-L + ML-Decoder | 96.41 | ? | (unverified, mirror only) |

The accuracy ceiling is about **97%**. Since around 2021, gains have come from strong backbones (TResNet-L, web-scale ViTs) more than from FGVC-specific modules. A frozen web-scale ViT with a linear head already reaches about 96.5%.

---

## 7. Practical takeaways (free T4 16 GB, 1–2 days)

**Best accuracy per GPU-hour, in order:**

1. **Frozen features plus a linear or logistic head on a CLIP-family model (hours, not days).**
   - Extract features once for all 16k images, plus their horizontal flips. Then fit a logistic regression or a linear layer on the CPU or GPU.
   - Published linear-probe numbers: CLIP L/14@336 at 91.5, OpenCLIP G/14 at 96.1, SigLIP-g and PE-G at 96.5.
   - On a T4, use ViT-L or SO400M-size models. g, G and 7B models run very slowly in fp16 on a T4.
   - Expect about 92–95% for L/SO400M-class models. This is my extrapolation from the tables, not a measured number.
   - This is the best baseline to have on day 1.
2. **Full fine-tuning of a CLIP / EVA-02 / SigLIP ViT-B or ViT-L at 224–336.** Use these settings:
   - Low LR (about 1e-5 to 3e-5), layer-wise LR decay, label smoothing.
   - Initialise the head from the linear probe or zero-shot text weights (LP-FT / WiSE-FT style).
   - AMP plus gradient checkpointing, 10–30 epochs.
   - This should beat the linear probe by a few points, but no T4-scale primary number was verified.
3. **CNN at 448 (ResNet-50 + PMG/CAL, or TResNet-L @368).**
   - These are proven ImageNet-only recipes at 95–97%.
   - They are expensive: PMD trains for 200 epochs, and 448² on a T4 is slow.
   - Choose this only if the hackathon forbids web-scale pretraining.
4. **Skip** PEFT-only approaches (VPT/adapters) for Cars, because they underperform full FT on this dataset. Also skip complex ViT-FGVC modules (TransFG etc.), which give little gain over a stronger backbone.

**Cheap boosters:**
- Higher resolution at test time.
- Horizontal-flip TTA.
- Averaging logits of the linear probe and the fine-tuned model (and the zero-shot head).
- An ensemble of two different backbones (e.g. a CLIP-family model plus DINOv2/v3).

**Suggested checkpoints**

Verified names:
- `timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (90.05 IN-1k top-1 per the model card)
- `timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`

Names I did not check this session (confirm on the HF Hub):
- `openai/clip-vit-large-patch14-336`
- `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`
- `google/siglip2-so400m-patch14-384`
- `facebook/dinov2-large`
- DINOv3 ViT-L distilled (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
- timm `convnextv2_base.fcmae_ft_in22k_in1k_384`
- timm TResNet-L

**Data-contamination and data-handling concerns**
- **DINOv2's LVD-142M was built by retrieval using the Stanford Cars *train* set as queries.** About 1M images were retrieved [P, DINOv2 appendix data-sources table]. Near-duplicates of benchmark *test/val* images were removed [P]. The pretraining data is therefore deliberately Cars-like. DINOv3 likely used similar curation (unverified).
- **Web-scale CLIP data (LAION, WebLI, MetaCLIP) is not deduplicated against Stanford Cars test images.** The test images themselves came from web search (unverified detail of Krause et al. 2013). EVA-02-CLIP-E reaches 94.6% *zero-shot*, which suggests very heavy exposure to captioned car-model images. CLIP's own overlap study found a median detected overlap of 2.2% across datasets [P]. I did not verify a Cars-specific figure.
- **Recommendation:** report an ImageNet-only baseline alongside any web-scale result, and check the hackathon rules on external pretraining. For a cleaner estimate, you can run a near-duplicate check, for example cosine similarity of CLIP embeddings between the test set and any extra data you add.
- **Getting the data:**
  - The original host (ai.stanford.edu/~jkrause/cars) is dead, and torchvision's `StanfordCars(download=True)` now raises an error (pytorch/vision #7545).
  - Use the Kaggle mirror and confirm that `cars_test_annos_withlabels.mat` exists. Some mirrors lack test labels.
- The dataset covers only cars up to 2012/2013 (Car-1000 paper), so real-world generalisation to newer cars will be poor.

---

## Sources
- B-CNN: https://arxiv.org/abs/1504.07889
- Compact Bilinear: https://arxiv.org/abs/1511.06062
- RA-CNN: https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf
- iSQRT-COV: https://arxiv.org/abs/1712.01034
- NTS-Net: https://arxiv.org/abs/1809.00287
- WS-DAN: https://arxiv.org/abs/1901.09891
- DCL: https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf
- S3N: https://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_Selective_Sparse_Sampling_for_Fine-Grained_Image_Recognition_ICCV_2019_paper.pdf
- Cross-X: https://arxiv.org/abs/1909.04412
- MC-Loss: https://arxiv.org/abs/2002.04264
- API-Net: https://arxiv.org/abs/2002.10191
- CIN: https://arxiv.org/abs/2003.05235
- PMG: https://arxiv.org/abs/2003.03836
- MMAL-Net: https://arxiv.org/abs/2003.09150
- CAL: https://arxiv.org/abs/2108.08728
- CAP: https://arxiv.org/abs/2101.06635
- SR-GNN: https://arxiv.org/abs/2209.02109
- PMAL/PMD (contains CMAL-Net comparison): https://arxiv.org/abs/2401.14336
- CMAL-Net listing: https://hyper.ai/en/papers/learn-from-each-other-to-classify-better
- TransFG: https://arxiv.org/abs/2103.07976
- FFVT: https://arxiv.org/abs/2107.02341
- RAMS-Trans: https://arxiv.org/abs/2107.08192
- AF-Trans: https://arxiv.org/abs/2110.01240
- SIM-Trans: https://arxiv.org/abs/2208.14607
- DCAL: https://arxiv.org/abs/2205.02151
- MetaFormer: https://arxiv.org/abs/2203.02751
- PIM: https://arxiv.org/abs/2202.03822
- HERBS: https://arxiv.org/abs/2303.06442
- IELT code: https://github.com/mobulan/ielt
- SCOPE (withdrawn): https://arxiv.org/abs/2508.06959
- Kornblith et al.: https://arxiv.org/abs/1805.08974
- EfficientNet: https://arxiv.org/abs/1905.11946
- TResNet: https://arxiv.org/abs/2003.13630
- Domain-Adaptive Transfer: https://arxiv.org/abs/1811.07056
- SAM: https://arxiv.org/abs/2010.01412
- DeiT: https://arxiv.org/abs/2012.12877
- BiT (no Cars result): https://arxiv.org/abs/1912.11370
- CLIP: https://arxiv.org/abs/2103.00020
- EVA-CLIP: https://arxiv.org/abs/2303.15389
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- VPT: https://arxiv.org/abs/2203.12119
- SSF: https://arxiv.org/abs/2210.08823
- V²APT: https://arxiv.org/abs/2503.17650
- GPS (CVPR 2024) supplement: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhang_Gradient-based_Parameter_Selection_CVPR_2024_supplemental.pdf
- CLIP vs DINO controlled study: https://arxiv.org/abs/2510.11835
- Car-1000: https://arxiv.org/abs/2503.12385
- Stanford Cars refined: https://github.com/morrisfl/stanford_cars_refined
- torchvision StanfordCars download issue: https://github.com/pytorch/vision/issues/7545
- timm EVA-02 card: https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in1k
- Leaderboard mirror: https://vercel.hyper.ai/en/sota/tasks/fine-grained-image-classification/benchmark/fine-grained-image-classification-on-stanford
