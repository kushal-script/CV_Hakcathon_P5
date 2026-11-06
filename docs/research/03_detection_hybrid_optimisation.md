# 03 — Detection/Localisation, Hybrid Classical+Deep Integration, and Deployment Optimisation

Survey date: 2026-11-06. I checked numbers against the primary source (paper PDF, official docs or official repo) unless they are marked **(unverified)**. "Docs" means the official Ultralytics, Roboflow or framework documentation as fetched on the survey date.

---

## SCOPE A — Detection and localisation

### A1. Detector families

**Two-stage (torchvision reference numbers, COCO val2017 box AP)**

| Model | Paper | Box AP | Params | GFLOPs |
|---|---|---|---|---|
| Faster R-CNN R50-FPN | Ren et al., NeurIPS 2015 | 37.0 | 41.8M | 134 |
| Faster R-CNN R50-FPN v2 | (torchvision improved recipe) | 46.7 | 43.7M | 280 |
| Faster R-CNN MobileNetV3-L FPN | — | 32.8 | 19.4M | 4.5 |
| Mask R-CNN R50-FPN (box) | He et al., ICCV 2017 | 37.9 | 44.4M | 134 |
| RetinaNet R50-FPN v2 (one-stage, ref.) | Lin et al., ICCV 2017 | 41.5 | 38.2M | 152 |
| FCOS R50-FPN | Tian et al., ICCV 2019 | 39.2 | 32.3M | 128 |

R-CNN (Girshick, CVPR 2014) and Fast R-CNN (Girshick, ICCV 2015) are of historical interest only. Cascade R-CNN (Cai & Vasconcelos, CVPR 2018) chains detection heads at increasing IoU thresholds (0.5/0.6/0.7) and gains about 2–4 AP over Faster R-CNN at high IoU **(numbers unverified)**. Two-stage models fine-tune easily in torchvision or Detectron2. Exporting them to ONNX works but is fiddly because of dynamic shapes and RoIAlign. Source: torchvision model table.

**One-stage YOLO line (Ultralytics docs, COCO val mAP50-95 at 640 px)**

| Model (release) | n | s | m | l | x | Notes |
|---|---|---|---|---|---|---|
| YOLOv8 (Jan 2023) | 37.3 | 44.9 | 50.2 | 52.9 | 53.9 | anchor-free, decoupled head; v8n 3.2M params, 80.4 ms CPU-ONNX |
| YOLOv9 (Wang, Yeh, Liao; ECCV 2024) | 38.3 (t) | 46.8 | 51.4 | 53.0 (c) | 55.6 (e) | PGI + GELAN |
| YOLOv10 (Wang A. et al., Tsinghua; NeurIPS 2024) | 38.5 | 46.3 | 51.1 | 53.2 | 54.4 | consistent dual assignment → **NMS-free**; T4 TRT FP16 1.84–10.7 ms |
| YOLO11 (27 Sep 2024) | 39.5 | 47.0 | 51.5 | 53.4 | 54.7 | 11n: 2.6M params, 56.1 ms CPU-ONNX, 1.5 ms T4-TRT |
| YOLO12 (Tian, Ye, Doermann; NeurIPS 2025) | 40.6 | 48.0 | 52.5 | 53.7 | 55.2 | Area Attention, R-ELAN, optional FlashAttention. Ultralytics warns of training instability, higher memory use and slower CPU throughput, and recommends YOLO11/YOLO26 for production |
| **YOLO26 (Jan 2026)** | **40.9** | **48.6** | **53.1** | **55.0** | **57.5** | current Ultralytics flagship (details below) |

**YOLO26 key changes (docs + official paper arXiv 2606.03748):**
- Dual head. The one-to-one head is NMS-free end-to-end; its e2e mAP is 40.1/47.8/52.5/54.4/56.9 for n→x. The one-to-many head uses NMS.
- **Caveat:** the current docs say "Prediction and validation default to the one-to-many head for accuracy." You get NMS-free inference with `nms=False`. The CPU and T4 speed benchmarks use the NMS-free head.
- DFL is removed from box regression, which simplifies the head and export.
- MuSGD optimiser (an SGD + Muon hybrid), ProgLoss, and STAL (small-target-aware label assignment).
- Docs claim "up to 43% faster CPU ONNX inference" for 26n vs 11n (Xeon @2.0 GHz). Measured: 26n is 38.9 ms vs 11n 56.1 ms CPU-ONNX, with 2.4M params and 5.5 GFLOPs.
- Exports include ONNX, TensorRT, OpenVINO, CoreML and LiteRT, e.g. `model.export(format="onnx")`.
- The docs advertise **YOLO27 as "coming soon" (waitlist only, not released)**.

All Ultralytics models fine-tune with one line (`model.train(data=...)`) and export with one line. This is the lowest-friction option for a hackathon. **License:** Ultralytics is AGPL-3.0 **(well known; not re-checked here)**.

**DETR family**

| Model | Venue | COCO AP | Speed / size | Fine-tune / export |
|---|---|---|---|---|
| DETR (Carion et al.) | ECCV 2020 | 42.0 R50 **(unverified)** | slow convergence (500 ep) | research |
| Deformable DETR (Zhu et al.) | ICLR 2021 oral | AP not in abstract | "10× fewer training epochs", better on small objects | research |
| DINO-DETR (Zhang H. et al.) | ICLR 2023 | 49.4 (R50, 12 ep); 51.3 (24 ep); 63.3 test-dev (SwinL + Objects365) | heavy | MMDet/detrex |
| RT-DETR (Zhao Y. et al., "DETRs Beat YOLOs") | CVPR 2024 | R18 46.5 / R34 48.9 / R50 53.1 / R101 54.3 | 217 / 161 / 108 / 74 FPS T4-TRT FP16; 20–76M params | Ultralytics ships RT-DETR-L (53.0) / X (54.8) |
| RT-DETRv2 (Lv et al., arXiv 2407.17140) | arXiv 2024 | S 48.1 / M 51.9 / L 53.4 / X 54.3 | same speeds as v1 tiers | official repo (not in Ultralytics) |
| D-FINE (Peng et al.) | ICLR 2025 spotlight | N 42.8 / S 48.5 / M 52.3 / L 54.0 / X 55.8 (COCO only); up to 59.3 with Objects365 | 2.1–12.9 ms T4-TRT; 4–62M params | repo has `export_onnx.py` + trtexec |
| RF-DETR (Roboflow, arXiv 2511.09554) | ICLR 2026 (per repo) | Nano 48.4 / S 53.0 / M 54.7 / L 56.5 (Apache-2.0); XL 58.6 / 2XL 60.1 (PML licence) | 2.3–17.2 ms T4-TRT FP16; ~30–34M params for N–L | "designed for fine-tuning"; ONNX export via the `rfdetr` package **(not confirmed on the README page)** |

D-FINE's key ideas are Fine-grained Distribution Refinement (box regression as iterative distribution refinement) and GO-LSD (self-distillation from the last decoder layer to earlier layers at no inference cost).

**Open-vocabulary detectors (zero-shot "car" boxes, no training)**
- OWL-ViT (Minderer et al., ECCV 2022). CLIP backbone with box heads. **Numbers unverified.**
- Grounding DINO (Liu S. et al., ECCV 2024; arXiv 2303.05499). **52.5 AP zero-shot on COCO**, 26.1 mean AP on ODinW.
- YOLO-World (Cheng et al., CVPR 2024; arXiv 2401.17270). **35.4 AP on LVIS at 52.0 FPS on V100.**
- These detectors are useful as a zero-training fallback or as box pseudo-labellers. They are heavier and less tight than a fine-tuned YOLO.

### A2. Stanford Cars bounding boxes: measured crop vs full-image gains

Dataset facts: 196 classes, 8,144 train / 8,041 test images, one GT box per image (Krause et al., 3dRR-13 @ ICCV-W 2013; split verified via the WS-DAN Table 1).

- **Krause et al., CVPR 2015, "Fine-Grained Recognition Without Part Annotations"** (Table 5, cars-196, VGGNet):
  - R-CNN with a predicted box, no fine-tuning: **57.4%**
  - R-CNN+ft (predicted box at test): **88.4%**
  - CNN + GT box at test + ft: **89.0%**
  - Their full model, no box at test: **92.6%**; with GT box at test: **92.8%**
  - On CUB the GT box at test adds only **+1.3%** (1.2% with ft).
  - Takeaway: with a good detector, predicted boxes lose only about 0.2–0.6 pts vs GT boxes on Cars.
  - The method itself is hybrid. It runs GrabCut-style co-segmentation initialised from boxes, and bbox regression lifts R-CNN detection AP on CUB **from 88.1 to 92.9**.
- **Lin, RoyChowdhury, Maji, ICCV 2015, Bilinear CNN** (Table 1):
  - Cars: B-CNN(D,M) **88.2%** with no boxes at all. The TPAMI version reaches **91.3%**.
  - Cars, hand-crafted: the FGComp'13 winner (a heavily engineered SIFT-Fisher-vector ensemble) goes **82.7% → 87.9% when boxes are given** (+5.2). On aircraft the same ensemble gains only 80.7 → 81.5. The authors conclude "localization in the cars dataset is more important than in aircrafts."
  - Birds: box cropping adds ~10 pts to FC/FV-CNN with the M-Net, ~6 pts with the D-Net, and only ~1 pt for B-CNN(D,M) (80.9 → 81.9). Strong models learn to localise implicitly.
- **Hu et al., arXiv 1901.09891, WS-DAN** (attention cropping):
  - **94.5%** on Cars (Inception-v3).
  - CUB ablation: attention cropping 87.8% vs random cropping 86.8% vs baseline 86.4%. The localisation mIoU of attention-cropped regions is **58.7% vs 40.0%** for random crops.
  - The authors skip localisation evaluation on Cars because "most of the images … occupy the whole space," i.e. Cars boxes are large, so crop gains are smaller than on birds.
- **Su & Maji, BMVC 2017, Cross-Quality Distillation** (arXiv 1604.00433):
  - On CUB, VGG-M trained on GT crops gets 67.0% on crops but **57.4% on full images**. Training on full images gives 60.8%. Distilling crop→full gives 64.4%.
  - They quote Krause: a detector instead of GT boxes costs ~2.6 pts on CUB (67.9 → 65.3).
  - CQD also covers "color → edges" and low-resolution cars.

**No paper I found reports COCO-style detection mAP on Stanford Cars boxes as a benchmark.** Expect near-saturated IoU@0.5 because there is one large car per image. Measure it yourself (A3).

### A3. Metrics
- **IoU** = area(B_pred ∩ B_gt) / area(B_pred ∪ B_gt).
- **COCO protocol** (cocodataset.org detection-eval):
  - AP is averaged over **10 IoU thresholds, .50:.05:.95**.
  - AP50 is the PASCAL VOC metric; AP75 is the "strict metric".
  - Area buckets: small < 32², medium 32²–96², large > 96².
  - AR@1/10/100. At most **100 detections per image**.
  - AP is averaged over categories, and "We make no distinction between AP and mAP."
- **Single-object-per-image evaluation** (recommended for Stanford Cars), on the top-1 box per image:
  - (a) **mean IoU**.
  - (b) **localisation accuracy / CorLoc**: % of images with IoU ≥ 0.5. WS-DAN reports the complement, "localization error = % with IoU < 50%". Grad-CAM uses ILSVRC "top-1 loc", which requires the correct class *and* IoU ≥ 0.5.
  - (c) COCO mAP50 and mAP50-95 via `pycocotools` or Ultralytics `val`, for comparability.
  - (d) **end-to-end top-1 classification accuracy** with predicted crops vs GT crops vs full image. This is the number that matters.

---

## SCOPE B — Hybrid classical + deep integration

### B1. Classical preprocessing: does it measurably help?
- **Dodge & Karam, QoMEX 2016** (arXiv 1604.04004). VGG/GoogLeNet/CaffeNet on ImageNet are "susceptible … particularly to blur and noise". JPEG and low contrast hurt much less. This argues for denoising/deblurring before CNNs more than for contrast stretching.
- **ExDark (Loh & Chan, CVIU 2019, vol. 178)**. 7,363 low-light images, 12 classes, box labels. The authors find low-light effects go deeper than simple illumination invariance can fix.
- **PE-YOLO (Yin et al., arXiv 2307.10953)**, ExDark mAP@0.5 with YOLOv3:

  | Setting | mAP@0.5 |
  |---|---|
  | baseline | **76.4** |
  | + KinD pre-enhancement | 76.3 |
  | + MBLLEN | 76.8 |
  | + Zero-DCE | 76.9 |
  | PE-YOLO (end-to-end enhancement) | 78.0 |

  Off-the-shelf enhancement before a detector gives **≤ 0.5 mAP**. Gains need joint training.
- Other secondary studies report similar marginal gains, e.g. YOLOv7 0.785 → 0.794 with Zero-DCE **(unverified primary)**. One reports TBEFN two-stage 0.49 → 0.574 (PMC12190514, **unverified detail**).
- **CLAHE** (Zuiderveld, Graphics Gems IV 1994). Evidence is mostly small single-dataset medical or other studies. One reports VGG16/InceptionV3/EfficientNet improving but ResNet34 dropping 95% → 84% **(unverified, secondary)**. Treat CLAHE as condition-specific: apply it on low-contrast or dark inputs only, and validate.
- **White balance.** Afifi & Brown (ICCV 2019, arXiv 1912.06960) show that colour-constancy errors fool classifiers and segmenters. WB-emulating augmentation and WB pre-correction give "notable improvements" on CIFAR/ADE20K (no numbers in abstract). Gray-world (Buchsbaum 1980) and MSRCR (Jobson et al., IEEE TIP 1997) are the classical baselines. I found **no study quantifying them on FGVC**.
- **Shape and edges.** Geirhos et al. (ICLR 2019, arXiv 1811.12231) find ImageNet CNNs are texture-biased. Shape-biased pretraining (Stylized-ImageNet) lifts Faster R-CNN from **70.7 → 75.1 mAP50 on VOC07** and **52.3 → 55.2 on COCO**.

### B2. Hand-crafted + CNN feature fusion
- **Hand-crafted baselines on Cars:**
  - SIFT-FV **59.2%** (B-CNN TPAMI Table 2) vs fine-tuned B-CNN 91.3%.
  - Engineered SIFT-FV ensembles (Gosselin et al., "Revisiting the Fisher vector for fine-grained classification", PRL 2014) reach **82.7%**, or 87.8–87.9% with boxes.
  - Gosselin et al. late-fuse two deliberately complementary FV systems; per-component gains **unverified**.
- **Nanni, Ghidoni, Brahnam, Pattern Recognition 71 (2017)**. Fuse CNN transfer features with many handcrafted descriptors via SVM score fusion. They report strong results across many datasets **(per-dataset gains unverified; no FGVC-car result found)**.
- **I found no primary paper with a measured LBP/HOG/colour-histogram + CNN fusion gain on Stanford Cars.** Vehicle-MMR fusion papers exist (e.g. CS-LBP + HOG; PCANet + HOG + Hu moments, BJUT), but they are not compared against modern CNNs.
- Practical reading: at ≥ 90% CNN accuracy, expect fusion gains of ≤ 1 pt. Late or score fusion (e.g. logistic regression on [CNN logits ‖ HSV-histogram ‖ HOG] or weighted logit averaging) is cheap to test and gives an honest ablation.

### B3. Classical ROI refinement
- **GrabCut** (Rother et al., SIGGRAPH 2004) initialised from boxes.
  - Khoreva et al., CVPR 2017 "Simple Does It" (arXiv 1603.07485), training-free instance masks from detector boxes on VOC12: rectangle **21.6 mAPr@0.5**, ellipse 29.5, MCG 28.3, **GrabCut 38.5**, GrabCut+ (HED edges as pairwise term) **41.1**.
  - As training labels for semantic segmentation: Box 61.2 → GrabCut+ 63.4 → GrabCut+i 64.3 val mIoU. M∩G+ reaches ~95% of full supervision.
  - Krause 2015 also uses box-initialised GrabCut for car/bird foregrounds.
- **Grad-CAM cropping** (Selvaraju et al., ICCV 2017; IJCV 2020; arXiv 1610.02391). Threshold the map at **15% of max**, take the largest connected component, draw its box. ILSVRC-15 top-1 localisation error with VGG-16: **56.51%** vs CAM 57.20% vs c-MWP 70.92%, with no retraining.
- **Attention-crop FGVC.** WS-DAN uses a crop threshold θc (see A2 for numbers). RA-CNN (Fu et al., CVPR 2017) gets 92.5% on Cars and NTS-Net (Yang et al., ECCV 2018) 93.9%. Both numbers are as listed in WS-DAN Table 6 (secondary).
- **Spectral-residual saliency** (Hou & Zhang, CVPR 2007). Available as `cv2.saliency.StaticSaliencySpectralResidual_create` (opencv-contrib). It is a cheap classical fallback ROI when the detector misses. **No FGVC gain numbers found.**

### B4. Edge, contour or geometric priors as extra input or auxiliary signal
- **Gated-SCNN** (Takikawa et al., ICCV 2019, arXiv 1907.05740). A separate gated shape stream gives about **+2% mIoU** and **+4% boundary F-score** on Cityscapes. Fusing Canny image gradients is a detail **not confirmed from the abstract**.
- **CoordConv** (Liu R. et al., NeurIPS 2018). Coordinate channels give **24% better IoU** for Faster R-CNN on a toy MNIST detection task. There is no COCO result in the abstract.
- **Khoreva DeepLabBOX.** RGB plus a binary box-mask 4th channel reaches 44.8 mAPr@0.5 vs GrabCut+ 41.1. This is a concrete "geometric prior as input channel" result.
- **RGB + Canny 4th channel for classification.** Only weak evidence: arXiv 2401.07028 adds edges back to RGB and reports gains on CIFAR-10/Caltech101 **(unverified numbers)**.
- Su & Maji distil colour → edge-map models on CUB/Cars, which shows edges alone carry a lot of fine-grained signal.

### B5. Model optimisation for deployment
- **Krishnamoorthi, arXiv 1806.08342 (2018):**
  - INT8 PTQ (per-channel weights, per-layer activations) stays within **2% of FP32** across many CNNs.
  - QAT narrows the gap to **~1%**; 4-bit QAT loses 2–10%, worse for small nets.
  - **2–3× CPU speed-up**; weights are ~4× smaller.
- **Jacob et al., CVPR 2018.** Integer-only inference; this is the basis of TFLite/ORT INT8.
- **ONNX Runtime docs:**
  - Dynamic quantization is "recommended … for RNNs and transformer-based models"; static (calibrated) quantization suits **CNNs**.
  - **S8S8 + QDQ is the default and "should be the first choice."** Try U8U8 only if accuracy drops a lot.
  - On AVX2/AVX512 machines without VNNI, U8S8 can **saturate**. Use `reduce_range` (and enable it whenever you use per-channel).
  - Calibration options: MinMax, Entropy, Percentile.
  - Debug with `qdq_loss_debug`. ORT does not do QAT.
- **Pruning on FGVC.** Luo & Wu (arXiv 1706.05791, entropy-based filter pruning) take VGG-16 on CUB from **71.95% → 69.55% (Pruned-FC) / 69.69% (Pruned-GAP)**. They note "fine-grained recognition … may need more channels to represent the subtle inter-class difference." This is the only primary evidence I found that FGVC is more compression-sensitive.
- **No primary study found that measures INT8 PTQ drop specifically on Stanford Cars.** ProtoQuant (arXiv 2602.06592) quantises prototypes on CUB/Cars, not backbones. **Measure it yourself:** FP32 vs FP16 vs INT8-static top-1 on the test split.
- **Distillation.** Hinton et al. 2015 (arXiv 1503.02531). Su & Maji CQD (above) is the FGVC-specific distillation evidence on CUB/Cars.
- **Other references.** Magnitude pruning / Deep Compression (Han et al., ICLR 2016) and structured filter pruning (Li et al., ICLR 2017). Their reported gains are on ImageNet/CIFAR, **not FGVC (numbers unverified here)**. NNCF/OpenVINO (Kozlov et al., arXiv 2002.08679) claims <0.5% top-1 drop for most ImageNet models **(unverified, from snippet)**.

---

## What to actually use in a 1–2 day hackathon

1. **Detector:** YOLO26n or YOLO26s (or YOLO11s if you hit tooling issues).
   - Start with **COCO-pretrained "car" class only (class 2)**, zero training. Then fine-tune 10–20 epochs on Stanford Cars boxes as a single class "car" if IoU is poor.
   - Export to ONNX for CPU. RF-DETR-Nano/Small (Apache-2.0) is the transformer alternative if AGPL is a problem.
   - Grounding DINO / YOLO-World is a zero-shot fallback only.
2. **Classical ROI refinement** (the visible "hybrid" piece):
   - Expand the detector box by 5–10%.
   - Run `cv2.grabCut` with `GC_INIT_WITH_RECT` (5 iterations), then Canny + `findContours`.
   - Take the largest external contour → tight box, plus geometric features: aspect ratio, extent, solidity, Hu moments.
   - If there is no detection, fall back to spectral-residual saliency or a Grad-CAM threshold-15% box.
3. **Preprocessing:**
   - Gray-world WB plus CLAHE on the L channel (LAB) **only when mean luminance is low**, e.g. V < 0.35.
   - Add light bilateral/NLM denoising for dark images.
   - Ablate it. The literature says gains are small or conditional.
4. **Classifier:** fine-tune a modern ImageNet backbone (EfficientNet/ConvNeXt-T/ViT-S) on **detector crops** (train on GT crops plus jittered predicted crops).
   - Report full-image vs GT-crop vs predicted-crop accuracy. Krause shows predicted ≈ GT on Cars (88.4 vs 89.0).
5. **Feature fusion:** late-fuse CNN embedding/logits with an HSV colour histogram, a HOG/LBP texture vector and contour-geometry features through a small logistic-regression or MLP head. Report it honestly even if the gain is ≤ 1 pt.
6. **Optimisation:**
   - ONNX Runtime static INT8 (QDQ, S8S8, per-channel; Entropy or Percentile calibration on ~200 train crops) vs FP32 vs FP16.
   - Report top-1, latency and size, and watch for the AVX2 saturation issue.
7. **Metrics:** mean IoU, IoU ≥ 0.5 localisation accuracy, mAP50/mAP50-95 (pycocotools), and end-to-end top-1/top-5 accuracy with a per-stage latency table.

---

## Sources
- Ultralytics models: https://docs.ultralytics.com/models/yolo26/ · https://docs.ultralytics.com/models/yolo11/ · https://docs.ultralytics.com/models/yolo12/ · https://docs.ultralytics.com/models/yolov10/ · https://docs.ultralytics.com/models/yolov9/ · https://docs.ultralytics.com/models/yolov8/ · https://docs.ultralytics.com/models/rtdetr/
- YOLO26 paper: https://arxiv.org/abs/2606.03748 ; review: https://arxiv.org/abs/2601.12882
- RT-DETR / RT-DETRv2: https://github.com/lyuwenyu/RT-DETR · https://arxiv.org/abs/2304.08069 · https://arxiv.org/abs/2407.17140
- D-FINE: https://github.com/Peterande/D-FINE
- RF-DETR: https://github.com/roboflow/rf-detr · https://arxiv.org/abs/2511.09554
- DINO: https://arxiv.org/abs/2203.03605 ; Deformable DETR: https://arxiv.org/abs/2010.04159
- Grounding DINO: https://arxiv.org/abs/2303.05499 ; YOLO-World: https://arxiv.org/abs/2401.17270
- torchvision detection table: https://docs.pytorch.org/vision/stable/models.html
- Krause et al. CVPR 2015: https://openaccess.thecvf.com/content_cvpr_2015/html/Krause_Fine-Grained_Recognition_Without_2015_CVPR_paper.html
- B-CNN (ICCV 2015 v1 / TPAMI): https://arxiv.org/abs/1504.07889
- WS-DAN: https://arxiv.org/abs/1901.09891
- Su & Maji, Cross-Quality Distillation: https://arxiv.org/abs/1604.00433
- COCO detection eval: https://cocodataset.org/#detection-eval
- Dodge & Karam 2016: https://arxiv.org/abs/1604.04004
- ExDark (Loh & Chan): https://arxiv.org/abs/1805.11227
- PE-YOLO: https://arxiv.org/abs/2307.10953
- Low-light two-stage study (secondary): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12190514/
- Afifi & Brown ICCV 2019: https://arxiv.org/abs/1912.06960
- Geirhos et al. ICLR 2019: https://arxiv.org/abs/1811.12231
- Nanni et al. 2017: https://doi.org/10.1016/j.patcog.2017.05.025
- Gosselin et al., Revisiting the Fisher vector: https://www.researchgate.net/publication/264425714_Revisiting_the_Fisher_vector_for_fine-grained_classification
- Khoreva et al. CVPR 2017: https://arxiv.org/abs/1603.07485
- Grad-CAM: https://arxiv.org/abs/1610.02391
- Gated-SCNN: https://arxiv.org/abs/1907.05740 ; CoordConv: https://arxiv.org/abs/1807.03247
- Edge enhancement for classification: https://arxiv.org/abs/2401.07028
- Krishnamoorthi quantization whitepaper: https://arxiv.org/abs/1806.08342
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- Entropy-based pruning (CUB): https://arxiv.org/abs/1706.05791
- ProtoQuant: https://arxiv.org/abs/2602.06592
- NNCF: https://arxiv.org/abs/2002.08679
- Hinton et al. KD: https://arxiv.org/abs/1503.02531
