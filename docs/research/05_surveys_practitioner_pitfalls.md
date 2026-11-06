# Agent report: surveys, practitioner sources, and Stanford Cars benchmark pitfalls

**How it was verified.** Numbers marked "verified" were read from the paper's own text: I downloaded the PDF, ran pdftotext, and checked the tables. Numbers taken only from abstracts, search snippets or READMEs are marked "(unverified)" or "abstract only".

---

## 1. Survey and journal articles

### 1.1 Wei, Song, Mac Aodha, Wu, Peng, Tang, Yang, Belongie — "Fine-Grained Image Analysis with Deep Learning: A Survey"
IEEE TPAMI 44(12):8927–8948, Dec 2022. DOI 10.1109/TPAMI.2021.3126648. arXiv 2111.06119. Verified from the full text.

**Taxonomy.** Recognition methods fall into three paradigms:
- **Localization–classification subnetworks:** detection/segmentation, deep filters, attention.
- **End-to-end feature encoding:** high-order/bilinear pooling, specific loss functions.
- **External information:** web data, multimodal data, humans-in-the-loop.

The survey also covers fine-grained retrieval.

**Stanford Cars results (Table 2, test accuracy), selected rows:**

| Method | Backbone | Resolution | Cars accuracy | Note |
|---|---|---|---|---|
| Krause et al. CVPR15 | CaffeNet | 224 | 92.6% | uses bounding boxes |
| Bilinear CNN ICCV15 | VGG-16+VGG-M | 448 | 91.3% | |
| RA-CNN | VGG-19 | 448 | 92.5% | |
| MA-CNN | VGG-19 | 448 | 92.8% | |
| NTS-Net ECCV18 | ResNet-50 | 448 | 93.9% | |
| DFL-CNN | VGG-16 | 448 | 93.8% | |
| S3N | ResNet-50 | 448 | 94.7% | |
| DCL CVPR19 | ResNet-50 | 448 | 94.5% | |
| Cross-X | ResNet-50 | 448 | 94.6% | |
| DF-GMM | ResNet-50 | 448 | 94.8% | |
| Sun et al. AAAI20 | ResNet-50 | 448 | 94.9% | |
| API-Net | ResNet-50 | 448 | 94.8% | |
| API-Net | DenseNet-161 | 448 | 95.3% | |
| PMG ECCV20 | ResNet-50 | 448 | 95.1% | |
| MaxEnt | plain VGG-16 | 224 | 83.9% | |

Methods specialised for fine-grained work, on a ResNet-50 at 448, cluster at **93.8–95.3%**.

**Resolution (Table 4).** This uses CUB rather than Cars, with a vanilla ResNet-50 at each resolution:

| Resolution | 224 | 280 | 336 | 392 |
|---|---|---|---|---|
| Accuracy | 81.6% | 83.3% | 85.0% | 85.6% |

**Conclusions that matter for our decisions:**
1. Accuracy rises with input resolution, so 448 is the de-facto standard on these benchmarks.
2. Rigid objects such as cars and aircraft are easier than non-rigid ones such as birds and dogs. Part-based reasoning helps most on non-rigid objects.
3. Localization–classification pipelines trade off recognition against localization. They often need alternating or multi-stage training, which complicates tuning.
4. Under "next-generation datasets", Stanford Cars is listed among the classic benchmarks that are "largely saturated".
5. Open problems the survey names:
   - robustness to viewpoint, scale, resolution, corruptions and adversarial inputs;
   - objects that do not fill the frame. Real deployments rarely look like Stanford Cars' large, centred cars, which supports a detect-then-crop design;
   - domain adaptation, long tail, open-world settings, and resource-constrained embedded deployment.

### 1.2 Kornblith, Shlens, Le — "Do Better ImageNet Models Transfer Better?"
CVPR 2019, arXiv 1805.08974. Verified. This is not a survey, but it is the standard large-scale transfer study and includes Stanford Cars.

- ImageNet top-1 accuracy correlates strongly with fine-tuned transfer accuracy.
- **Stanford Cars is an outlier: pretraining gives little final-accuracy benefit.**
  - Inception v4 at 299: 92.7% from scratch versus 93.3% fine-tuned. At 448: 93.4%.
  - Reason given: ImageNet has more car images than Stanford Cars (12,455 vs 8,144) but only about 10 coarse car classes.
- Pretraining still makes convergence about 17× faster on average (measured as steps to reach 90% of maximum odds).
- Regularizers that slightly improve ImageNet accuracy (label smoothing, dropout and similar) make penultimate features much worse for transfer.
  - This matters if we use frozen features: logistic regression or SVM on top, or fusion with hand-crafted features.

### 1.3 Ni & Huttunen — "Vehicle Attribute Recognition by Appearance: Computer Vision Methods for Vehicle Type, Make and Model Classification"
J. Signal Processing Systems 2020, DOI 10.1007/s11265-020-01567-6, arXiv 2006.16400. Verified from the full text.

- Surveys hand-crafted methods (HOG/SIFT-style, mostly older) and deep methods for type, make and model recognition, with tables per dataset.
- Its main argument: plain classification is "doomed to fail on unseen car models", because new models appear every year.
- A metric-learning, gallery-based approach (ResNet-50 with triplet or lifted-structure loss, nearest-neighbour lookup) "outperforms the classification approach by a clear margin" in their experiments.
- It points to few-shot and multi-task learning as directions.

### 1.4 Other vehicle make/model recognition surveys (bibliographic or abstract only)
- **Gayen, Maity, Singh, Geem, Sarkar, "Two decades of vehicle make and model recognition – Survey, challenges and future directions"**, J. King Saud Univ. – CIS 36(1):101885, 2024. Abstract only.
  - Factors covered: lighting, viewpoint, occlusion.
  - Open needs: larger and more diverse datasets, robustness, real-time methods.
- **Boukerche & Ma, "Vision-based Autonomous Vehicle Recognition: A New Challenge for Deep Learning-based Systems"**, ACM CSUR 54(4), 2022, DOI 10.1145/3447866. Abstract only.
  - Splits the pipeline into vehicle detection → make/model recognition → re-identification. This matches our detect→classify architecture and is a citable framing.
- **Boukerche, Siddiqui, Mammeri et al., "Automated Vehicle Detection and Classification: Models, Methods, and Techniques"**, ACM CSUR 50(5), 2017. Bibliographic only.
- I found no dedicated make/model survey in IEEE T-ITS.

### 1.5 Hybrid hand-crafted + deep features
- **Nanni, Ghidoni, Brahnam, "Handcrafted vs. non-handcrafted features for computer vision classification"**, Pattern Recognition 71:158–172, 2017, DOI 10.1016/j.patcog.2017.05.025.
  - Fuses CNN transfer features (from the output layer, the penultimate layer, and several layers after dimensionality reduction) with LTP/LPQ/LBP variants, classified by SVMs, across many datasets.
  - From the abstract alone I could not confirm the size of any fusion gain (unverified).
  - The paper's methodology is directly reusable for our classical-CV integration and ablation: Wilcoxon signed-rank tests, and the Q-statistic to measure classifier independence.
- I found no Pattern Recognition or TPAMI survey dedicated to hand-crafted/deep fusion for image classification.

### 1.6 OOD and open-set recognition
- **Yang, Zhou, Li, Liu — "Generalized Out-of-Distribution Detection: A Survey"**, IJCV 2024, DOI 10.1007/s11263-024-02117-4, arXiv 2110.11334. Verified.
  - Unifies anomaly detection, novelty detection, open-set recognition, OOD detection and outlier detection.
  - Benchmark findings, run on OpenOOD:
    - post-hoc, inference-only scores generally beat methods that need special training;
    - augmentation-based methods (PixMix, CutMix, Mixup) and ensembles are the most effective;
    - extra outlier data "seems not necessary";
    - KNN on features is robust across benchmarks.
- **Vaze, Han, Vedaldi, Zisserman — "Open-Set Recognition: a Good Closed-Set Classifier is All You Need?"**, ICLR 2022 (oral). Verified via ar5iv and the GitHub README.
  - Open-set ability correlates strongly with closed-set accuracy.
  - The maximum-logit score on a well-trained classifier matches or beats the specialist method ARPL+.
  - Their Semantic Shift Benchmark (SSB) includes **a Stanford Cars split**: 98 known classes; 76 "Easy" and 22 "Hard" unknown classes (no "Medium" classes); 3,948 known-class test images.
  - ResNet-50 results:

| Method | Closed-set accuracy (98 classes) | AUROC Easy / Hard |
|---|---|---|
| MLS | 97.1% | 94.0 / 82.2 |
| ARPL+ | 96.9% | 94.8 / 83.6 |

  - The 97.1% is a 98-class figure and is not comparable to 196-way accuracy.
  - Their recipe: 600 epochs, RandAugment, label smoothing (both tuned on a validation set), and cosine learning rate with warmup.
  - This gives us a ready, citable protocol for evaluating "unknown car model" rejection.

### 1.7 Compression and quantization
- **Blalock, Gonzalez Ortiz, Frankle, Guttag — "What is the State of Neural Network Pruning?"**, MLSys 2020. Verified.
  - Reviewed 81 papers; finds no standard benchmarks and fragmented comparisons.
  - Pruning can improve the size/speed-versus-accuracy trade-off of a given architecture, but "generally does not help as much as switching to a better architecture". It helps most on architectures that are inefficient to begin with.
  - Magnitude pruning is a consistent strong baseline.
- **Gholami et al. — "A Survey of Quantization Methods for Efficient Neural Network Inference"**, 2021 book chapter (Low-Power Computer Vision), arXiv 2103.13630. Abstract only.
  - States "reductions of 4x to 8x are often realized in practice" when moving to low-precision integers of 4 bits or fewer.

---

## 2. Practitioner sources

### 2.1 Reproducible Stanford Cars numbers

| Source | Setup | Stanford Cars top-1 |
|---|---|---|
| ResNet Strikes Back (Wightman, Touvron, Jégou, arXiv 2110.00476, Table 5) | ResNet-50, DeiT-style fine-tuning. Pretraining: PyTorch default / A1 / A2 / A3 | 92.5 / 92.7 / 92.6 / 92.5. Fine-tuning "smooths" pretraining-recipe differences on Cars |
| DeiT (Touvron et al., Table 7) | DeiT-B 224 / DeiT-B↑384 / DeiT-B distilled ↑384; EfficientNet-B7 reference | 92.1 / 93.3 / 93.9; EfficientNet-B7 94.7 |
| EfficientNet paper (Tan & Le 2019) | B3 / B7; Inception-v4 baseline | 93.6 / 94.7; Inception-v4 93.4 |
| Kornblith et al. 2019 | Inception v4 fine-tuned at 299 / at 448; from scratch at 299 | 93.3 / 93.4; 92.7 |
| CLIP (Radford et al. 2021), ViT-L/14@336 | linear probe / zero-shot | 91.5 / 78.8 (my reading of Tables 10/11) |
| fast.ai forum (M. McGuire, 2019) | EfficientNet-B3 + Mish + Ranger, 299×299, 40 epochs, flat-cosine schedule, default fastai augmentations; **test set used as validation set** | 93.8% (5-run mean); about 93.4% without Mish |
| same thread | EfficientNet-B7 + Mish + Ranger, 40 epochs, 5 runs | 94.79% (std 0.094) |
| GitHub jianshen92/stanford-car-grab-challenge (unverified README) | ResNet-152 at 299 + mixup | 92.53%; ResNet-50 88.0→89.3% with mixup |

On CLIP specifically: the paper notes zero-shot CLIP beats logistic regression on ResNet-50 features by more than 20% on Stanford Cars. It attributes this to web data carrying make/model text, versus ImageNet's narrow supervision.

**What these sources agree on:**
- A modern ImageNet-pretrained CNN or ViT at about 299–448 px, plus standard augmentation, reaches **about 92–94%** with no FGVC-specific modules.
- FGVC-specific modules add about 1–2 points.
- The fast.ai author also found EfficientNet unstable with one-cycle and RMSProp, and stable with Ranger and a flat-then-cosine schedule (forum report).

### 2.2 Ultralytics (detector side), docs.ultralytics.com, current docs
- **Export:**
  - The `quantize` argument replaces the deprecated `half`/`int8`; it takes `16`, `8` or `32`.
  - ONNX INT8 uses ONNX Runtime static quantization with calibration data from `data=` (dataset YAML) and `fraction=`.
  - Other ONNX arguments: `dynamic`, `simplify=True` (onnxslim), `opset` (default 18), `nms`, `imgsz` (default 640).
- **Speed claims (vendor marketing, unverified):** "up to 5x GPU speedup with TensorRT and 3x CPU speedup with ONNX or OpenVINO".
- **Freezing (YOLOv5 tutorial):** Pascal VOC, YOLOv5m, 50 epochs. Freezing the backbone (`--freeze 10`) speeds training and cuts GPU memory at "a little" mAP cost. Training all layers gives the best accuracy. The exact numbers are only in images, so they are not extractable.
- **Practical note:** COCO-pretrained YOLO already detects "car" (and truck/bus). For our pipeline, detection can likely be used off the shelf, with no fine-tuning, just to crop the car region.

### 2.3 ONNX Runtime quantization docs (onnxruntime.ai)
- Static quantization is recommended for CNNs; dynamic for transformers and RNNs.
- Default and first choice: **QDQ format, S8S8**. If accuracy drops, try U8U8. Avoid S8S8 in QOperator format on x86.
- On AVX2/AVX512 CPUs without VNNI, U8S8 can saturate. Use `reduce_range` (7-bit weights) or U8U8. Per-channel quantization on those CPUs generally also needs reduce_range.
- Calibration methods: MinMax, Entropy, Percentile.
- Run `quant_pre_process()` first.
- `qdq_loss_debug` matches weights and activations between FP32 and INT8, to find layers to leave unquantized.
- No speedup numbers are given. Speed "depends on model and hardware", and older CPUs can get *slower*.

### 2.4 PyTorch quantization
- **Blog, "Practical Quantization in PyTorch":**
  - about 100 mini-batches are enough for calibration;
  - fuse Conv-BN-ReLU;
  - use symmetric per-channel quantization for weights and affine per-tensor for activations;
  - models over 10M parameters are more robust to quantization;
  - keep sensitive layers in FP32 (example: skipping 2 of 28 MobileNet-v1 conv layers recovers near-FP32 accuracy);
  - QAT: about 10% of the original schedule, starting at 1% of the original learning rate.
- **Deprecation:** the PyTorch docs say `torch.ao.quantization` is moving to **torchao**: eager mode to `quantize_`, FX to `torchao.quantization.pt2e`. They plan deletion "in 2.10 if there are no blockers". Docs snapshots for 2.11/2.12 carry the same notice, so I could not confirm the actual removal status.
- **Implication:** for a hackathon, quantizing through ONNX Runtime is the lower-risk path than the torch.ao APIs.

### 2.5 Hugging Face Spaces deployment (docs fetched today, 2026-11-06)
- **CPU Basic: 2 vCPU, 16 GB RAM, 50 GB non-persistent disk, free per hour.**
- **Policy change:** the docs now state that "creating a Space that runs on compute (Gradio or Docker) requires a paid plan" (PRO). Static Spaces are free.
- **Free accounts** in good standing (verified email, account older than 30 days) can host up to **2 ZeroGPU Gradio Spaces**.
  - ZeroGPU: half an RTX Pro 6000, 48 GB.
  - Daily quota: free account 5 min; unauthenticated visitors 2 min; PRO 40 min.
  - Default 60 s per `@spaces.GPU` call.
  - Gradio SDK only, PyTorch 2.8+, no `torch.compile`.
- **Sleep and cold starts:** free CPU Spaces sleep after **48 h** inactivity, and a visitor restarts them (cold start). Paid hardware can set a custom sleep time.
- Only ports 80, 443 and 8080 are open for outbound requests.
- **Implication:** check the team's HF account type now.
  - Plan for CPU inference with ONNX FP32 or INT8 on 2 vCPU.
  - Fallback options: a ZeroGPU Gradio Space under a free account older than 30 days, or the team's own host.

---

## 3. Benchmark and reproducibility pitfalls on Stanford Cars

1. **The official download is dead.** The ai.stanford.edu dataset URL returns 404 (torchvision issue #7545, opened Apr 2023).
   - torchvision disabled `download=True` from v0.18; it now raises an error pointing to manual Kaggle instructions.
   - The common mirror `jessicali9530/stanford-cars-dataset` has nested duplicate folders (`cars_test/cars_test`), and its devkit **lacks test labels**. Users source `cars_test_annos_withlabels.mat` from a Kaggle notebook.
   - Kaggle mirrors with a torchvision-compatible layout: `rickyyyyyyy/torchvision-stanford-cars` (Aug 2024) and `emanuelriquelmem/stanford-cars-pytorch` (Dec 2024).
   - On Hugging Face, `tanganke/stanford_cars` (parquet) has train 8,144 and test 8,041. It also has **8,041-image corrupted test splits**: contrast, gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate, spatter. These are ready-made robustness and ablation sets. The card gives no source or licence, and I did not verify its labels against the official ones.
2. **There is no official validation split.** Practitioners often tune on the test set; the fast.ai thread says so explicitly. Many reported "test accuracies" are therefore optimistically biased.
   - Remedy: hold out about 10% of train, stratified per class, for tuning and calibration. Report test accuracy once.
3. **Label noise.** Cleanlab (vendor blog, May 2023) reports "hundreds" of mislabeled images. Examples: a Jeep Grand Cherokee labeled as a Dodge Durango, wrong model within the correct make, out-of-class images, and images showing two cars. No exact count or peer review.
   - Implication: some of the error at about 94% is label noise. Worth noting in failure analysis.
4. **ImageNet overlap.** Kornblith Table H.1 found 10 train images (0.12%) and 14 test images (0.17%) of Stanford Cars that also appear in ImageNet-train (conservative CNN duplicate detector). Negligible.
5. **Web-scale pretraining contamination.**
   - CLIP's overlap analysis over 35 datasets: median overlap 2.2%, mean 3.2%, accuracy shifts of at most 0.6%. The Stanford Cars-specific figure is only in a figure, so not extracted.
   - OpenCLIP (Cherti et al. 2022) found pHash duplicates against LAION-400M of about 1% for most datasets; their table does not list Stanford Cars.
   - Conclusion: contamination is probably small, but unmeasured for Cars on LAION. If we use CLIP or other web-pretrained backbones, state this caveat.
6. **Train/test near-duplicates within Stanford Cars.** I found no published audit. The method from Barz & Denzler's "Do we train on test data?" (CNN-feature nearest neighbour plus manual check) could be run cheaply as part of failure analysis.
7. **Comparability.** Reported numbers mix:
   - resolutions (224 vs 448);
   - bounding-box use at test time (e.g. Krause 92.6% uses boxes);
   - backbones and pretraining data.
   Always report resolution, pretraining and whether ground-truth or detected crops were used. For our app, detected crops are the honest setting.

---

## Takeaways for our decisions

1. **Target:** a strong pretrained backbone at about 384–448 px with standard augmentation. Expect about 92–94% top-1 (RSB ResNet-50 92.5–92.7; DeiT-B↑384 93.3; EfficientNet-B3 93.6–93.8; B7 94.7). FGVC modules add about 1–2 points at extra complexity. Resolution is the cheapest lever.
2. **Pretraining helps speed more than final accuracy on Cars** (about 17× faster convergence). Always fine-tune. Do not expect big gains from a fancier ImageNet recipe.
3. **Detect, then crop.** The Wei survey names objects not filling the frame as a key real-world gap. COCO-pretrained YOLO "car" detection likely suffices without fine-tuning. Evaluate the classifier on detected crops, not only on the dataset's tight images.
4. **Low-confidence and unknown models:** use max-logit or max-softmax (post-hoc) thresholds tuned on a held-out split, possibly with feature-KNN. The OOD survey and Vaze et al. both find these competitive with specialised methods. Evaluate rejection on the SSB Stanford Cars Easy/Hard split. Consider a gallery or metric-learning fallback for new models (Ni & Huttunen).
5. **Hand-crafted fusion:** follow the Nanni et al. pattern (deep features plus LBP/LTP/colour into an SVM). Validate gains with a Wilcoxon test across seeds or folds. Do not presume a gain: the claimed improvement could not be confirmed from the abstract. Strong ImageNet regularizers can hurt frozen-feature transfer.
6. **Optimisation:** export to ONNX and apply ORT static INT8 in QDQ/S8S8 format, calibrated on about 100 batches from the train split. Check accuracy per layer with `qdq_loss_debug`. Prefer a smaller architecture over pruning (Blalock). Avoid building on deprecated `torch.ao.quantization`.
7. **Deployment:** HF CPU Basic is 2 vCPU / 16 GB, with a 48 h sleep and cold start. Gradio and Docker Spaces on compute now require a PRO plan; free accounts can host up to 2 ZeroGPU Gradio Spaces (5 min/day quota). Confirm account status early.
8. **Data hygiene:**
   - get data from a known Kaggle or HF mirror and verify counts (8,144 / 8,041 / 196) and test labels;
   - carve a validation split and never tune on test;
   - mention label noise (Cleanlab) and the unmeasured LAION overlap;
   - use the HF corrupted test splits for robustness ablations.

---

## Source URLs
- https://arxiv.org/abs/2111.06119 (Wei et al., TPAMI 2022)
- https://arxiv.org/abs/1805.08974 (Kornblith et al., CVPR 2019)
- https://arxiv.org/abs/2006.16400 (Ni & Huttunen, JSPS 2020)
- https://doaj.org/article/60de5e4223bc41158797b1341c0b1d64 (Gayen et al., JKSU-CIS 2024)
- https://www.x-mol.com/paper/1389667688725372928 (Boukerche & Ma, ACM CSUR 2022)
- https://pure.kfupm.edu.sa/en/publications/automated-vehicle-detection-and-classification-models-methods-and/ (Boukerche et al., ACM CSUR 2017)
- https://bearworks.missouristate.edu/articles-cob/358 (Nanni et al., Pattern Recognition 2017)
- https://arxiv.org/abs/2110.11334 (Yang et al., IJCV 2024)
- https://ar5iv.labs.arxiv.org/html/2110.06207 and https://github.com/sgvaze/osr_closed_set_all_you_need (Vaze et al., ICLR 2022)
- https://arxiv.org/abs/2003.03033 (Blalock et al., MLSys 2020)
- https://arxiv.org/abs/2103.13630 (Gholami et al.)
- https://arxiv.org/abs/2110.00476 (ResNet strikes back)
- https://arxiv.org/abs/2012.12877 (DeiT)
- https://arxiv.org/abs/1905.11946 (EfficientNet)
- https://arxiv.org/abs/2103.00020 (CLIP)
- https://arxiv.org/abs/2212.07143 (OpenCLIP scaling / LAION duplicates)
- https://forums.fast.ai/t/project-stanford-cars-with-fastai-v1/34311?page=2 and https://forums.fast.ai/t/project-stanford-cars-with-fastai-v1/34311/43
- https://www.github.com/jianshen92/stanford-car-grab-challenge
- https://github.com/pytorch/vision/issues/7545
- https://pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html
- https://huggingface.co/datasets/tanganke/stanford_cars
- https://cleanlab.ai/blog/csa/csa-3/
- https://docs.ultralytics.com/modes/export/
- https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/
- https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- https://pytorch.org/blog/quantization-in-practice/
- https://docs.pytorch.org/docs/stable/quantization.html
- https://huggingface.co/docs/hub/spaces-overview
- https://huggingface.co/docs/hub/spaces-gpus
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://arxiv.org/pdf/1902.00423 (Barz & Denzler, near-duplicate methodology)

Local PDF text extracts were used for verification during the survey and are not kept in the repository.
