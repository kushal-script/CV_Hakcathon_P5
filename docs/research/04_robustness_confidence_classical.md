# 04: Classical baselines, robustness, confidence and evaluation for Stanford Cars FGVC

*Literature survey, compiled 2026-11-06. Verification key: **[P]** means the number was read in the primary source (paper PDF or the author's dissertation reprint). **[S]** means it was read in a peer-reviewed secondary paper that cites the primary; the citing paper is named. **(unverified)** means I could not confirm it from the primary source, or from any reliable secondary source.*

> Note on access: HAL (Anubis bot-wall), openaccess.thecvf.com (HTTP 403) and core.ac.uk (403) blocked automated fetches. Classical-era numbers were therefore read from Krause's 2016 Stanford dissertation, which reprints the 3dRR'13, ICPR'14 and CVPR'15 papers, and from Lin et al.'s bilinear-CNN paper (arXiv 1504.07889).

---

## 1. Classical (pre-deep and early-deep) pipelines on Stanford Cars

Stanford Cars has 196 classes at make/model/year level, 8,144 training and 8,041 test images. It was introduced as "car-197" in the 3dRR'13 paper and later released with 196 classes. In 2013 it was also the cars domain of the FGComp challenge. FGComp Track 1 gave bounding boxes at test time; Track 2 did not.

| Method | Paper | Cars accuracy | Verif. |
|---|---|---|---|
| BubbleBank (BB), 2D | Deng et al. CVPR'13 (as re-run by Krause) | 63.6% | [P] Krause 3dRR'13 / dissertation |
| LLC + SPM (SIFT BoVW-style) | Wang et al. CVPR'10 (re-run by Krause) | 69.5% | [P] |
| SPM-3D-L | Krause et al. 3dRR'13 | 65.6% | [P] |
| BB-3D-G (3D-lifted BubbleBank) | Krause et al. 3dRR'13 | 67.6% | [P] |
| Stack of SPM, BB, SPM-3D-L, BB-3D-G | Krause et al. 3dRR'13 | 75.5% | [P] |
| CNN trained from scratch (no ImageNet pretraining) | Krause et al. ICPR'14 | 70.5% | [P] |
| CNN-SPM (small / large grid) | Krause et al. ICPR'14 (ablation) | 67.9% / 69.3% | [P] |
| **ELLF** (Ensemble of Localized Learned Features) | Krause, Gebru, Deng, Li, Fei-Fei, ICPR'14 | **73.9%** | [P] |
| Fisher vector, SIFT, 256-word GMM (baseline) | Lin et al. ICCV'15 | 51.2% (arXiv v1) [P]; 59.2% in the ICCV version (unverified) | [P] v1 |
| **Inria-Xerox Fisher-vector ensemble**, no test-time bbox (FGComp Track 2, winner) | Gosselin, Murray, Jégou, Perronnin, PRL 49:92-98, 2014 | **82.7%** | [S] Lin et al. 1504.07889 |
| Same, with bbox (Track 1) | Gosselin et al. 2014 | **87.8%** (text also gives 87.9%) | [S] Lin et al. |
| Symbiotic segmentation + parts (FGComp entry) | Chai, Lempitsky, Zisserman ICCV'13 | 78.0% | [S] Lin et al. |
| R-CNN (no fine-tune) / R-CNN + fine-tune | Krause et al. CVPR'15 table | 57.4% / 88.4% | [P] |
| CNN + GT bbox / + fine-tune | Krause et al. CVPR'15 | 59.9% / 89.0% | [P] |
| Co-segmentation + alignment parts (no part annotations) | Krause, Jin, Yang, Fei-Fei CVPR'15 | 92.6% (no test bbox), 92.8% (with GT bbox) | [P] |
| B-CNN (M,M) / B-CNN (D,M), fine-tuned | Lin, RoyChowdhury, Maji ICCV'15 | 82.3% / 88.2% (arXiv v1) | [P] |
| FV-CNN (D), fine-tuned | Lin et al. ICCV'15 | 79.2% | [P] |

What each paper contributes:

- **Gosselin et al., "Revisiting the Fisher vector for fine-grained classification" (PRL 2014).** This is the FGComp 2013 winner. It late-fuses two complementary FV systems built from dense multi-scale SIFT and colour descriptors, with descriptor filtering and spatial-coordinate coding. Lin et al. list how it differs from a vanilla FV: a 1024-word GMM instead of 256, 1×1 + 3×1 spatial pyramid pooling, several SIFT variants, and multi-scale SIFT. **I could not open the PRL/HAL PDF (hal-01056223). The 82.7% and 87.8% figures come only from Lin et al.'s quotation of the FGComp leaderboard and are not confirmed in Gosselin's own tables.** Lin et al. also note that adding the bbox lifts FV by about 5 points on cars but under 1 point on aircraft, so localisation matters more for cars.
- **Krause et al., 3dRR'13.** Fits 3D CAD models to estimate viewpoint, rectifies patches, and pools descriptors in azimuth-elevation space. On car-197, the 3D lifting beat its own 2D counterpart (BB-3D-G 67.6% vs BB 63.6%). Plain LLC-SPM (69.5%) still beat it, which the authors blame on CAD-model coverage for some body types, such as pickups. Horizontal-flip augmentation added 8.6 to 13.3 points on BMW-10, which shows how much viewpoint coverage in training matters.
- **Krause et al., ICPR'14 (ELLF).** Unsupervised part discovery (GrabCut segmentation plus alignment), with CNN features pooled per part, then a linear SVM. The failure analysis is directly reusable. Part detectors misfire when GrabCut segmentation fails and under 180° rotations (front versus rear views). The most-confused pairs differ only in tiny details, for example Audi S5 vs A5 Coupe, Aston Martin V8 Vantage Coupe vs Convertible, Chevrolet Express vs GMC Savana van, and Silverado Regular vs Extended Cab.
- **Gavves et al., "Fine-grained categorization by alignments" (ICCV'13; IJCV 2015, 111(2):191-212).** Unsupervised global/local alignment followed by FV encoding, evaluated **on CUB-2011 and Stanford Dogs only. I found no Stanford Cars result.** One third-party summary gives 60.1% on CUB and 60.3% on Dogs (unverified).
- **POOF (Berg & Belhumeur CVPR'13)** and **DPD deformable part descriptors (Zhang et al. ICCV'13).** These are part-annotation-based. CUB-2011 accuracy is 56.8% (POOF) and 51.0% (DPD), as listed in Razavian et al.'s Table 3 [S]. No Cars results. They need part keypoints, which Stanford Cars does not provide.
- **Razavian et al., "CNN Features off-the-shelf: an astounding baseline" (CVPRW 2014; arXiv 1403.6382).** OverFeat fc features with a linear SVM. CUB-2011: CNN-SVM 53.3%, CNNaug-SVM 61.8%. Oxford Flowers-102 without segmentation: 74.7% and 86.8% [P]. They report no Stanford Cars result. The "off-the-shelf" equivalent on cars is FC-CNN without fine-tuning in Lin et al.: 33.5% (M-Net) and 32.0% (VGG-D) [P]. These are far below fine-tuned results (50.9% and 67.7%). **For cars, generic frozen features transfer poorly, and the domain gap is larger than for birds.**

---

## 2. Robustness

**Corruption protocol.** Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations", ICLR 2019 (arXiv 1903.12261). ImageNet-C has 15 corruptions in four families (noise, blur, weather, digital) at 5 severities, plus 4 held-out corruptions. The metrics are Corruption Error, normalised by AlexNet's error and averaged into mCE, and Relative mCE, which measures the degradation relative to clean error. Their headline finding is that relative robustness barely changed from AlexNet to ResNet. The protocol details come from my knowledge of the paper; the abstract I fetched confirms only the benchmark's purpose.

**Corruption benchmarks for fine-grained data.**
- Yucel, Cinbis, Duygulu, "How robust are discriminatively trained zero-shot learning models?", Image and Vision Computing 2022 (arXiv 2201.10972). Releases **CUB-C** (with SUN-C and AWA2-C), using ImageNet-C's 15 corruptions × 5 severities. It is aimed at ZSL, but the corruption code is reusable.
- The HF dataset `tanganke/stanford_cars` ships seven corrupted copies of the 8,041-image test set: contrast, gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate and spatter. **It has no generation documentation or severity levels, so it is not a citable benchmark. Generating our own Stanford-Cars-C with the `imagecorruptions` code is safer.**
- I found no peer-reviewed "Stanford-Cars-C" benchmark.

**Low resolution on Stanford Cars** (all crop to the bbox and downsample to 50×50):
- Peng, Hoffman, Yu, Saenko, "Fine-to-coarse knowledge transfer for low-res image classification", ICIP 2016 (arXiv 1605.06695) [P]. AlexNet trained on high-res and tested on low-res scores **1.7%**. Trained and tested at low res: 50.4%. Super-resolution baseline NBSRF: 50.8%. Mixed high+low training: 59.3% on low-res and 72.9% on high-res. Staged training (high-res first, then low-res): **59.5%**. High-res oracle: 80.3%. **The resolution-mismatch collapse is the single most important robustness result for our design.**
- Cai, Chen, Qian, Kämäräinen, "Convolutional low-resolution fine-grained classification" (arXiv 1703.05393; published in Pattern Recognition Letters per later citations (unverified)) [P]. RACNN puts SR layers in front of the classifier. On 50×50 Cars it reports 63.8% vs 50.4% for AlexNet. In the frozen-feature setting: GoogLeNet 46.85% → 50.76%. On CUB the gain is larger at 25×25 (+13 points) than at 100×100 (under 2 points).
- More recent: "Vision Mamba Distillation for low-resolution FGVC" (arXiv 2411.17980), and DADKD (reported as 88.19% on Cars (unverified), venue unverified).

**Occlusion and regional dropout.**
- Cutout: DeVries & Taylor, arXiv 1708.04552 (2017). Random square masking; 2.56% error on CIFAR-10 and 15.20% on CIFAR-100.
- Random Erasing: Zhong et al., arXiv 1708.04896, published at AAAI 2020. Erases a random rectangle with random values; claims robustness to occlusion. The abstract gives no numbers.
- Hide-and-Seek: Singh & Lee, ICCV 2017 (1704.04232). Hides grid patches so the network uses non-discriminative parts. Aimed at weakly-supervised localisation.
- CutMix: Yun et al., ICCV 2019 oral (1905.04899). Pastes patches with area-proportional label mixing; reports improved corruption robustness and OOD detection.

These are the standard ablation arms for our occlusion requirement. Test them on synthetic occlusion at test time, for example 10/25/50% of the bbox masked with random patches.

**Viewpoint.**
- Krause 3dRR'13 (above): explicit 3D viewpoint normalisation.
- Sochor et al., BoxCars (CVPR 2016; journal extension arXiv 1703.00686). "Unpacks" the 3D bounding box to normalise viewpoint. In the CVPR version, accuracy rose from 0.772 to 0.832, a 26% error reduction, as reported by search snippets (unverified against PDF).
- Anderson et al. (below) find that unseen poses are a major failure class.

**Low light.** ExDark: Loh & Chan, arXiv 1805.11227, published in CVIU 2019. 7,363 low-light images, 12 VOC-style classes, 10 lighting conditions. Low-light images make up under 2% of VOC, ImageNet and COCO. It has no car make/model labels, so it is only useful as a qualitative low-light stress set for the "car" class. Synthetic gamma, brightness and noise darkening of Stanford Cars is more practical.

**Test-time augmentation and adaptation.**
- Shanmugam et al., "Better aggregation in test-time augmentation", ICCV 2021 (2011.11156). Even when TTA raises mean accuracy, it flips many correct predictions to wrong ones. Learned aggregation beats averaging. **Report TTA's flip counts, not just its Δaccuracy.**
- Wang et al., TENT, ICLR 2021 spotlight (2006.10726). Entropy minimisation that updates BN statistics and affine parameters online. Reported state-of-the-art on ImageNet-C. Exact numbers not checked (unverified). It needs batches of test data, so with single-image inference fall back to BN-stat adaptation only.

---

## 3. Confidence, calibration, selective prediction, OOD

**Calibration.**
- Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks", ICML 2017 (1706.04599) [P]. **They report Stanford Cars directly** with a ResNet-50 and an 8,041/4,020/4,020 train/val/test split, where the original test set is halved into validation and test. Cars ECE uncalibrated: 4.30%. After temperature scaling: 2.35%. Histogram binning: 1.74%. BBQ: 1.84%. Matrix scaling: 10.5% (it overfits with 196 classes). Cars MCE goes from 41.55% uncalibrated to 20.23% with TS. Error is 14.28%, unchanged by TS by construction. Birds (CUB) ECE: 9.19% → 1.85%. **This is our reference protocol: hold out a calibration split from train, fit T by NLL, and report ECE, MCE and reliability diagrams.**
- Minderer et al., "Revisiting the calibration of modern neural networks", NeurIPS 2021 (2106.07998). Recent non-convolutional models (ViT, MLP-Mixer) are among the best calibrated. Earlier trends, such as calibration worsening with size or under shift, are weaker for them, and architecture is a major factor.
- LeVine et al. (arXiv 2303.12748, 2023). Zero-shot CLIP is miscalibrated, and one learned temperature per CLIP model transfers across datasets and prompts.
- Oh et al., "Towards calibrated robust fine-tuning of VLMs", NeurIPS 2024 (2311.01723). Standard fine-tuning hurts ID and especially OOD ECE.

**Selective classification and reject option.**
- Geifman & El-Yaniv, "Selective classification for deep neural networks", NeurIPS 2017 (1705.08500). SR/MSP thresholding with a guaranteed risk bound (SGR). ImageNet: 2% top-5 risk at about 60% coverage, guaranteed with 99.9% probability.
- SelectiveNet (Geifman & El-Yaniv, ICML 2019; 1901.09192). Jointly trains the predictor and the selection head at a target coverage.
- Geifman, Uziel, El-Yaniv, ICLR 2019 (1805.08206). Usually cited for the AURC and E-AURC risk-coverage metrics. The fetched abstract does not mention them, so their definition there is unverified.
- **Use risk-coverage curves, AURC, and the accuracy at fixed coverage (e.g. 80/90/95%) as the threshold-analysis deliverable.**

**Hierarchical back-off.**
- Deng, Krause, Berg, Fei-Fei, "Hedging your bets", CVPR 2012. The DARTS algorithm predicts internal nodes of a label hierarchy to guarantee a target accuracy while maximising information gain. Krause is a co-author, so the cars hierarchy fits naturally: model-year → model → make → body type.
- Goren, Galil, El-Yaniv, "Hierarchical selective classification" (arXiv 2405.11533). Hierarchical risk-coverage curves and a guaranteed-accuracy algorithm. CLIP, IN-21k pretraining and distillation help.
- Conformal options: Mortier et al. (arXiv 2501.19038) restrict conformal sets to internal hierarchy nodes; den Hengst et al., "Hierarchical Conformal Classification" (arXiv 2508.13288), report that users prefer hierarchical sets over flat ones.
- The natural design is "predict year-model if conf ≥ τ₁, else model, else make, else abstain".

**OOD and open-set detection.**
- Standard post-hoc scorers: MSP (Hendrycks & Gimpel, ICLR 2017); ODIN (Liang et al., ICLR 2018); Mahalanobis (Lee et al., NeurIPS 2018); Energy (Liu et al., NeurIPS 2020, 2010.03759), with 18.03% lower average FPR95 than softmax on CIFAR-10 WRN; ViM (Wang et al., CVPR 2022, 2203.10807), with 90.91% average AUROC using BiT-S on ImageNet OOD; KNN-OOD (Sun et al., ICML 2022, 2204.06507), 24.77% FPR95 reduction vs SSD+ on ImageNet.
- Vaze, Han, Vedaldi, Zisserman, "Open-Set Recognition: a Good Closed-Set Classifier is All You Need?", ICLR 2022 oral (2110.06207). Open-set AUROC correlates strongly with closed-set accuracy, so a well-trained Maximum Logit Score (MLS) baseline matches the state of the art. They introduce the **Semantic Shift Benchmark (SSB)**: CUB, **Stanford Cars**, FGVC-Aircraft and ImageNet-21K-P, with unknown classes split by attribute-based semantic similarity. **SSB Stanford Cars split [P, Table 2]:** 98 known classes (3,948 test images); "Easy" unknowns 76 classes (3,170 images); no Medium; "Hard" unknowns 22 classes (923 images). For comparison, CUB has 100 known and 32/34/34 Easy/Medium/Hard; the repo merges Medium into Hard. The SCars result numbers I saw were all second-hand: ARPL about 96.9% accuracy and 94.8/83.6 AUROC (Easy/Hard); MLS Hard AUROC about 82.2 (all unverified; my PDF copy of Vaze et al. was truncated after Table 2). **Hard-split AUROC in the low 80s means a near-duplicate unseen car model is hard to reject even for strong models.** The official class splits are in `github.com/sgvaze/osr_closed_set_all_you_need`.

---

## 4. Ablation, evaluation methodology and failure taxonomies

- **Anderson, Gwilliam, Teuscher, Merrill, Farrell, "Facing the Hard Problems in FGVC"** (arXiv 2006.13190, 2020) [P]. Pools predictions from several state-of-the-art FGVC models. In the cross-method overlap on Cars, about 88.8% of test images were correct for all models and about 2.7% (217 images) were wrong for all, as read from their overlap figure. Models make complementary errors; ensembling helped by over 5% on CUB. Their **failure taxonomy** has five classes: Similar Class (confusable classes); Non-target Subject (multiple objects, or the target off-centre); Inadequate Representation (pose or background unseen in training); Poor Quality (occlusion, blur, bad lighting, too small); and Other. Use this taxonomy directly for our failure-case section.
- **Label noise in Stanford Cars.** The Cleanlab blog audit (May 2023, not peer-reviewed) says it found "hundreds" of label errors but gives no count. Its examples include a Jeep Grand Cherokee labelled Dodge Durango, multi-vehicle ambiguous images, and out-of-scope images. Treat a small residual error floor as label noise, and inspect top-loss test images before calling them model failures.
- **Explanations.** Grad-CAM (Selvaraju et al., ICCV 2017, arXiv 1610.02391) and Score-CAM (Wang et al., CVPRW 2020, arXiv 1910.01279, gradient-free). ELLF's part-importance analysis found tyres uninformative and grille and headlight regions discriminative. That is a sanity check we can reproduce with CAMs, together with a "does the CAM land on the background?" metric.
- **Confusion analysis.** Krause's ICPR'14 Eq. 3.3 ranks symmetric confusion pairs. Its top pairs are trim and body-style variants (Coupe vs Convertible, Regular vs Extended Cab, rebadged vans). Also aggregate the confusion matrix at the make and body-type level to separate "near-miss" errors (same make) from gross errors.

---

## What this means for our design

1. **Classical baseline.** Implement dense SIFT + Fisher vectors (GMM k=256, power+L2 norm, linear SVM). Expect roughly 51-59% without bbox (Lin et al. baselines). The engineered FGComp winner reached 82.7% (87.8% with bbox), so frame our FV as a lower bound. **Hybrid arm:** FV-pooled CNN conv features (FV-CNN reached 70-79%) and/or CNN features + SVM. **Deep arm:** a fine-tuned ImageNet/CLIP backbone. Expect 88-93% for older CNNs and more for modern ones.
2. **Robustness suite.** Build Stanford-Cars-C from ImageNet-C corruptions at 5 severities, plus resolution sweeps at 224, 112, 56 and 28 px, synthetic occlusion at 10-50% of the bbox, low-light gamma, and viewpoint bins (front, side, rear) from a pose estimator or manual tags. Report mCE-style relative degradation. **Train with multi-resolution augmentation**, since high-res-only training collapses at 50 px (1.7%). Add RandomErasing or CutMix as ablation arms.
3. **Confidence.** Hold out a calibration split. Apply temperature scaling and report ECE, MCE and reliability diagrams before and after; Guo's ResNet-50 Cars ECE of 4.3% → 2.35% is the reference point. Sweep τ on MSP, energy and maxlogit and plot risk-coverage with AURC. Implement **hierarchical back-off** (year-model → model → make → abstain) in the DARTS style.
4. **Open-set.** Optionally evaluate OOD on the SSB Stanford Cars splits with MLS, energy and KNN.
5. **Failure analysis.** Use the Anderson et al. taxonomy, plus Grad-CAM/Score-CAM, plus top confusion pairs, plus a label-noise check.
6. **TTA.** Report flip counts. Use TENT only in batch mode.

---

## Sources
- Lin, RoyChowdhury, Maji, Bilinear CNN (arXiv v1): https://arxiv.org/abs/1504.07889
- Gosselin et al. PRL 2014 (HAL, not fetchable): https://hal.archives-ouvertes.fr/hal-01056223
- Krause PhD dissertation (reprints 3dRR'13 / ICPR'14 / CVPR'15): https://svl.stanford.edu/assets/publications/pdfs/krause2016dissertation.pdf
- Krause et al. 3dRR'13: https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/html/Krause_3D_Object_Representations_2013_ICCV_paper.html
- Krause et al. ICPR'14 ELLF: https://cs.stanford.edu/groups/vision/pdf/icpr14.pdf
- Chai et al. ICCV'13: https://openaccess.thecvf.com/content_iccv_2013/html/Chai_Symbiotic_Segmentation_and_2013_ICCV_paper.html
- Gavves et al. ICCV'13: https://openaccess.thecvf.com/content_iccv_2013/html/Gavves_Fine-Grained_Categorization_by_2013_ICCV_paper.html ; IJCV: https://ivi.fnwi.uva.nl/isis/publications/2015/GavvesIJCV2015/GavvesIJCV2015.pdf
- Razavian et al.: https://arxiv.org/abs/1403.6382
- Hendrycks & Dietterich: https://arxiv.org/abs/1903.12261
- Yucel et al. CUB-C: https://arxiv.org/abs/2201.10972
- HF corrupted Stanford Cars: https://huggingface.co/datasets/tanganke/stanford_cars
- Peng et al. ICIP'16: https://arxiv.org/abs/1605.06695
- Cai et al. RACNN: https://arxiv.org/abs/1703.05393
- ViMD low-res FGVC: https://arxiv.org/abs/2411.17980
- Cutout: https://arxiv.org/abs/1708.04552 ; Random Erasing: https://arxiv.org/abs/1708.04896 ; Hide-and-Seek: https://arxiv.org/abs/1704.04232 ; CutMix: https://arxiv.org/abs/1905.04899
- BoxCars: https://arxiv.org/abs/1703.00686 ; https://openaccess.thecvf.com/content_cvpr_2016/papers/Sochor_BoxCars_3D_Boxes_CVPR_2016_paper.pdf
- ExDark: https://arxiv.org/abs/1805.11227
- TTA aggregation: https://arxiv.org/abs/2011.11156 ; TENT: https://arxiv.org/abs/2006.10726
- Guo et al.: https://arxiv.org/abs/1706.04599 ; Minderer et al.: https://arxiv.org/abs/2106.07998 ; LeVine et al.: https://arxiv.org/abs/2303.12748 ; Oh et al.: https://arxiv.org/abs/2311.01723
- Geifman & El-Yaniv 2017: https://arxiv.org/abs/1705.08500 ; SelectiveNet: https://arxiv.org/abs/1901.09192 ; Geifman et al. ICLR'19: https://arxiv.org/abs/1805.08206
- Hedging your bets (dblp): https://dblp1.uni-trier.de/rec/conf/cvpr/DengKBF12.html ; Hierarchical selective classification: https://arxiv.org/abs/2405.11533 ; Conformal hierarchical: https://arxiv.org/abs/2501.19038 , https://arxiv.org/abs/2508.13288
- Energy OOD: https://arxiv.org/abs/2010.03759 ; ViM: https://arxiv.org/abs/2203.10807 ; KNN-OOD: https://arxiv.org/abs/2204.06507
- Vaze et al. OSR / SSB: https://arxiv.org/abs/2110.06207 ; https://github.com/sgvaze/osr_closed_set_all_you_need ; secondary SSB numbers: https://arxiv.org/abs/2404.19527
- Anderson et al. Hard Problems in FGVC: https://arxiv.org/abs/2006.13190
- Cleanlab Stanford Cars audit: https://cleanlab.ai/blog/csa/csa-3/
- Grad-CAM: https://arxiv.org/abs/1610.02391 ; Score-CAM: https://arxiv.org/abs/1910.01279
