# Literature Survey: Fine-Grained Vehicle Make, Model and Year Recognition on Stanford Cars

*Project P5: detection, classical CV and deep-learning hybrid, confidence handling and model optimisation.*
*Consolidated from five verified research reports (docs/research/01–05), compiled 2026-11-06.*

---

## How to read this survey

Every number in this document was taken from one of the five team research reports. No figure was added during consolidation. Each number keeps the verification marker that its report assigned:

- **[P]** (primary): the number was read in the original paper's own table or text, or in official documentation.
- **[S]** (secondary): the number was read only in another paper's comparison table or in a survey; where known, the citing source is named, for example [S: Wei].
- **(unverified)**: the number could not be confirmed from any paper table. It is kept only for orientation and must not be quoted as fact.
- **(abstract only)**: the claim comes from an abstract, not the full text.

Where two reports disagree on a number, both values are given with their sources. "Res" means input resolution. "Extra data" means pretraining beyond ImageNet-1k/21k. In-text citations [n] refer to the numbered reference list at the end. Papers withdrawn by their authors are marked **WITHDRAWN** and are not used as evidence.

---

## 1. Introduction and scope

### 1.1 Problem statement

Project P5 asks for a system that localises a vehicle in an image and predicts its make, model and year. It must also expose intermediate computer-vision steps, compare classical, deep and hybrid methods, report calibrated confidence, behave sensibly on unseen or degraded input, and deploy an optimised model. This survey reviews the literature behind each requirement and records where the literature is silent, since those gaps are where the project can contribute something measurable.

The task is **fine-grained visual categorisation (FGVC)**. Wei et al. [18] divide FGVC recognition into localisation–classification subnetworks, end-to-end feature encoding, and methods using external information. Vehicle make and model recognition (VMMR) is its application-specific branch, surveyed by Ni & Huttunen [19], Gayen et al. [20] (abstract only), Boukerche & Ma [21] (abstract only) and Boukerche et al. [22] (bibliographic only). Boukerche & Ma split the vehicle pipeline into detection, make/model recognition and re-identification, which matches this project's detect-then-classify design.

### 1.2 Why fine-grained vehicle recognition is hard

1. **Small inter-class differences.** Vehicles of different types can look more alike than vehicles of the same type when colour and viewpoint match [94], and year- and trim-level differences are extremely subtle [7]. ELLF's most-confused pairs (A5 vs S5 Coupe, V8 Vantage Coupe vs Convertible, Express vs Savana, Silverado Regular vs Extended Cab) differ only in tiny details [35].
2. **Viewpoint.** CompCars per-view models range from 42.8% (side) to 59.8% (rear-side) at model level [3] [P].
3. **Year and generation.** Successive years often look identical (Section 5.4).
4. **Rebadging and shared platforms** (Section 2.6).
5. **Imaging conditions.** A network trained at high resolution scores 1.7% on 50×50 px Stanford Cars crops [99] [P].
6. **Temporal drift and open sets.** Plain classification is bound to fail on unseen models [19], and accuracy declines across years [9].

Rigid objects such as cars are easier than non-rigid ones, and part-based reasoning helps most on the latter [18]. TransFG's authors attribute the weaker performance of plain ViTs on Cars to the regular, simple shapes of cars, which favour high-resolution CNNs [58].

### 1.3 The Stanford Cars dataset

Stanford Cars was introduced by Krause, Stark, Deng and Fei-Fei at the 3dRR workshop at ICCV 2013 as "car-197" and released with 196 classes [1]. It contains 16,185 images split into 8,144 training and 8,041 test images, with one 2D bounding box per image and a class label of the form *Make Model [Body] Year* (for example *Audi S4 Sedan 2012*). It provides no viewpoint or part annotations [1]. In 2013 it also served as the cars domain of the FGComp challenge, whose Track 1 supplied bounding boxes at test time and Track 2 did not [2].

**How the "year" labels were defined.** This is the most important fact for interpreting year predictions. Krause et al. reasoned that a car model's appearance does not change from year to year. They crawled every car type produced since 1990, de-duplicated example images with perceptual hashing down to a limited set of visually distinct classes, and sampled 197 of those [1] [P]. The year in a Stanford Cars label is therefore a *representative year for a visually distinct appearance*, not a precise model year. Year errors should not be scored as "off by N years", and a prediction of "2012" means "the appearance this dataset associated with 2012".

The dataset covers only models released before 2013 [8], has no official validation split, contains some label noise [16], overlaps with some web-scale pretraining corpora [79, 77], and is listed by Wei et al. among the classic benchmarks that are largely saturated [18] (Section 2).

### 1.4 Scope

Sections 2–11 cover datasets, classical methods, deep FGVC, vehicle-specific recognition, detection, hybrid integration, robustness, confidence, optimisation and evaluation (2004–2026). Section 12 lists research gaps and Section 13 the design implications.

---

## 2. Datasets for vehicle recognition

### 2.1 Comparison of datasets

**Table 2.1. Vehicle recognition datasets.** Counts are primary unless marked.

| Dataset | Size / classes | Labels | Year granularity | Availability |
|---|---|---|---|---|
| **Stanford Cars** [1] (3dRR@ICCVW 2013) | 16,185 images; 196 classes (197 in paper); 8,144 / 8,041 train/test | *Make Model [Body] Year* string; one 2D box | Representative year per visually distinct appearance | Original URL dead; Kaggle and Hugging Face mirrors (Section 2.2) |
| **CompCars** [3] (CVPR 2015) | Web part: 136,727 whole-car + 27,618 part images, 163 makes, 1,716 models. Surveillance part: 44,481 front views, 281 models. 208,826 images in total | Web: make → model → year hierarchy, 5 viewpoints, 8 parts, 5 attributes. Surveillance: box, model, colour | Hierarchy has years, but classification baselines **merge all years of a model into one class** | On request (CUHK); research-only licence |
| **BoxCars21k** [4] (CVPR 2016) | 21,250 vehicles, 63,750 images, 27 makes, 148 classes [S] | 3D boxes, surveillance views | make+model+submodel+model year | medusa.fit.vutbr.cz |
| **BoxCars116k** [5] (T-ITS 2018/19) | 27,496 vehicles, 116,286 images, 45 makes, 693 classes, 137 cameras | 3D box, foreground mask, tracks | "Hard" split keeps model years (107 classes); "medium" merges them | Same server; code on GitHub |
| **VMMRdb** [6] (CVPRW 2017) | 291,752 images, 9,170 classes, model years 1950–2016 (README) | make, model, year; unaligned crowd-sourced photos; experiments use the VMMRdb-3036 subset | Per model year | Dropbox ZIP; MIT-licensed repo |
| **Gebru et al. "visual census"** [7] (AAAI 2017) | 712,430 images, 2,657 classes; nearly all car types produced after 1990 | make, model, year, trim, grouped into sets of indistinguishable classes | Year and trim | Not publicly released (as far as found) |
| **Car-1000** [8] (FGVC11@CVPR 2024; arXiv v2 2026) | 140,267 images, 1,000 models, 166 makers, 1960s–2020s; 60/20/20 split | 3-tier hierarchy: 7 types → 21 sub-types → 1,000 models | No year/generation labels described | GitHub |
| **CaMiT** [9] (NeurIPS 2025 D&B) | 787K labelled images (190 models, 2007–2023) + 5.1M unlabelled (2005–2023) | Time-stamped images for studying temporal drift | Image timestamp | Released by authors (CEA List) |
| **VehicleID** [10] (CVPR 2016) | 221,763 images of 26,267 vehicles; model labels on a subset (47,558 + 42,638 images) | Identity; partial model labels | — | On request |
| **VeRi-776** [10] (2016) | About 50k images (49,357 per one source), 776 vehicles, 20 cameras | Identity, colour, type, brand | — | On request |

Smaller historical sets, with counts from Ni & Huttunen's Table 1 [19] [S]: Petrovic–Cootes 2004 (1,132 images, 77 classes), Clady 2008 (1,121 images, 50 classes), car-types (Stark 2012; 1,904 images, 14 classes), BMW-10 (512 images, 10 classes) and FG3DCar (Lin 2014; 300 images, 30 classes).

Two observations follow. First, **no large public dataset labels model year at the granularity Stanford Cars implies**: CompCars merges years in its baselines, Car-1000 has no year labels, and BoxCars116k's year-level split is small. Second, the re-identification sets label *identity*, not taxonomy, so they are useful only for metric-learning pretraining or surveillance-domain images.

### 2.2 Access and mirrors

The original host (ai.stanford.edu/~jkrause/cars) returns 404 [12], and torchvision disabled `StanfordCars(download=True)` from v0.18 [11].

- The common Kaggle mirror `jessicali9530/stanford-cars-dataset` has nested duplicate folders, and its devkit **lacks test labels**; users take `cars_test_annos_withlabels.mat` from a Kaggle notebook.
- `rickyyyyyyy/torchvision-stanford-cars` (2024) and `emanuelriquelmem/stanford-cars-pytorch` (2024) offer a torchvision-compatible layout.
- `tanganke/stanford_cars` on Hugging Face has 8,144/8,041 images plus seven corrupted test copies (Section 8.1); its card gives no source or licence, and its labels were not checked [14].
- Class names in `cars_meta` order are mirrored in clip-benchmark `wds_cars` [13]; index 1 is *AM General Hummer SUV 2000*, and torchvision uses the same order 0-based.

### 2.3 Splits

The official split is roughly 50/50 per class, and **there is no official validation split**. Practitioners often tune on the test set, as one fast.ai thread states explicitly [90], so many reported test accuracies are optimistically biased. Other protocols redefine the split and are not comparable with 196-way results: Guo et al. use 8,041 / 4,020 / 4,020 by halving the test set [140] [P]; the VTAB-style PEFT protocol uses 7,329/815 at 224 px [83]; and the SSB uses 98 known classes [154] (Section 9.4).

### 2.4 Label noise

A Cleanlab audit (vendor blog, May 2023, not peer-reviewed) reports "hundreds" of label errors without an exact count [16]. Its examples are a Jeep Grand Cherokee labelled as a Dodge Durango, wrong models within the correct make, out-of-scope images and two-car images. Part of the residual error at 94–97% top-1 may therefore be label noise, and top-loss images should be inspected before being called model failures. `stanford_cars_refined` relabels Cars by colour into 1,288 classes and is not a label audit [15].

### 2.5 Contamination and duplication

- **DINOv2's LVD-142M was built by retrieval using the Stanford Cars *train* set as queries**, retrieving about 1M images; near-duplicates of benchmark test/val images were removed [79] [P]. The pretraining data are therefore deliberately Cars-like. Similar curation for DINOv3 is likely but unverified.
- **CLIP-family corpora (LAION, WebLI, MetaCLIP) are not de-duplicated against the Cars test set.** CLIP's overlap study over 35 datasets found a median overlap of 2.2%, a mean of 3.2%, and accuracy shifts of at most 0.6% [77] [P]; the Cars-specific value is only in a figure. OpenCLIP found about 1% pHash duplicates against LAION-400M for most datasets, but does not list Cars [81]. EVA-02-CLIP-E's 94.6% *zero-shot* accuracy [78] [P] suggests heavy exposure to captioned car images. That the Cars test images came from web search is unverified.
- **ImageNet overlap is negligible**: 10 train (0.12%) and 14 test (0.17%) images [69] [P].
- **Train/test near-duplicates** within Stanford Cars have not been audited; Barz & Denzler's method (CNN-feature nearest neighbours plus manual check) applies cheaply [17].

### 2.6 Class-structure analysis: year twins, rebadges and platform twins

Report 02 computed the following groupings with a script over the 196 class names [13]. **The generation and platform relationships are the report author's automotive domain knowledge, not taken from papers, and should be checked.**

**Table 2.2. The seven year-twin pairs.** Same make and model string, differing only in year; 1-based indices.

| Pair (index) | Generation relationship (domain knowledge) | Expected difficulty |
|---|---|---|
| Audi S4 Sedan 2012 (23) vs 2007 (24) | B8 vs B7 | moderate |
| Bentley Continental GT Coupe 2012 (42) vs 2007 (43) | 2nd vs 1st generation; evolutionary styling | hard |
| Dodge Caliber Wagon 2012 (83) vs 2007 (84) | **same generation (2007–12)**, mild refresh only | near label noise; hardest |
| Dodge Durango SUV 2012 (94) vs 2007 (95) | 3rd vs 2nd generation | easy–moderate |
| Ford F-150 Regular Cab 2012 (113) vs 2007 (114) | 12th vs 11th generation | moderate |
| Honda Odyssey Minivan 2012 (126) vs 2007 (127) | 4th vs 3rd generation | moderate |
| Volkswagen Golf Hatchback 2012 (190) vs 1991 (191) | Mk6 vs Mk2 | easy |

**Table 2.3. Other twin groups** (domain knowledge; 1-based indices).

| Twin type | Groups |
|---|---|
| Near-year, model string differs | Silverado 1500 Classic Ext. Cab 2007 (70) vs Silverado 1500 Ext. Cab 2012 (74); Charger SRT-8 2009 (97) vs Charger Sedan 2012 (96); Malibu 2007 (73) vs Malibu Hybrid 2010 (67); BMW 6 Series Conv. 2007 (31) vs M6 Conv. 2010 (36), same E64 body; M5 2010 (35) vs ActiveHybrid 5 2012 (26); Corvette Ron Fellows Z06 2007 (57) vs Convertible/ZR1 2012 (55, 56), all C6; Ram 3500 Quad Cab 2009 (87) vs Crew Cab 2010 (86); Dakota Club Cab 2007 (91) vs Crew Cab 2010 (90); Elantra Sedan 2007 (135) vs Elantra Touring 2012 (139); Bentley Continental family (39, 42, 43, 44) |
| Rebadge, near-identical body | Express Van (71) and Express Cargo Van (64) ↔ GMC Savana (119); Dodge Sprinter (88) ↔ Mercedes-Benz Sprinter (166), same NCV3 body; Tahoe Hybrid (62) ↔ Yukon Hybrid (120); Chrysler Aspen (76) ↔ Dodge Durango 2007 (95); Buick Rainier (48) ↔ TrailBlazer SS (68) ↔ Isuzu Ascender (143), GMT360; Town and Country (78) ↔ Ram C/V (174) |
| Platform sibling, distinct styling | GM Lambda: Traverse (58), Acadia (121), Enclave (50); Escalade EXT (53) ↔ Avalanche (65); Canyon (122) ↔ H3T (124); Patriot (145), Compass (149), Caliber (83, 84); Chrysler LX: Charger (96, 97), 300 SRT-8 (79), Magnum (92), Challenger (93) |
| Trim twin | A5 / S5 Coupe (13, 22); TT / TTS / TT RS (19, 14, 25); Sonata / Sonata Hybrid (138, 134); Silverado cluster (54, 69, 70, 74, 75) |
| Body-style twin (same make, model, year) | V8 Vantage (8/9), Virage (10/11), Veyron (45/46), 458 (103/104), Spyker C8 (179/180), S5 (21/22), BMW 1 Series (27/28), 3 Series sedan/wagon (29/30), Accord (128/129), SX4 (183/184), Audi 100 (17/18), Phantom (175/177) |
| Twin outside the label set (open-set risk) | Geo Metro ≈ Suzuki Swift; Eagle Talon ≈ Mitsubishi Eclipse; Plymouth Neon ≈ Dodge Neon; Mazda Tribute ≈ Ford Escape. Photos of the absent partner will be confidently mislabelled |

The list is consistent with empirical confusions: ELLF's most-confused pairs include the Express/Savana rebadge, the A5/S5 trim twin, a V8 Vantage body-style twin and a Silverado cab variant [35] [P], and Cleanlab's Grand Cherokee/Durango example is a cross-make near-twin [16].

### 2.7 The 49 makes and label-parsing quirks

Report 02 counted 49 makes, with the number of classes per make in parentheses: AM General (1), Acura (6), Aston Martin (4), Audi (14), BMW (13), Bentley (6), Bugatti (2), Buick (4), Cadillac (3), Chevrolet (22), Chrysler (6), Daewoo (1), Dodge (15), Eagle (1), FIAT (2), Ferrari (4), Fisker (1), Ford (12), GMC (5), Geo (1), HUMMER (2), Honda (4), Hyundai (11), Infiniti (2), Isuzu (1), Jaguar (1), Jeep (5), Lamborghini (4), Land Rover (2), Lincoln (1), MINI (1), Maybach (1), Mazda (1), McLaren (1), Mercedes-Benz (6), Mitsubishi (1), Nissan (4), Plymouth (1), Porsche (1), Ram (1), Rolls-Royce (3), Scion (1), Spyker (2), Suzuki (4), Tesla (1), Toyota (4), Volkswagen (3), Volvo (3), smart (1).

**Parsing pitfalls**, which affect make-level scoring and hierarchical back-off (Section 9.3):

- "AM General", "Aston Martin" and "Land Rover" contain spaces, and "Mercedes-Benz" and "Rolls-Royce" are hyphenated; splitting on the first space wrongly yields 52 makes.
- The year is always the last token, but the body type is sometimes missing (for example "Acura TL Type-S 2008", "Chevrolet Corvette ZR1 2012", "Buick Regal GS 2012", "Jaguar XK XKR 2012", "FIAT 500 Abarth 2012", "Chrysler 300 SRT-8 2010").
- "Bentley Continental Supersports Conv. Convertible 2012" has an abbreviation and a redundant word, and casing is inconsistent ("FIAT", "MINI", "HUMMER", "smart").
- Brands are split: *AM General Hummer SUV 2000* sits under AM General while the H2 and H3T sit under "HUMMER"; *Ram C/V* is under "Ram" while *Dodge Ram Pickup 3500* is under Dodge.

---

## 3. Classical approaches

### 3.1 Hand-crafted features and encodings on Stanford Cars

Before deep learning, the strongest Stanford Cars results came from dense local descriptors (SIFT and colour) encoded into bag-of-visual-words, locality-constrained linear coding (LLC) with spatial pyramid matching (SPM), or Fisher vectors (FV). Pose-normalised pooling was the main alternative.

**Table 3.1. Classical and early-deep results on Stanford Cars** (196/197 classes).

| Method | Paper | Cars accuracy | Marker |
|---|---|---|---|
| BubbleBank (BB), 2D | Deng et al. CVPR'13, re-run by Krause [1] | 63.6% | [P] |
| LLC + SPM (SIFT) | Wang et al. CVPR'10, re-run by Krause [1] | 69.5% | [P] |
| SPM-3D-L | Krause et al. 3dRR'13 [1] | 65.6% | [P] |
| BB-3D-G (3D-lifted BubbleBank) | Krause et al. 3dRR'13 [1] | 67.6% | [P] |
| Stack of SPM, BB, SPM-3D-L, BB-3D-G | Krause et al. 3dRR'13 [2] | 75.5% | [P] |
| SIFT Fisher vector, 256-word GMM (vanilla baseline) | Lin et al. [39] | **51.2%** in arXiv v1 [P, report 04]; **59.2%** per report 03 (B-CNN TPAMI Table 2) and per report 04 for the ICCV version (unverified) | see text |
| Symbiotic segmentation + parts (FGComp entry) | Chai, Lempitsky, Zisserman ICCV'13 [36] | 78.0% | [S: Lin et al.] |
| Inria-Xerox FV ensemble, no test box (FGComp winner) | Gosselin et al. PRL 2014 [34] | **82.7%** | [S: Lin et al.] |
| Same, with test box (Track 1) | Gosselin et al. [34] | **87.8%** per report 04 (whose source text also gives 87.9%); **87.9%** per report 03 (B-CNN Table 1) | [S] |
| Different FV setup reported elsewhere | source not named in report 02 | 83.7% | [S] |
| CNN from scratch, no ImageNet | Krause et al. ICPR'14 [35] | 70.5% | [P] |
| CNN-SPM, small / large grid | Krause et al. ICPR'14 [35] | 67.9% / 69.3% | [P] |
| **ELLF** (Ensemble of Localized Learned Features) | Krause et al. ICPR'14 [35] | **73.9%** | [P] |
| FC-CNN, off-the-shelf (no fine-tuning), M-Net / VGG-D | Lin et al. [39] | 33.5% / 32.0% (fine-tuned: 50.9% / 67.7%) | [P] |
| FV-CNN (D), fine-tuned | Lin et al. [39] | 79.2% | [P] |

**Discrepancies.**

- *Vanilla SIFT-FV baseline.* Report 04 read **51.2%** in arXiv v1 of the bilinear-CNN paper [P] and could not verify 59.2% for the ICCV version. Report 03 cites **59.2%** from the TPAMI Table 2. Both are plausibly correct for different versions, so this survey quotes 51–59% and names the version when a single value is needed.
- *Gosselin et al. with boxes.* Report 04 gives 87.8% (its source text also gives 87.9%), report 03 gives 87.9%, and report 02's unverified recollection is "about 82.7–87.8%". **No report could open Gosselin's own PDF**; all values come from Lin et al.'s quotation of the FGComp leaderboard.

**Gosselin et al.** late-fuse two deliberately complementary FV systems built from dense multi-scale SIFT and colour descriptors, with descriptor filtering and spatial-coordinate coding [34]. Relative to a vanilla FV they use a 1024-word GMM, 1×1 + 3×1 spatial pyramids and several SIFT variants [39]. Adding the box lifts the ensemble by about 5 points on cars but under 1 point on aircraft, so localisation matters more for cars [39].

**Krause et al., 3dRR'13** fit 3D CAD models, rectify patches and pool descriptors in azimuth–elevation space [1]. On car-197, BB-3D-G beat 2D BB (67.6 vs 63.6%) but plain LLC-SPM (69.5%) beat both; the authors blame limited CAD coverage of body types such as pickups. On small sets the lifting helped more: BB-3D-G reached 94.5% on car-types and 66.1% on BMW-10 (BB 58.7%, HOG 28.3%) [P]. Horizontal flips added 8.6–13.3 points on BMW-10 [2]. Geometric normalisation therefore helps on small data but loses its edge as data grow.

**ELLF (ICPR'14)** discovers parts without supervision (GrabCut segmentation plus alignment), pools CNN features per part and uses a linear SVM [35]. Its part detectors misfire when GrabCut fails and under 180° (front vs rear) rotations, and its part-importance analysis found tyres uninformative and grilles and headlights discriminative.

**Related classical FGVC without Cars results.** Gavves et al.'s alignment plus FV was evaluated only on CUB-2011 and Dogs [37] (60.1% / 60.3% per a third-party summary, unverified). POOF and DPD need part keypoints that Cars lacks (CUB 56.8% / 51.0% [S: Razavian]). Razavian et al.'s off-the-shelf OverFeat-SVM [38] reports CUB at 53.3% / 61.8% (with augmentation) [P]. Its Cars analogue, Lin et al.'s un-fine-tuned FC-CNN at 32–34%, shows that **generic frozen ImageNet features transfer poorly to cars**, with a larger domain gap than for birds.

### 3.2 Vehicle-specific classical VMMR

Classical VMMR typically assumed a constrained view (usually frontal or rear) and used a licence plate or symmetry to locate a reference region.

**Table 3.2. Classical VMMR methods** (mostly not on Stanford Cars).

| Work | Method | Data | Result | Marker |
|---|---|---|---|---|
| Petrovic & Cootes, BMVC 2004 [26] | Frontal patch located relative to the licence plate; raw, Harris and gradient structure features (gradient best); nearest neighbour | >1,000 images, 77 classes | >93% recognition; <5.6% verification EER | [P, abstract] |
| Clady et al. 2008 | Oriented-contour-point models, frontal | 50 classes | 93.1% | [S: Ni] |
| Psyllos et al., T-ITS 2010 [28] | Logo recognition with enhanced SIFT matching | 1,200 logos, 10 makes | 91% vs 88% (standard SIFT); another paper cites 97% | (unverified) |
| Pearce & Pears, AVSS 2011 [27] | Harris-corner / LNHS frontal descriptor, Naive Bayes | 262 images, 74 classes | about 96% | (unverified) |
| Stark et al., BMVC 2012 [32] | structDPM (multi-class DPM) + part bank | car-types, 14 classes | 93.5% | [P, quoted in Krause 2013] |
| Hsieh, Chen & Chen, T-ITS 2014 [29] | Symmetrical SURF localises the vehicle without background subtraction; grid-wise HOG+SURF SVM ensemble; sparse-representation extension in *Pattern Recognition* 2015 | — | not verified | (unverified) |
| Llorca et al., ITSC 2014 [30] | Rear model and trim emblems; HOG + linear SVM; plate-assisted | 8 makes, 28 models, **52 classes counting generations** | 93.75% | [S: search snippet of PDF] |
| Lin, Morariu, Hsu & Davis, ECCV 2014 [33] | DPM parts → landmark regression → 3D model fit → SVM; predicted class refines the fit | FG3DCar, 300 images, 30 classes | 90.0% | [S] |
| Boonsim & Prakoonwit, PAA 2017 [31] | Night-time rear taillight and plate geometry; majority vote of SVM, decision tree and kNN | night images | about 93% | [S] |

Two of these are especially relevant here. Llorca et al. is one of the few classical works that explicitly separates *generations*, using rear emblems [30]. Hsieh et al.'s symmetrical SURF is the only symmetry-prior VMMR method found; no deep counterpart was found [29].

---

## 4. Deep fine-grained classification

Following Wei et al. [18], deep FGVC methods are grouped here into (i) part and attention localisation, (ii) high-order feature encoding, (iii) training strategies and regularisers, (iv) Vision-Transformer methods, (v) strong-backbone transfer without FGVC modules, (vi) foundation models used zero-shot or with linear probes, and (vii) parameter-efficient fine-tuning (PEFT). Krause et al.'s CVPR 2015 annotation-free part method is treated with vehicle-specific work in Section 5.

### 4.1 Part / attention localisation, high-order pooling and training strategies (CNN era)

**Part and attention methods** find and zoom in on discriminative regions: recurrent attention zoom with an inter-scale ranking loss (RA-CNN [41]), channel-grouped part attention (MA-CNN), self-supervised navigator–teacher–scrutiniser proposals (NTS-Net [43]), attention-guided cropping and dropping (WS-DAN [44]), sparse sampling around class peaks (S3N [46]), separate raw/object/part branches (MMAL-Net [52]), counterfactual attention (CAL [53]), context-aware attentional pooling (CAP [54]), region graphs (SR-GNN [55]) and layers acting as experts that exchange attention regions (CMAL-Net [57]).

**High-order pooling** captures feature interactions: the outer product of two CNN streams (B-CNN [39]), its Tensor-Sketch approximation (CBP [40]), polynomial kernels (Kernel Pooling), Newton–Schulz covariance square roots (iSQRT-COV [42]), cross-layer pooling (HBP), bilinear transformation (DBTNet), and channel discriminality/diversity losses (MC-Loss [48]).

**Training strategies** change the learning signal: region shuffling and layout recovery (DCL [45]), cross-category and cross-layer regularisation (Cross-X [47]), pairwise interaction between confusable images (API-Net [49]), channel interaction (CIN [50]), progressive multi-granularity jigsaw training (PMG [51]), and progressive multi-task anti-noise learning distilled back into a plain backbone at no extra inference cost (PMAL/PMD [56]).

**Table 4.1. CNN-era FGVC methods on Stanford Cars** (top-1 %). No extra data unless stated.

| Method (venue) | Family | Backbone @ res | Cars top-1 | Marker |
|---|---|---|---|---|
| B-CNN (ICCV'15 / TPAMI) [39] | pooling | VGG-M+VGG-D @448 | 88.2 (ICCV/arXiv v1, no boxes); **91.3** (TPAMI). VGG-D alone 90.6 | [P] |
| Compact Bilinear (CVPR'16) [40] | pooling | VGG-16 @448 | not in original paper; 91.2 (VGG-D) in iSQRT-COV table | [S: iSQRT] |
| Kernel Pooling (CVPR'17) | pooling | VGG-16 | 92.4 | [S: DCL, iSQRT] |
| RA-CNN (CVPR'17) [41] | part/attention | VGG-19 @448 | 92.5 | [P] |
| MA-CNN (ICCV'17) | part/attention | VGG-19 @448 | 92.8 | [S: DCL, NTS-Net; Wei] |
| iSQRT-COV (CVPR'18) [42] | pooling | ResNet-50 / 101 @448 | 92.8 / 93.3 | [P] |
| HBP (ECCV'18) | pooling | VGG-16 | 93.7 | [S: MMAL, CIN] |
| DFL-CNN (CVPR'18) | part/attention | VGG-16 / ResNet-50 | 93.8 / 93.1 | [S: MC-Loss, PMG; Wei] |
| NTS-Net (ECCV'18) [43] | part/attention | ResNet-50 @448 | 93.9 | [P] |
| MC-Loss (TIP'20) [48] | pooling/loss | ResNet-50 / B-CNN @448 | 93.7 / 94.4 | [P] |
| CIN (AAAI'20) [50] | training | ResNet-50 / 101 | 94.1 / 94.5 | [P] |
| WS-DAN (arXiv'19) [44] | part/attention | Inception-v3 @448 (res unverified) | 94.5 | [P] |
| DCL (CVPR'19) [45] | training | ResNet-50 @448 | 94.5 | [P] |
| DBTNet (NeurIPS'19) | pooling | ResNet-101 | 94.5 | [S: TransFG] |
| Cross-X (ICCV'19) [47] | training | ResNet-50 / SENet-50 | 94.6 / 94.5 | [P] |
| S3N (ICCV'19) [46] | part/attention | ResNet-50 @448 | 94.7 | [P] |
| DF-GMM | — | ResNet-50 @448 | 94.8 | [S: Wei] |
| API-Net (AAAI'20) [49] | training | R-50 / R-101 / DenseNet-161 @448 | 94.8 / 94.9 / 95.3 | [P] |
| Sun et al. (AAAI'20) | — | ResNet-50 @448 | 94.9 | [S: Wei] |
| PMG (ECCV'20) [51] | training | ResNet-50 @448 (550 resize) | 95.0 (95.1 combined; Wei lists 95.1) | [P] |
| MMAL-Net (MMM'21) [52] | part/attention | ResNet-50 @448 | 95.0 | [P] |
| PMD (ResNet-50) [56] | training + distillation | ResNet-50 @448 | 95.4 (PMAL) / 95.3 (PMD) | [P] |
| CAL (ICCV'21) [53] | part/attention | ResNet-101 @448 | 95.5 | [P] |
| CAP (AAAI'21) [54] | part/attention | Xception @224 | 95.7 | [P] |
| SR-GNN (TIP'22) [55] | part/attention | Xception @224 | 96.1 | [P] |
| CMAL-Net (PR'23) [57] | part/attention | TResNet-L (@368 per PMD) | 97.1 | [S: PMD Table V] |
| PMAL / PMD (arXiv 2401.14336; venue unverified) [56] | training + distillation | TResNet-L @368 | **97.3 / 97.3** | [P] |
| MaxEnt | regulariser | plain VGG-16 @224 | 83.9 | [S: Wei] |

Wei et al. observe that FGVC-specialised methods on a ResNet-50 at 448 cluster at **93.8–95.3%** [18]. They also note that localisation–classification pipelines often need alternating or multi-stage training, which complicates tuning.

### 4.2 Vision-Transformer FGVC

**Table 4.2. ViT-based FGVC methods.**

| Method (venue) | Key idea | Backbone @ res | Cars top-1 | Marker |
|---|---|---|---|---|
| TransFG (AAAI'22) [58] | Part selection via attention rollout; contrastive loss; overlapping patches | ViT-B/16 (IN-21k) @448 | 94.8; AF-Trans's re-run gives 94.1 | [P] |
| AF-Trans (ICASSP'22) [61] | Selective attention collection; multi-scale fusion | ViT-B/16 | 95.0 | [P] |
| DCAL (CVPR'22) [63] | Global–local and pairwise cross-attention | R50-ViT-B | 95.3 | (unverified) |
| MetaFormer (arXiv'22) [64] | Hybrid conv+attention; optional meta-information | MetaFormer-1 @384 | 94.9 (IN-1k) / 95.0 (IN-21k) / 95.0 (iNat21) | [P] |
| CAMF (venue unverified) | Swin multi-feature fusion | Swin-B @224 | 95.3 | [S: SR-GNN] |
| FFVT (BMVC'21) [59]; RAMS-Trans (ACM MM'21) [60]; PIM / HERBS [65, 66] | Layer token fusion; recurrent multi-scale crops; background suppression | ViT-B/16; Swin-L @384 | **no Cars result** (checked) | — |
| SIM-Trans (ACM MM'22) [62]; IELT (TMM'23) [67] | Structure-information modules; internal layer ensemble | ViT-B/16 | no Cars result per aggregators | (unverified) |
| ACC-ViT (2024) | — | ViT-B | primary paper not found | (unverified) |
| SCOPE (arXiv 2508.06959) [68] | — | — | **WITHDRAWN** by authors (inconsistent experiment records); not cited as evidence | — |

**Observation.** Plain ViT-B/16 FGVC methods reach about 94–95% on Cars. That is below the best 448-px CNNs and below TResNet-L. TransFG's authors attribute this to the regular, simple shapes of cars, which favour high-resolution CNNs [58].

### 4.3 Strong-backbone transfer and reproducible practitioner baselines

Since about 2021, Cars accuracy gains have come more from stronger backbones and pretraining than from FGVC-specific modules.

**Table 4.3. Transfer baselines without FGVC modules.**

| Source | Backbone @ res | Pretraining | Cars top-1 | Marker |
|---|---|---|---|---|
| Kornblith et al. CVPR'19 [69] | Inception-v4 @299 / @448; from scratch @299 | IN-1k | 93.3 / 93.4; 92.7 | [P] |
| EfficientNet, ICML'19 [70] | B3 / B7 (@600); Inception-v4 reference | IN-1k | 93.6 / 94.7; 93.4 | [P] |
| GPipe, NeurIPS'19 | AmoebaNet-B | IN-1k | 94.6 | [S: MetaFormer, CAP] |
| DeiT, ICML'21 [74] | DeiT-B @224 / ↑384 / distilled ↑384 | IN-1k | 92.1 / 93.3 / 93.9 | [P] |
| ResNet Strikes Back [75] (Table 5) | ResNet-50, DeiT-style fine-tuning; PyTorch default / A1 / A2 / A3 pretraining | IN-1k | 92.5 / 92.7 / 92.6 / 92.5 | [P] |
| TResNet, WACV'21 [71] | TResNet-L @368, CE + triplet | IN-1k | **96.0** | [P] |
| Domain-Adaptive Transfer [72] | Inception-v3 / AmoebaNet-B on a "JFT-Car" subset | JFT (**extra data**) | 96.0 / 96.2 | [P] |
| SAM, ICLR'21 [73] | EffNet-L2 without / with SAM | JFT / NoisyStudent (**extra data**) | 95.07 / 95.96 | [P] |
| fast.ai forum [90] | EfficientNet-B3 + Mish + Ranger @299, 40 epochs, flat-cosine; **test set used as validation** | IN-1k | 93.8 (5-run mean); about 93.4 without Mish | practitioner |
| same thread [90] | EfficientNet-B7 + Mish + Ranger, 5 runs | IN-1k | 94.79 (std 0.094) | practitioner |
| jianshen92 repo [91] | ResNet-152 @299 + mixup; ResNet-50 88.0 → 89.3 with mixup | IN-1k | 92.53 | (unverified) |

BiT reports no Stanford Cars result [76].

Kornblith et al. [69] report that **Stanford Cars is an outlier where ImageNet pretraining gives little final-accuracy benefit** (92.7% from scratch vs 93.3% fine-tuned), because ImageNet has more car images than Cars (12,455 vs 8,144) but only about 10 coarse car classes. Pretraining still speeds convergence about 17×. Regularisers that slightly improve ImageNet accuracy, such as label smoothing and dropout, make penultimate features much worse for transfer, which matters for any frozen-feature or fusion design.

Practitioner sources agree that a modern ImageNet-pretrained CNN or ViT at 299–448 px with standard augmentation reaches about 92–94% without FGVC modules, and that such modules add about 1–2 points [75, 74, 70, 90]. The fast.ai author found EfficientNet unstable with one-cycle and RMSProp but stable with Ranger and a flat-then-cosine schedule [90]. Wei et al.'s resolution study (CUB, ResNet-50: 81.6 / 83.3 / 85.0 / 85.6% at 224 / 280 / 336 / 392 px) explains why 448 px became standard [18].

### 4.4 Foundation models: zero-shot and frozen linear probes

**Table 4.4. Foundation models on Stanford Cars.** All use web-scale or curated extra data.

| Model | Source | Protocol | Cars top-1 | Marker |
|---|---|---|---|---|
| Supervised IN-21k ViT-B/16 | CLIP Table 10 [77] | Linear probe | 54.7 | [P] |
| CLIP ViT-B/16 | Radford et al. [77] | Zero-shot / linear probe | 65.6 / 86.7 | [P] |
| CLIP ViT-L/14@336 | Radford et al. [77] | Zero-shot / linear probe | 78.8 / 91.5 | [P] |
| EVA-02-CLIP-L/14+ (336) | EVA-CLIP [78] | Zero-shot | 90.5 | [P] |
| OpenCLIP ViT-G/14 (LAION-2B) | EVA-CLIP Table 2 [78] | Zero-shot | 94.5 | [P] |
| EVA-02-CLIP-E/14+ | EVA-CLIP Table 2 [78] | Zero-shot | 94.6 | [P] |
| DINOv2 ViT-S / B / L / g (/14) | DINOv2 Table 8 [79] | Linear (logistic regression) | 81.6 / 88.2 / 90.1 / 91.4 | [P] |
| OpenCLIP ViT-G/14 | DINOv2 Table 8 [79] | Linear | 96.1 | [P] |
| DINOv2 g/14 (re-evaluated) | DINOv3 Table 22 [80] | Linear | 94.6 | [P] |
| DINOv3 7B/16 | DINOv3 Table 22 [80] | Linear ("Fine-S") | 95.0 | [P] |
| AIMv2 3B / EVA-CLIP 18B / AM-RADIOv2.5 g | DINOv3 Table 22 [80] | Linear | 95.6 / 95.4 / 95.4 | [P] |
| SigLIP g/16 / PE-core G/14 | DINOv3 Table 22 [80] | Linear | **96.5 / 96.5** | [P] |

Language-supervised models beat self-supervised ones on Cars. DINOv2 trails OpenCLIP by 4.7 points [79]; a controlled 2025 study reports CLIP 74.7 vs DINO 54.1 under identical data [87] (unverified, search summary only); and zero-shot CLIP beats logistic regression on ResNet-50 features by more than 20%, which its authors attribute to web text carrying make and model names [77]. These results must be read with the contamination caveats of Section 2.5.

Training-free and LVLM-based approaches (FineR, ICLR 2024; Enriched-FineR, 2025) [178] are useful for open-vocabulary discovery but remain well below supervised accuracy (numbers unverified).

### 4.5 Parameter-efficient fine-tuning

**Table 4.5. PEFT on Stanford Cars** (VTAB-style protocol: ViT-B/16 IN-21k at **224 px**, 7,329/815 split; **not comparable** with the 448-px literature).

| Method | Cars top-1 | Marker |
|---|---|---|
| Full fine-tune / linear probe | 84.5 / 51.3 | [S: SSF Table 3] |
| Adapter / BitFit | 68.6 / 79.4 | [S] |
| VPT-shallow / VPT-deep [82] | 68.7 / 83.6 | [S: SSF Table 3] |
| SSF [83] | **89.2** | [P] |
| GPS (CVPR'24) [85], V²APT [84], frequency-based PEFT [86] | not verified; V²APT reports that full fine-tuning still beats it on Cars | (unverified) |

Cars is the dataset in this benchmark where PEFT methods trail full fine-tuning most. Report 01 reads this as evidence that capturing fine detail at low resolution requires adapting the backbone.

### 4.6 Leaderboard snapshot

paperswithcode.com no longer serves leaderboards; report 01 cross-checked a mirror [88] against the papers. Without extra data, the top results are TResNet-L + PMAL/PMD at 97.3% [56] [P] and CMAL-Net at 97.1% [57] [S] (flagged "extra data" on the mirror, but listed by PMD as TResNet-L), followed by SR-GNN (96.1%) [55] and TResNet-L (96.0%) [71]. With web-scale data, frozen SigLIP g/16 or PE-core G linear probes reach 96.5% [80], Domain-Adaptive Transfer 96.2% [72], the OpenCLIP G/14 probe 96.1% [79] and EffNet-L2 + SAM 95.96% [73]. A TResNet-L + ML-Decoder entry at 96.41% appears only on the mirror (unverified). The practical ceiling is therefore about **97%**, and a frozen web-scale ViT with a linear head already reaches about 96.5%, illustrating the shift from FGVC-specific architectures to strong backbones.

---

## 5. Vehicle-specific deep recognition

### 5.1 CompCars baselines

Yang et al. established deep baselines on CompCars [3] [P]:

- **431 models, all views:** AlexNet 81.9%, Overfeat 87.9%, GoogLeNet 91.2% top-1.
- **Per-view Overfeat models (half train / half test), model level:** front 52.4%, rear 43.1%, side 42.8%, front-side 56.3%, rear-side 59.8%; all views 76.7%.
- **Same models, make level:** 71.0 / 52.1 / 50.7 / 68.0 / 65.6 / 82.9%.
- **Parts alone:** the taillight was best at 68.4%, partly because the model name is printed near it. Voting over all eight parts gives 80.8%.
- **Surveillance front views** (including rain, fog and night): GoogLeNet 98.4%, because the viewpoint is fixed.

CompCars merges all production years of a model into one class, noting that splitting them would make the task harder. This was not tested.

### 5.2 Deep VMMR on Stanford Cars and CompCars

- **Krause et al., CVPR 2015** [92] combine box-initialised co-segmentation, alignment and discriminative parts without part annotations: 92.6% with no test box and 92.8% with the GT box [P, reports 03/04; report 02 gives 92.8% with GT box, S]. Wei et al. list the 92.6% entry as CaffeNet at 224 px "using bounding boxes" [18], whereas report 03 reads it from a VGGNet table; the backbone and box-use discrepancy is unresolved.
- **Hu, Wang, Li & Shen, T-ITS 2017** [93]: spatially weighted pooling takes Stanford Cars from 92.6% to 93.1% (ResNet-101) and reaches 97.6% on CompCars [S].
- **Xiang et al. 2019**, global topology constraint network: Stanford Cars 94.3%, CompCars 98.5% [S: Ni] [19].
- **Valev et al. 2018** [94] systematically compare architectures on Stanford Cars; the best reaches 94.6%.
- **Ni & Huttunen** [19] find that a metric-learning, gallery-based approach (ResNet-50 with triplet or lifted-structure loss and nearest-neighbour lookup) beats classification by a clear margin in their experiments. They point to few-shot and multi-task learning as directions.

### 5.3 BoxCars and geometry-aware networks

BoxCars provides the strongest evidence that explicit geometry helps vehicle FGVC.

- **BoxCars (CVPR 2016)** [4] adds three inputs to the CNN: the image "unpacked" via the 3D box, a rasterised low-resolution shape, and the 3D orientation. Accuracy is reported as rising from 77.2% to 83.2%, a 26% error reduction, and verification AP from 0.378 to 0.785 (report 02; report 04 marks the accuracy figures as unverified against the PDF).
- **BoxCars116k (T-ITS 2018)** [5] [P] adds vanishing-point 3D-box estimation for uncalibrated cameras, colour augmentation and random-noise "ImageDrop", with disjoint train/test cameras. On the hard split, ResNet-152 rises from 77.68 / 86.20% (single image / track) to 83.74 / 91.71% with GT 3D boxes and 80.87 / 90.93% with estimated boxes; generic FGVC baselines are weaker (B-CNN 69.6%, Simon & Rodner 75.9%, Gao 70.6%, Sochor 84.1% [S]). Report 02 summarises the gains as up to +12 points and up to 50% error reduction.
- **3D Perspective Network** (Zeng et al., AAAI 2020) [95] fuses a 2D global feature with a 3D-box feature, using a vanishing-point loss and RoI perspective layers: 81.2% on BoxCars [S].

### 5.4 Viewpoint and hierarchy

- **Viewpoint-aware networks.** EP-CNN [96] adds a pose sub-network and pose-conditioned SE blocks (CompCars, Stanford Cars); Chen et al. [177] jointly learn viewpoint and class (BoxCars116k, CompCars). Accuracies for both are unverified.
- **Make → model hierarchy.** Manole & Diosan [97] find that parallel or cascaded multi-task heads help CNNs in almost all scenarios on Stanford Cars and CompCars (numbers not checked).
- **Large-scale make/model/year.** Gebru et al. [7] [P] report 31.27% on ground-truth boxes and 33.27% on DPM detections over 2,657 classes, with make accuracy 66.38% and model accuracy 51.83%. Errors cluster between sedan and coupe and between extended-cab and crew-cab trucks.

**Year, generation and rebadge evidence.** Stanford Cars splits years only where appearance differs [1] and CompCars merges them [3]. The only controlled test is BoxCars116k, whose hard (with year) and medium (without) splits gave almost identical accuracy [5]. Llorca et al. separate generations via rear emblems [30], and CaMiT documents decline across years [9]. **No paper measures rebadge confusion**: Valev et al. and Gebru et al. give only general similarity arguments [94, 7], and ELLF's confusion list is anecdotal [35].

### 5.5 Open-set recognition and temporal drift

Car-1000 is motivated by Stanford Cars covering only pre-2013 models [8]. Vázquez-Santiago et al. treat VMMR as open-set recognition and discover new classes with a GMM plus clustering on VMMRdb [100]. Veri-Car uses retrieval with a hierarchical multi-similarity loss for unseen models [101]. CaMiT shows that time-incremental pretraining and classifier-only updates both reduce drift [9]. For a deployed demo, a photograph of a post-2012 car is out of distribution by construction, so rejection or back-off to a coarser label is more defensible than a confident wrong answer (Section 9).

---

## 6. Detection and localisation

### 6.1 Detector families

**Two-stage and reference detectors** (torchvision, COCO val2017 box AP [102] [P]): Faster R-CNN R50-FPN 37.0 (41.8M params, 134 GFLOPs), its v2 recipe 46.7 (43.7M, 280), MobileNetV3-L FPN 32.8 (19.4M, 4.5), Mask R-CNN R50-FPN 37.9 (44.4M, 134), RetinaNet R50-FPN v2 41.5 (38.2M, 152) and FCOS R50-FPN 39.2 (32.3M, 128). R-CNN and Fast R-CNN are of historical interest only. Cascade R-CNN chains heads at IoU 0.5/0.6/0.7 and gains about 2–4 AP at high IoU (unverified). Two-stage models fine-tune easily, but ONNX export is fiddly because of dynamic shapes and RoIAlign [102].

**Table 6.1. YOLO family** (Ultralytics docs, COCO val mAP50-95 at 640 px [103] [P]).

| Model | n | s | m | l | x | Notes |
|---|---|---|---|---|---|---|
| YOLOv8 (Jan 2023) | 37.3 | 44.9 | 50.2 | 52.9 | 53.9 | Anchor-free, decoupled head; v8n 3.2M params, 80.4 ms CPU-ONNX |
| YOLOv9 (ECCV 2024) | 38.3 (t) | 46.8 | 51.4 | 53.0 (c) | 55.6 (e) | PGI + GELAN |
| YOLOv10 (NeurIPS 2024) | 38.5 | 46.3 | 51.1 | 53.2 | 54.4 | Consistent dual assignment, NMS-free; T4 TensorRT FP16 1.84–10.7 ms |
| YOLO11 (Sep 2024) | 39.5 | 47.0 | 51.5 | 53.4 | 54.7 | 11n: 2.6M params, 56.1 ms CPU-ONNX, 1.5 ms T4-TRT |
| YOLO12 (NeurIPS 2025) | 40.6 | 48.0 | 52.5 | 53.7 | 55.2 | Area Attention, R-ELAN; Ultralytics warns of training instability and slower CPU, and recommends YOLO11/26 for production |
| **YOLO26 (Jan 2026)** [104] | **40.9** | **48.6** | **53.1** | **55.0** | **57.5** | Current flagship |

YOLO26 [104, 103] has an NMS-free one-to-one head (e2e mAP 40.1 / 47.8 / 52.5 / 54.4 / 56.9 for n to x) and a one-to-many head with NMS. Prediction and validation default to the one-to-many head, so NMS-free inference needs `nms=False`, although the speed benchmarks use the NMS-free head. It removes DFL from box regression and trains with MuSGD, ProgLoss and STAL. The docs claim up to 43% faster CPU-ONNX inference than YOLO11n; the listed figures are 38.9 vs 56.1 ms, with 2.4M parameters and 5.5 GFLOPs. It exports to ONNX, TensorRT, OpenVINO, CoreML and LiteRT. YOLO27 is advertised but not released.

Ultralytics' YOLOv5 tutorial reports that freezing the backbone speeds training and cuts GPU memory at a small mAP cost, while training all layers is most accurate (exact numbers only in images) [115]. Ultralytics models fine-tune and export in one line each, under AGPL-3.0 (well known; not re-checked).

**Table 6.2. DETR family.**

| Model | Venue | COCO AP | Speed / size | Marker |
|---|---|---|---|---|
| DETR | ECCV 2020 | 42.0 (R50); 500-epoch convergence | — | (unverified) |
| Deformable DETR [110] | ICLR 2021 | not in abstract; "10× fewer epochs", better on small objects | — | [P, abstract] |
| DINO-DETR [109] | ICLR 2023 | 49.4 (R50, 12 ep); 51.3 (24 ep); 63.3 test-dev (Swin-L + Objects365) | heavy | [P] |
| RT-DETR [105] | CVPR 2024 | R18 46.5 / R34 48.9 / R50 53.1 / R101 54.3; Ultralytics ships L 53.0 / X 54.8 | 217 / 161 / 108 / 74 FPS T4-TRT FP16; 20–76M params | [P] |
| RT-DETRv2 [106] | arXiv 2024 | S 48.1 / M 51.9 / L 53.4 / X 54.3 | as v1 | [P] |
| D-FINE [107] | ICLR 2025 | N 42.8 / S 48.5 / M 52.3 / L 54.0 / X 55.8 (COCO); up to 59.3 with Objects365 | 2.1–12.9 ms T4-TRT; 4–62M params | [P] |
| RF-DETR [108] | ICLR 2026 (per repo) | Nano 48.4 / S 53.0 / M 54.7 / L 56.5 (Apache-2.0); XL 58.6 / 2XL 60.1 (PML licence) | 2.3–17.2 ms T4-TRT FP16 | [P]; ONNX export not confirmed on README |

D-FINE contributes Fine-grained Distribution Refinement (iterative refinement of box distributions) and GO-LSD (self-distillation from the last decoder layer to earlier layers at no inference cost) [107].

**Open-vocabulary detectors** give zero-shot "car" boxes: Grounding DINO reports 52.5 AP zero-shot on COCO and 26.1 mean AP on ODinW [111] [P], and YOLO-World 35.4 AP on LVIS at 52.0 FPS on a V100 [112] [P]; OWL-ViT numbers are unverified. They are heavier and looser than a fine-tuned YOLO, so they suit fallback or pseudo-labelling roles. A COCO-pretrained YOLO already detects "car" (and truck and bus), so off-the-shelf detection is likely sufficient for cropping [114].

### 6.2 Crop versus full image on Stanford Cars

Several studies measure how much localisation matters for Cars.

**Table 6.3. Effect of localisation on fine-grained accuracy.**

| Study | Setting | Result | Marker |
|---|---|---|---|
| Krause et al. CVPR'15 [92], Table 5, cars-196 | R-CNN with predicted box, no fine-tune / with fine-tune | 57.4% / 88.4% | [P] |
| same | CNN + GT box at test, without / with fine-tune | 59.9% / 89.0% | [P] |
| same | Full model, no box at test / GT box at test | 92.6% / 92.8% | [P] |
| same, CUB | GT box at test | +1.3 points (+1.2 with fine-tune); bbox regression lifts R-CNN detection AP 88.1 → 92.9 | [P] |
| Lin et al. B-CNN [39] | B-CNN(D,M), no boxes | 88.2% (TPAMI: 91.3%) | [P] |
| same | FGComp'13 FV ensemble, without → with box | 82.7 → 87.9% on cars (+5.2); 80.7 → 81.5 on aircraft | [P via Lin; S for Gosselin] |
| same, CUB | Box cropping | about +10 points for FC/FV-CNN (M-Net), about +6 (D-Net), about +1 for B-CNN(D,M) (80.9 → 81.9) | [P] |
| WS-DAN [44] | Attention cropping on CUB: attention vs random crop vs baseline | 87.8 / 86.8 / 86.4%; localisation mIoU 58.7% vs 40.0% | [P] |
| Su & Maji CQD [116], CUB | VGG-M trained on GT crops: tested on crops / full images; trained on full; distilled crop → full | 67.0% / 57.4%; 60.8%; 64.4% | [P] |
| same, quoting Krause | Detector instead of GT boxes on CUB | 67.9 → 65.3 (about −2.6) | [P] |

**With a good detector, predicted boxes cost only about 0.2–0.6 points relative to GT boxes on Cars** (88.4 vs 89.0%). Hand-crafted pipelines depend far more on localisation (+5.2 points) than strong deep models, which learn to localise implicitly. WS-DAN's authors skip localisation evaluation on Cars because most cars fill the frame [44]. A crop mismatch between training and test is harmful (67.0 → 57.4% on CUB [116]), which favours training on detector-like or jittered crops. Finally, Wei et al. list objects that do not fill the frame as an open real-world problem [18]; real photographs rarely resemble Stanford Cars' large, centred cars, so detection matters more in deployment than on the benchmark.

### 6.3 Metrics

- **IoU** = area(B_pred ∩ B_gt) / area(B_pred ∪ B_gt).
- **COCO protocol** [113] [P]: AP averaged over 10 IoU thresholds (0.50:0.05:0.95) and over categories (COCO does not distinguish AP from mAP); AP50 (VOC) and AP75 (strict); small/medium/large buckets at 32² and 96²; AR@1/10/100 with at most 100 detections per image.
- **Single-object protocols** suit Stanford Cars: mean IoU of the top-1 box; **CorLoc**, the share of images with IoU ≥ 0.5 (WS-DAN reports its complement as localisation error); and ILSVRC "top-1 loc", which also requires the correct class (used by Grad-CAM [125]).
- **End-to-end top-1 accuracy** with predicted crops vs GT crops vs the full image, which is the number that matters for the application.

**No paper found reports COCO-style detection mAP on Stanford Cars boxes as a benchmark.** IoU@0.5 is expected to be near saturation, since each image contains one large car, but this has to be measured.

---

## 7. Hybrid classical + deep integration

This section separates evidence obtained on Stanford Cars or vehicles from evidence obtained elsewhere, and lists the gaps explicitly.

### 7.1 Classical preprocessing

- **Degradation sensitivity.** Dodge & Karam [117] find ImageNet CNNs most susceptible to blur and noise and much less affected by JPEG and low contrast. This argues for denoising and deblurring before contrast stretching.
- **Low-light enhancement before detection.** ExDark [118] (7,363 low-light images, 12 classes) shows that low-light effects go deeper than simple illumination invariance can fix. PE-YOLO [119] [P] reports ExDark mAP@0.5 with YOLOv3:

  | Setting | mAP@0.5 |
  |---|---|
  | Baseline | 76.4 |
  | + KinD pre-enhancement | 76.3 |
  | + MBLLEN | 76.8 |
  | + Zero-DCE | 76.9 |
  | PE-YOLO (end-to-end enhancement) | 78.0 |

  **Off-the-shelf enhancement before a detector gains at most 0.5 mAP. Larger gains need joint training.** Secondary studies agree: YOLOv7 improves from 0.785 to 0.794 with Zero-DCE (unverified primary), and one study reports TBEFN two-stage 0.49 → 0.574 [120] (unverified detail).
- **CLAHE** [167]. The evidence is mostly small single-dataset studies. One reports gains for VGG16, InceptionV3 and EfficientNet but a drop from 95% to 84% for ResNet34 (unverified, secondary). CLAHE should be treated as condition-specific: apply it only to low-contrast or dark inputs, and validate.
- **White balance.** Afifi & Brown [121] show that colour-constancy errors fool classifiers and segmenters, and that WB-emulating augmentation and WB pre-correction help on CIFAR and ADE20K (no numbers in the abstract). Gray-world [169] and MSRCR [170] are the classical baselines. **No study quantifying them on FGVC was found.**
- **Shape bias.** Geirhos et al. [122] find ImageNet CNNs texture-biased. Shape-biased pretraining (Stylized-ImageNet) lifts Faster R-CNN from 70.7 to 75.1 mAP50 on VOC07 and from 52.3 to 55.2 on COCO [P].

### 7.2 Hand-crafted + CNN feature fusion

- **Baselines on Cars.** Hand-crafted pipelines reach 51–59% (vanilla SIFT-FV; Section 3.1 explains the version discrepancy) up to 82.7% and 87.8/87.9% with boxes (engineered FV ensemble), compared with 91.3% for a fine-tuned B-CNN [39, 34]. FV-pooled CNN features reach 70–79% [39]. Gosselin et al. obtain their gain by late-fusing two complementary FV systems; per-component gains are unverified [34].
- **General fusion.** Nanni, Ghidoni & Brahnam [123] fuse CNN transfer features (output layer, penultimate layer, and several reduced layers) with LTP, LPQ and LBP variants through SVM score fusion across many datasets. The size of the fusion gain could not be confirmed from the abstract (unverified), and no car FGVC result was found. Their methodology is reusable: Wilcoxon signed-rank tests across folds, and the Q-statistic for classifier independence.
- **Vehicle-specific fusion** (CS-LBP + HOG; PCANet + HOG + Hu moments on BJUT; HOG + EfficientNet + SVM on Nigerian road data, MYJAS 2023; a 2016 *Sensors* local-tiled CNN) is low-tier and never compared with modern CNNs. Frozen features from a label-smoothed backbone may also be a weak fusion partner [69].

**Gap.** *No primary paper reports a measured LBP, HOG or colour-histogram + CNN fusion gain on Stanford Cars.* Report 03 expects a fusion gain of at most about 1 point once the CNN is above 90%; this is an expectation, not a measurement. Late or score-level fusion, for example logistic regression on [CNN logits ‖ HSV histogram ‖ HOG], is cheap to test and yields an honest ablation.

### 7.3 Classical ROI refinement

- **GrabCut** [166] initialised from boxes. Khoreva et al.'s "Simple Does It" [124] [P] builds training-free instance masks from detector boxes on VOC12, scored in mAPr@0.5: rectangle 21.6, ellipse 29.5, MCG 28.3, **GrabCut 38.5**, and GrabCut+ (HED edges as the pairwise term) **41.1**. Used as training labels for semantic segmentation, they give Box 61.2 → GrabCut+ 63.4 → GrabCut+i 64.3 val mIoU; M∩G+ reaches about 95% of full supervision. Krause et al. also used box-initialised GrabCut for car foregrounds [92], and ELLF found that GrabCut failures cause part-detector failures [35].
- **Grad-CAM boxes** [125] [P]. Threshold the map at 15% of its maximum, take the largest connected component and draw its box. With VGG-16, ILSVRC-15 top-1 localisation error is 56.51%, against 57.20% for CAM and 70.92% for c-MWP, with no retraining.
- **Attention cropping** (WS-DAN, RA-CNN, NTS-Net; Section 4.1) is the learned counterpart.
- **Spectral-residual saliency** [168] is available in opencv-contrib and is a cheap fallback ROI when the detector misses. No FGVC gain numbers were found.

### 7.4 Edge, contour and geometric priors

- **Vehicle geometry (strongest evidence).** BoxCars' unpacked image, rasterised shape and orientation inputs [4, 5]; 3DPN's vanishing-point constraint [95]; and Lin et al.'s joint 3D fitting and classification [33] (Section 5.3).
- **Shape streams.** Gated-SCNN [127] adds a gated shape stream and reports about +2% mIoU and +4% boundary F-score on Cityscapes. Whether Canny gradients are fused could not be confirmed from the abstract.
- **Extra input channels.** CoordConv [128] reports 24% better IoU for Faster R-CNN, on a toy MNIST detection task only. Khoreva's DeepLabBOX, which feeds RGB plus a binary box mask as a fourth channel, reaches 44.8 mAPr@0.5 against 41.1 for GrabCut+ [124]. This is a concrete "geometric prior as an input channel" result.
- **Edges for classification.** arXiv 2401.07028 adds edges back to RGB and reports gains on CIFAR-10 and Caltech101 (numbers unverified) [129]. Su & Maji distil colour models into edge-map models on CUB and Cars, which shows that edges alone carry much of the fine-grained signal [116].
- **Symmetry.** Hsieh's symmetrical SURF is classical only [29]; no deep symmetry-prior VMMR paper was found.

### 7.5 Summary of hybrid evidence and gaps

| Hybrid component | Evidence on Stanford Cars / vehicles | Evidence elsewhere | Gap |
|---|---|---|---|
| Low-light / CLAHE / WB preprocessing | none | ≤0.5 mAP for off-the-shelf enhancement (ExDark); WB helps CIFAR/ADE20K | **No FGVC or Cars measurement** |
| Hand-crafted + CNN fusion | hand-crafted alone 51–88%; no fusion measurement | Nanni (gain size unverified) | **No measured fusion gain on Cars** |
| GrabCut / CAM ROI refinement | GrabCut used inside Krause 2015 and ELLF | Khoreva VOC masks; Grad-CAM ILSVRC localisation | **No isolated ablation of classical refinement on Cars crops** |
| Edge map as extra CNN channel | none for VMMR | weak (CIFAR, Caltech101; unverified) | **Open niche**, consistent with BoxCars' gain from a low-resolution shape input |
| 3D / geometric priors | BoxCars, 3DPN, Lin 2014: verified gains | — | Requires 3D boxes or calibration, which Stanford Cars lacks |
| Symmetry prior | classical only (Hsieh) | — | No deep counterpart |
| Detection mAP on Cars boxes | none published | — | Must be measured |

---

## 8. Robustness

### 8.1 Common corruptions

- **Protocol.** Hendrycks & Dietterich define ImageNet-C: 15 corruptions in four families (noise, blur, weather, digital) at 5 severities, plus 4 held-out corruptions [130]. Its metrics are Corruption Error normalised by AlexNet and averaged into mCE, and Relative mCE, which measures degradation relative to clean error. The headline finding is that relative robustness barely changed from AlexNet to ResNet. Report 04 takes the protocol details from its own knowledge of the paper; the fetched abstract confirms only the benchmark's purpose.
- **Fine-grained corruption sets.** Yucel et al. [131] release CUB-C (with SUN-C and AWA2-C) using the same 15 × 5 design. It is aimed at zero-shot learning, but the code is reusable.
- **Stanford Cars.** `tanganke/stanford_cars` ships seven corrupted copies of the 8,041-image test set: contrast, gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate and spatter [14]. It has no documentation of how they were generated and no severity levels, so it is not a citable benchmark. **No peer-reviewed "Stanford-Cars-C" exists.** Generating one with the `imagecorruptions` code is the safer choice.

### 8.2 Low resolution

All of the following crop to the bounding box and downsample Cars to 50×50 px.

**Table 8.1. Low-resolution results on Stanford Cars.**

| Study | Setting | Accuracy | Marker |
|---|---|---|---|
| Peng et al., ICIP'16 [99] | AlexNet trained high-res, tested low-res | **1.7%** | [P] |
| same | Trained and tested low-res | 50.4% | [P] |
| same | Super-resolution baseline (NBSRF) | 50.8% | [P] |
| same | Mixed high+low training (low-res / high-res test) | 59.3% / 72.9% | [P] |
| same | Staged training (high-res first, then low-res) | 59.5% | [P] |
| same | High-res oracle | 80.3% | [P] |
| Cai et al. RACNN [132] | SR layers in front of the classifier | 63.8% vs 50.4% (AlexNet) | [P]; PRL venue (unverified) |
| same, frozen features | GoogLeNet | 46.85 → 50.76% | [P] |
| DADKD | — | 88.19% | (unverified; venue unverified) |

On CUB, Cai et al.'s gain is larger at 25×25 (+13 points) than at 100×100 (under 2 points) [132]. Vision Mamba distillation for low-resolution FGVC [133] is a more recent line. Gebru et al. downsampled product shots to match Street View resolution [7], and BoxCars notes that surveillance images are usually small [5].

**The collapse under resolution mismatch (1.7%) is the single most important robustness result for this project.** Training must include multi-resolution augmentation.

### 8.3 Occlusion

Gunasekaran & Jaiman add synthetic occlusions to Stanford Cars and test VGG19, ResNet-50/101, GoogLeNet and DenseNet-121 (numbers not checked) [98], and BoxCars' ImageDrop targets occlusion [5]. Standard regional-dropout ablation arms are Cutout (2.56% / 15.20% error on CIFAR-10/100) [134], Random Erasing (claims occlusion robustness; no numbers in abstract) [135], Hide-and-Seek [136], and CutMix, which reports better corruption robustness and OOD detection [137]. They can be evaluated on synthetic occlusion of 10/25/50% of the box.

### 8.4 Viewpoint

CompCars shows a large per-view gap (52.4% front vs 42.8% side) [3]. Krause's 3D normalisation and flip augmentation show the value of viewpoint coverage [1, 2]. BoxCars evaluates on unseen cameras, the realistic setting [5]. ELLF's part detectors fail under front/rear rotation [35], and Anderson et al. identify unseen poses as a major failure class [156].

### 8.5 Low light

ExDark [118] has 7,363 images, 12 classes and 10 lighting conditions, and notes that low-light images are under 2% of VOC, ImageNet and COCO. It has no make or model labels, so it is only a qualitative stress set for "car"; synthetic gamma, brightness and noise darkening of Stanford Cars is more practical. The CompCars surveillance part reaches about 98% despite night, rain and fog because its views are fixed and frontal [3], and taillight features reach about 93% at night [31]. Enhancement must be trained jointly to help (Section 7.1) [119].

### 8.6 Test-time augmentation and adaptation

Shanmugam et al. [138] show that TTA flips many correct predictions to wrong ones even when mean accuracy rises, and that learned aggregation beats averaging, so **TTA should be reported with its flip counts**. TENT [139] minimises entropy online by updating BatchNorm statistics and affine parameters (ImageNet-C numbers unverified); it needs test batches, so single-image inference allows only BatchNorm-statistic adaptation. Horizontal-flip TTA and higher test resolution are cheap boosters (Section 4.3).

---

## 9. Confidence: calibration, selective classification, hierarchical back-off and open-set recognition

### 9.1 Calibration

Guo et al. [140] [P] report Stanford Cars directly, using ResNet-50 with an 8,041 / 4,020 / 4,020 split in which the original test set is halved into validation and test.

**Table 9.1. Calibration of ResNet-50 on Stanford Cars** (Guo et al.).

| Measure | Uncalibrated | Temperature scaling | Histogram binning | BBQ | Matrix scaling |
|---|---|---|---|---|---|
| ECE | 4.30% | **2.35%** | 1.74% | 1.84% | 10.5% (overfits with 196 classes) |
| MCE | 41.55% | 20.23% | — | — | — |
| Error | 14.28% | 14.28% (unchanged by construction) | — | — | — |

For comparison, CUB ECE falls from 9.19% to 1.85%. The resulting reference protocol is to hold out a calibration split, fit the temperature by NLL, and report ECE, MCE and reliability diagrams.

Minderer et al. [141] find recent non-convolutional models (ViT, MLP-Mixer) among the best calibrated, with weaker size and shift trends; architecture is a major factor. Zero-shot CLIP is miscalibrated, but one learned temperature per CLIP model transfers across datasets and prompts [142]. Standard fine-tuning of vision-language models hurts in-distribution and especially out-of-distribution ECE [143].

### 9.2 Selective classification and the reject option

Geifman & El-Yaniv [144] threshold the softmax response with a guaranteed risk bound (SGR), reaching 2% top-5 risk at about 60% coverage on ImageNet with 99.9% confidence. SelectiveNet [145] trains the predictor and a selection head jointly at a target coverage. Geifman, Uziel & El-Yaniv [146] are usually cited for AURC and E-AURC, although the fetched abstract does not mention them (unverified). The recommended deliverables are risk–coverage curves, AURC and accuracy at fixed coverage (for example 80/90/95%).

### 9.3 Hierarchical back-off

"Hedging your bets" (DARTS) [147] predicts internal hierarchy nodes to guarantee a target accuracy while maximising information gain; Krause is a co-author, and the cars hierarchy (model-year → model → make → body type) fits naturally. Goren et al. [148] add hierarchical risk–coverage curves and a guaranteed-accuracy algorithm, and find that CLIP, IN-21k pretraining and distillation help. Conformal variants restrict prediction sets to internal nodes [149], and users prefer hierarchical sets to flat ones [150].

The natural design predicts the model-year if confidence ≥ τ₁, otherwise the model, otherwise the make, and otherwise abstains. Three Stanford Cars issues shape it:

- The "year" is a representative-appearance label (Section 1.3), and same-generation twins such as the Dodge Caliber 2007/2012 are near label noise (Table 2.2), so backing off from year to model is often the *correct* answer.
- Make aggregation must handle multi-word makes and the split Hummer and Ram brands (Section 2.7).
- Rebadged twins such as Express/Savana cross make boundaries, so make-level back-off cannot resolve them; body type or platform is the coherent coarser label.

### 9.4 Out-of-distribution and open-set detection

Post-hoc scorers include MSP [171], ODIN [172], Mahalanobis [173], Energy (18.03% lower average FPR95 than softmax on CIFAR-10 WRN) [151], ViM (90.91% average AUROC with BiT-S on ImageNet OOD) [152] and KNN-OOD (24.77% FPR95 reduction vs SSD+ on ImageNet) [153]. The generalised OOD survey [23] [P] finds on OpenOOD that post-hoc scores generally beat methods needing special training, that augmentation-based methods (PixMix, CutMix, Mixup) and ensembles are the most effective, that extra outlier data seems unnecessary, and that feature KNN is robust across benchmarks.

**Semantic Shift Benchmark (SSB).** Vaze et al. [154] show that open-set AUROC correlates strongly with closed-set accuracy, and that a well-trained Maximum Logit Score (MLS) baseline matches or beats specialist methods such as ARPL+. The SSB covers CUB, **Stanford Cars**, FGVC-Aircraft and ImageNet-21K-P, with unknowns split by attribute-based semantic similarity. The Stanford Cars split [P, Vaze Table 2] has 98 known classes (3,948 test images), 76 "Easy" unknown classes (3,170 images), no "Medium" classes, and 22 "Hard" unknown classes (923 images). CUB, by contrast, has 100 known and 32/34/34 Easy/Medium/Hard, with Medium merged into Hard in the repository. The official splits are on GitHub [154].

**Table 9.2. SSB Stanford Cars results** (ResNet-50).

| Method | Closed-set accuracy (98 classes) | AUROC Easy / Hard |
|---|---|---|
| MLS | 97.1% | 94.0 / 82.2 |
| ARPL+ | 96.9% | 94.8 / 83.6 |

*Verification discrepancy.* Report 05 marks these values as verified via ar5iv and the official README. Report 04 saw them only second-hand [155], because its PDF copy was truncated after Table 2, and marks them unverified (ARPL about 96.9% accuracy and 94.8/83.6 AUROC; MLS Hard about 82.2). The two reports agree on the values. The 97.1% is a 98-class figure and is not comparable with 196-way accuracy. Vaze et al.'s recipe uses 600 epochs, RandAugment and label smoothing (both tuned on validation), and a cosine schedule with warmup.

**Interpretation.** A Hard-split AUROC in the low 80s means that an unseen car model that closely resembles a known one is hard to reject, even for strong models. This matches the out-of-set twin risk in Section 2.6. Ni & Huttunen's gallery-based metric learning [19], Veri-Car's hierarchical retrieval [101] and Vázquez-Santiago's GMM clustering [100] offer complementary ways to handle unseen models (Section 5.5).

---

## 10. Model optimisation and deployment

### 10.1 Quantization

- **General evidence.** Krishnamoorthi [157] [P] reports that INT8 post-training quantization (per-channel weights, per-layer activations) stays within 2% of FP32 across many CNNs. Quantization-aware training (QAT) narrows the gap to about 1%, while 4-bit QAT loses 2–10%, more for small networks. The paper reports 2–3× CPU speed-ups and roughly 4× smaller weights. Jacob et al. [174] introduced the integer-only inference that underlies TFLite and ONNX Runtime INT8. Gholami et al. [25] state that reductions of 4× to 8× are often realised at 4 bits or fewer (abstract only).
- **ONNX Runtime** [158] [P] recommends static (calibrated) quantization for CNNs and dynamic quantization for transformers and RNNs. QDQ with S8S8 is the default and first choice; U8U8 is the fallback if accuracy drops, and S8S8 in QOperator format should be avoided on x86. On AVX2/AVX512 CPUs without VNNI, U8S8 can saturate, so `reduce_range` (7-bit weights) is needed, generally also for per-channel quantization. Calibration options are MinMax, Entropy and Percentile; `quant_pre_process()` should run first, and `qdq_loss_debug` locates layers to keep in FP32. ORT does not do QAT and gives no speed-up figures, warning that older CPUs can get *slower*.
- **PyTorch** [163] advises about 100 calibration mini-batches, Conv-BN-ReLU fusion, symmetric per-channel weights with affine per-tensor activations, and keeping sensitive layers in FP32 (skipping 2 of 28 MobileNet-v1 conv layers recovers near-FP32 accuracy). Models over 10M parameters are more robust, and QAT needs about 10% of the original schedule at 1% of the original learning rate. `torch.ao.quantization` is moving to torchao, with deletion planned "in 2.10 if there are no blockers" [164]; its removal status could not be confirmed, so ONNX Runtime is the lower-risk path.
- **Ultralytics export** [114] replaces `half`/`int8` with `quantize` (16, 8 or 32); ONNX INT8 uses ORT static quantization calibrated on `data=` with `fraction=`. Vendor speed claims (up to 5× GPU with TensorRT, 3× CPU with ONNX or OpenVINO) are unverified.
- **NNCF / OpenVINO** [161] claims under 0.5% top-1 drop for most ImageNet models (unverified; from a snippet).

**Gap.** No primary study measuring the INT8 PTQ drop specifically on Stanford Cars was found. ProtoQuant [160] quantises prototypes on CUB and Cars, not backbones.

### 10.2 Pruning

- Blalock et al. [24] [P] reviewed 81 papers and found no standard benchmarks and fragmented comparisons. Pruning improves the size/speed–accuracy trade-off of a given architecture but generally helps less than switching to a better architecture. It helps most on architectures that are inefficient to begin with, and magnitude pruning is a consistently strong baseline.
- Luo & Wu [159] [P] prune VGG-16 on CUB with entropy-based filter pruning, going from 71.95% to 69.55% (Pruned-FC) or 69.69% (Pruned-GAP). They note that fine-grained recognition may need more channels to represent subtle inter-class differences. **This is the only primary evidence found that FGVC is more sensitive to compression.**
- Deep Compression [175] and structured filter pruning [176] report their gains on ImageNet and CIFAR, not FGVC (numbers unverified here).

### 10.3 Distillation

Hinton et al. introduced knowledge distillation [162]. Su & Maji's Cross-Quality Distillation is the FGVC-specific evidence: crop → full image (CUB 57.4 → 64.4%), colour → edges, and low-resolution cars [116]. PMD distils a multi-task model into a plain backbone at no inference cost, reaching 97.3% (TResNet-L) and 95.3% (ResNet-50) [56]. Low-resolution distillation (ViMD [133]; DADKD, unverified) targets the resolution gap, and D-FINE's GO-LSD self-distillation is free at inference [107].

### 10.4 Deployment practicalities

- **Hugging Face Spaces** [165] (docs fetched 2026-11-06). CPU Basic offers 2 vCPU, 16 GB RAM and 50 GB non-persistent disk, but the docs now state that compute-backed (Gradio or Docker) Spaces require a PRO plan; static Spaces are free. Free accounts older than 30 days with a verified email can host up to 2 ZeroGPU Gradio Spaces (half an RTX Pro 6000, 48 GB; daily quota 5 min free, 2 min unauthenticated, 40 min PRO; 60 s per `@spaces.GPU` call; Gradio only, PyTorch 2.8+, no `torch.compile`). Free CPU Spaces sleep after 48 h and restart cold, and only outbound ports 80, 443 and 8080 are open.
- **Detector export.** Two-stage detectors export awkwardly; YOLO, RT-DETR, D-FINE and RF-DETR have documented ONNX or TensorRT paths (Section 6.1).
- **Foundation backbones on a T4.** g-, G- and 7B-size models run very slowly in fp16 on a T4, so report 01 recommends ViT-L or SO400M-size models and extrapolates, without measurement, about 92–95% for their linear probes. Verified checkpoint names are `timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (90.05% IN-1k top-1 per its model card [89]) and `timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`; other candidate names (CLIP-L/336, LAION bigG, SigLIP2-SO400M, DINOv2-L, DINOv3-L, ConvNeXtV2, TResNet-L) were not checked.

---

## 11. Evaluation methodology and failure taxonomies

### 11.1 Reproducibility pitfalls

1. **No validation split.** Tuning on test is common [90]. Hold out about 10% of train, stratified per class, for tuning and calibration, and report test accuracy once.
2. **Comparability.** Reported numbers mix resolutions (224 vs 448), test-time box use (for example Krause's 92.6%), backbones and pretraining data, and protocols (VTAB-style PEFT at 224 px; SSB 98-class) [18, 83, 154]. Every result should state its resolution, pretraining data and crop source (GT, detected or full image). For a deployed application, detected crops are the honest setting.
3. **Contamination.** Report an ImageNet-only baseline alongside any web-scale result. State the DINOv2 retrieval and LAION/CLIP caveats (Section 2.5), and consider a near-duplicate check between the test set and any extra data, for example with CLIP-embedding cosine similarity [17].
4. **Label noise.** Inspect top-loss images before attributing them to the model [16].
5. **Data integrity.** Verify the counts (8,144 / 8,041 / 196) and the presence of test labels after downloading from a mirror [12, 14].

### 11.2 Metrics

| Aspect | Metrics |
|---|---|
| Classification | Top-1/top-5; make- and body-type-level accuracy from an aggregated confusion matrix, separating near-miss (same make) from gross errors [35]; year errors read as appearance-class errors (Section 1.3) |
| Localisation | Mean IoU, CorLoc at IoU ≥ 0.5, COCO mAP50 and mAP50-95 (Section 6.3) |
| Confidence | ECE, MCE, reliability diagrams; risk–coverage curves and AURC; OOD AUROC and FPR95 on SSB Easy/Hard |
| Robustness | mCE-style relative degradation; accuracy vs resolution, occlusion fraction and darkening; TTA flip counts |
| Efficiency | Model size, per-stage latency, FP32 vs FP16 vs INT8 top-1 |
| Statistics | Wilcoxon signed-rank tests across seeds or folds; Q-statistic for fusion or ensemble diversity [123] |

### 11.3 Failure taxonomies

- **Anderson et al.** [156] [P] pool predictions from several state-of-the-art FGVC models. On Cars, about 88.8% of test images were correct for all models and about 2.7% (217 images) wrong for all, as read from their overlap figure; errors are complementary, and ensembling helped by over 5% on CUB. Their taxonomy has five classes: *Similar Class* (confusable classes), *Non-target Subject* (several objects or an off-centre target), *Inadequate Representation* (pose or background unseen in training), *Poor Quality* (occlusion, blur, bad lighting, too small) and *Other*.
- **Confusion-pair ranking.** Krause et al.'s ICPR'14 Eq. 3.3 ranks symmetric confusion pairs. The top pairs are trim and body-style variants and rebadged vans [35]. The class-structure analysis in Section 2.6 extends this into a predefined twin taxonomy (year twins, rebadges, platform siblings, trim twins, body-style twins) against which confusions can be tallied.
- **Explanations.** Grad-CAM [125] and the gradient-free Score-CAM [126] can reproduce ELLF's finding that grilles and headlights are discriminative and tyres are not [35]. A "does the CAM land on the background?" measure acts as a sanity check.

---

## 12. Research gaps this project can address

Each gap below can be addressed with modest compute on Stanford Cars.

1. **Rebadge and platform-twin confusion.** No paper measures it; only anecdotal evidence exists [35, 94]. A twin-group confusion analysis over Table 2.3, plus a test of whether hierarchical back-off helps (it cannot at make level), would be new.
2. **Year-twin behaviour.** Apart from BoxCars116k's hard vs medium comparison [5], no study isolates same-model, different-year confusion. The seven year-twin pairs (Table 2.2) are a natural testbed, read in light of the representative-year definition.
3. **INT8 quantization drop on Cars.** Unmeasured [160], although the only FGVC compression evidence suggests heightened sensitivity [159]. FP32 vs FP16 vs ORT static INT8 fills it directly [158].
4. **Hand-crafted + deep fusion on Cars.** No measured LBP/HOG/colour + CNN gain exists [123]; a late-fusion ablation with a Wilcoxon test is informative even if null.
5. **Edge maps as an extra VMMR input.** Untested, although shape inputs help [4] and edges carry fine-grained signal [116].
6. **Classical ROI refinement on Cars.** No isolated ablation of GrabCut/contour box tightening or saliency/Grad-CAM fallback boxes [124, 125].
7. **Preprocessing on FGVC.** Gray-world, MSRCR and CLAHE are unquantified on any FGVC benchmark [121, 167].
8. **Detection on Stanford Cars boxes.** No published mAP, mean IoU or CorLoc, and no modern predicted- vs GT-crop vs full-image comparison since Krause 2015 [92].
9. **A documented Stanford-Cars-C** does not exist [14, 130].
10. **Contamination audits.** Cars–LAION overlap and within-dataset near-duplicates are unmeasured [81, 17].
11. **Calibration of modern backbones on Cars.** The reference is a 2017 ResNet-50 [140]; fine-tuned ViT/CLIP calibration and hierarchical back-off on the make/model/year tree [147, 148] are unreported.

---

## 13. Implications for our design

The detailed plan is in `docs/FINAL_APPROACH.md`. The literature supports the following high-level choices.

- **Backbone.** Report 01 ranks a frozen CLIP-family linear probe as the best accuracy per GPU-hour, followed by full fine-tuning of a CLIP/EVA-02/SigLIP ViT; report 05 targets a strong pretrained backbone at about 384–448 px. ImageNet-only CNNs and ViTs reach about 92–94% [75, 74, 70], and web-scale features reach about 96.5% with only a linear probe [80]. FGVC-specific modules add about 1–2 points [18, 90]. Contamination caveats and an ImageNet-only baseline must accompany any web-scale result (Section 2.5).
- **Detection.** Use detect-then-crop with an off-the-shelf COCO "car" detector (YOLO26 or YOLO11 class, or Apache-2.0 RF-DETR if the licence matters) [103, 108]. Train on crops of the same type as at test time [116]. Report full-image vs GT-crop vs predicted-crop accuracy [92], plus mean IoU and CorLoc.
- **Hybrid pipeline.** Conditional preprocessing, GrabCut or contour ROI refinement with a saliency or Grad-CAM fallback, and late fusion of CNN outputs with colour, texture and geometric features, each reported as an ablation including null results [119, 124, 123]. A dense-SIFT Fisher-vector pipeline is the classical baseline, framed as a lower bound (51–59% vanilla; 82.7% engineered) [39, 34].
- **Robustness.** Multi-resolution training to avoid the 1.7% collapse [99], an occlusion-augmentation arm [137, 135], and evaluation on a self-generated Stanford-Cars-C, resolution sweeps, occlusion and darkening [130].
- **Confidence.** Temperature scaling on a held-out split [140]; MSP, maximum-logit and energy thresholds with risk–coverage analysis [144, 154, 151]; hierarchical back-off along model-year → model → make (or body type for cross-make twins) → abstain [147]; and open-set evaluation on the SSB Cars split [154].
- **Optimisation.** Export through ONNX and compare ONNX Runtime static INT8 (QDQ/S8S8) with FP32 and FP16 on top-1, size and latency [158]. Prefer a smaller architecture to pruning [24]. Plan for 2-vCPU CPU inference, or a ZeroGPU fallback, on Hugging Face [165].
- **Evaluation.** Use a stratified validation split; report the resolution, pretraining and crop source of every number; tally failures with the Anderson taxonomy and the twin taxonomy [156]; and note label noise [16].

---

## References

### Datasets and data resources

- **[1]** Krause, Stark, Deng, Fei-Fei. "3D Object Representations for Fine-Grained Categorization." 3dRR Workshop at ICCV 2013. https://www.cs.princeton.edu/~jiadeng/paper/KrauseStarkDengFei-Fei_3DRR13.pdf ; https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/html/Krause_3D_Object_Representations_2013_ICCV_paper.html
- **[2]** Krause. PhD dissertation, Stanford University, 2016 (reprints the 3dRR'13, ICPR'14 and CVPR'15 papers). https://svl.stanford.edu/assets/publications/pdfs/krause2016dissertation.pdf
- **[3]** Yang, Luo, Loy, Tang. CompCars. CVPR 2015; journal version arXiv 1506.08959. https://arxiv.org/abs/1506.08959 ; https://openaccess.thecvf.com/content_cvpr_2015/html/Yang_A_Large-Scale_Car_2015_CVPR_paper.html
- **[4]** Sochor, Herout, Havel. "BoxCars: 3D Boxes as CNN Input for Improved Fine-Grained Vehicle Recognition." CVPR 2016. https://openaccess.thecvf.com/content_cvpr_2016/html/Sochor_BoxCars_3D_Boxes_CVPR_2016_paper.html ; https://openaccess.thecvf.com/content_cvpr_2016/papers/Sochor_BoxCars_3D_Boxes_CVPR_2016_paper.pdf
- **[5]** Sochor, Špaňhel, Herout. BoxCars116k. IEEE T-ITS 2018/19. https://arxiv.org/abs/1703.00686 ; https://github.com/JakubSochor/BoxCars
- **[6]** Tafazzoli, Frigui, Nishiyama. VMMRdb. CVPR Workshops 2017. https://openaccess.thecvf.com/content_cvpr_2017_workshops/w9/html/Tafazzoli_A_Large_and_CVPR_2017_paper.html ; https://github.com/faezetta/VMMRdb
- **[7]** Gebru et al. Fine-grained car detection for visual census estimation. AAAI 2017. https://ar5iv.labs.arxiv.org/html/1709.02480 ; https://ojs.aaai.org/index.php/AAAI/article/view/11174
- **[8]** Hu, Li, Yan, Shao, Luo. Car-1000. FGVC11 Workshop at CVPR 2024; arXiv 2503.12385. https://arxiv.org/abs/2503.12385 ; https://github.com/toggle1995/Car-1000
- **[9]** Lin, Popescu, Le Borgne et al. CaMiT. NeurIPS 2025 Datasets & Benchmarks. https://arxiv.org/abs/2510.17626
- **[10]** Liu et al. VehicleID (CVPR 2016) and VeRi-776 (2016), via the Vehicle re-ID collection. https://github.com/layumi/Vehicle_reID-Collection
- **[11]** torchvision `StanfordCars` dataset source and documentation. https://github.com/pytorch/vision/blob/main/torchvision/datasets/stanford_cars.py ; https://pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html
- **[12]** pytorch/vision issue #7545 (Stanford Cars download broken). https://github.com/pytorch/vision/issues/7545
- **[13]** clip-benchmark `wds_cars` (class-name mirror). https://huggingface.co/datasets/clip-benchmark/wds_cars
- **[14]** `tanganke/stanford_cars` (Hugging Face, with corrupted test splits). https://huggingface.co/datasets/tanganke/stanford_cars
- **[15]** `stanford_cars_refined` (colour-refined labels). https://github.com/morrisfl/stanford_cars_refined
- **[16]** Cleanlab. Stanford Cars label-error audit (blog, May 2023). https://cleanlab.ai/blog/csa/csa-3/
- **[17]** Barz, Denzler. "Do we train on test data?" https://arxiv.org/pdf/1902.00423

### Surveys

- **[18]** Wei, Song, Mac Aodha, Wu, Peng, Tang, Yang, Belongie. "Fine-Grained Image Analysis with Deep Learning: A Survey." IEEE TPAMI 44(12), 2022. https://arxiv.org/abs/2111.06119
- **[19]** Ni, Huttunen. "Vehicle Attribute Recognition by Appearance: Computer Vision Methods for Vehicle Type, Make and Model Classification." J. Signal Processing Systems, 2020. https://arxiv.org/abs/2006.16400
- **[20]** Gayen, Maity, Singh, Geem, Sarkar. "Two decades of vehicle make and model recognition – Survey, challenges and future directions." J. King Saud Univ. – CIS 36(1), 2024. https://doaj.org/article/60de5e4223bc41158797b1341c0b1d64
- **[21]** Boukerche, Ma. "Vision-based Autonomous Vehicle Recognition: A New Challenge for Deep Learning-based Systems." ACM CSUR 54(4), 2022. https://www.x-mol.com/paper/1389667688725372928
- **[22]** Boukerche, Siddiqui, Mammeri et al. "Automated Vehicle Detection and Classification: Models, Methods, and Techniques." ACM CSUR 50(5), 2017. https://pure.kfupm.edu.sa/en/publications/automated-vehicle-detection-and-classification-models-methods-and/
- **[23]** Yang, Zhou, Li, Liu. "Generalized Out-of-Distribution Detection: A Survey." IJCV 2024. https://arxiv.org/abs/2110.11334
- **[24]** Blalock, Gonzalez Ortiz, Frankle, Guttag. "What is the State of Neural Network Pruning?" MLSys 2020. https://arxiv.org/abs/2003.03033
- **[25]** Gholami et al. "A Survey of Quantization Methods for Efficient Neural Network Inference." 2021. https://arxiv.org/abs/2103.13630

### Classical methods

- **[26]** Petrovic, Cootes. "Analysis of features for rigid structure vehicle type recognition." BMVC 2004. https://bmva-archive.org.uk/bmvc/2004/abstracts/abstract_165.html
- **[27]** Pearce, Pears. "Automatic make and model recognition from frontal images of cars." AVSS 2011. https://www-users.york.ac.uk/~np7/research/bib/avss11pearce-bib.html
- **[28]** Psyllos, Anagnostopoulos, Kayafas. Vehicle logo recognition with enhanced SIFT. IEEE T-ITS 2010. https://www.scilit.net/publications/fa0bd52b7d0c232f04b0a241bd43a868
- **[29]** Hsieh, Chen, Chen. Symmetrical SURF for vehicle make and model recognition. IEEE T-ITS 2014. https://scholars.ntou.edu.tw/handle/123456789/24744
- **[30]** Llorca et al. Vehicle model and trim emblem recognition. ITSC 2014. https://invett.aut.uah.es/sotelo/ITSC2014-VehicleLogo.pdf
- **[31]** Boonsim, Prakoonwit. Night-time vehicle make and model recognition. Pattern Analysis & Applications 2017. https://link.springer.com/article/10.1007/s10044-016-0559-6
- **[32]** Stark et al. Fine-grained categorization for 3D scene understanding (structDPM). BMVC 2012. https://bmva-archive.org.uk/bmvc/2012/BMVC/paper036/index.html
- **[33]** Lin, Morariu, Hsu, Davis. "Jointly optimizing 3D model fitting and fine-grained classification." ECCV 2014. https://www.doi.org/10.1007/978-3-319-10593-2_31
- **[34]** Gosselin, Murray, Jégou, Perronnin. "Revisiting the Fisher vector for fine-grained classification." Pattern Recognition Letters 49:92–98, 2014. https://hal.archives-ouvertes.fr/hal-01056223 ; https://www.researchgate.net/publication/264425714_Revisiting_the_Fisher_vector_for_fine-grained_classification
- **[35]** Krause, Gebru, Deng, Li, Fei-Fei. "Learning Features and Parts for Fine-Grained Recognition" (ELLF). ICPR 2014. https://cs.stanford.edu/groups/vision/pdf/icpr14.pdf
- **[36]** Chai, Lempitsky, Zisserman. "Symbiotic Segmentation and Part Localization for Fine-Grained Categorization." ICCV 2013. https://openaccess.thecvf.com/content_iccv_2013/html/Chai_Symbiotic_Segmentation_and_2013_ICCV_paper.html
- **[37]** Gavves et al. "Fine-grained categorization by alignments." ICCV 2013; IJCV 111(2), 2015. https://openaccess.thecvf.com/content_iccv_2013/html/Gavves_Fine-Grained_Categorization_by_2013_ICCV_paper.html ; https://ivi.fnwi.uva.nl/isis/publications/2015/GavvesIJCV2015/GavvesIJCV2015.pdf
- **[38]** Razavian et al. "CNN Features off-the-shelf: an astounding baseline for recognition." CVPRW 2014. https://arxiv.org/abs/1403.6382

### Deep fine-grained classification

- **[39]** Lin, RoyChowdhury, Maji. Bilinear CNN models for fine-grained visual recognition. ICCV 2015 / TPAMI. https://arxiv.org/abs/1504.07889
- **[40]** Gao et al. Compact Bilinear Pooling. CVPR 2016. https://arxiv.org/abs/1511.06062
- **[41]** Fu, Zheng, Mei. "Look Closer to See Better" (RA-CNN). CVPR 2017. https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf
- **[42]** Li et al. iSQRT-COV. CVPR 2018. https://arxiv.org/abs/1712.01034
- **[43]** Yang et al. NTS-Net. ECCV 2018. https://arxiv.org/abs/1809.00287
- **[44]** Hu et al. WS-DAN. arXiv 2019. https://arxiv.org/abs/1901.09891
- **[45]** Chen et al. "Destruction and Construction Learning for Fine-Grained Image Recognition." CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf
- **[46]** Ding et al. "Selective Sparse Sampling for Fine-Grained Image Recognition." ICCV 2019. https://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_Selective_Sparse_Sampling_for_Fine-Grained_Image_Recognition_ICCV_2019_paper.pdf
- **[47]** Luo et al. Cross-X. ICCV 2019. https://arxiv.org/abs/1909.04412
- **[48]** Chang et al. MC-Loss. IEEE TIP 2020. https://arxiv.org/abs/2002.04264
- **[49]** Zhuang et al. API-Net. AAAI 2020. https://arxiv.org/abs/2002.10191
- **[50]** Gao et al. CIN. AAAI 2020. https://arxiv.org/abs/2003.05235
- **[51]** Du et al. PMG. ECCV 2020. https://arxiv.org/abs/2003.03836
- **[52]** Zhang et al. MMAL-Net. MMM 2021. https://arxiv.org/abs/2003.09150
- **[53]** Rao et al. CAL. ICCV 2021. https://arxiv.org/abs/2108.08728
- **[54]** Behera et al. CAP. AAAI 2021. https://arxiv.org/abs/2101.06635
- **[55]** Bera et al. SR-GNN. IEEE TIP 2022. https://arxiv.org/abs/2209.02109
- **[56]** Liu et al. PMAL / PMD. arXiv 2401.14336, 2024 (venue unverified). https://arxiv.org/abs/2401.14336
- **[57]** Liu et al. CMAL-Net ("Learn from each other to classify better"). Pattern Recognition 2023. Listing: https://hyper.ai/en/papers/learn-from-each-other-to-classify-better
- **[58]** He et al. TransFG. AAAI 2022. https://arxiv.org/abs/2103.07976
- **[59]** Wang et al. FFVT. BMVC 2021. https://arxiv.org/abs/2107.02341
- **[60]** Hu et al. RAMS-Trans. ACM MM 2021. https://arxiv.org/abs/2107.08192
- **[61]** Zhang et al. AF-Trans. ICASSP 2022. https://arxiv.org/abs/2110.01240
- **[62]** Sun et al. SIM-Trans. ACM MM 2022. https://arxiv.org/abs/2208.14607
- **[63]** Zhu et al. DCAL. CVPR 2022. https://arxiv.org/abs/2205.02151
- **[64]** Diao et al. MetaFormer. arXiv 2022. https://arxiv.org/abs/2203.02751
- **[65]** Chou et al. PIM. arXiv 2022. https://arxiv.org/abs/2202.03822
- **[66]** Chou et al. HERBS. arXiv 2023. https://arxiv.org/abs/2303.06442
- **[67]** Xu et al. IELT. IEEE TMM 2023. Code: https://github.com/mobulan/ielt
- **[68]** Xu et al. SCOPE. arXiv 2508.06959, 2025. **WITHDRAWN by the authors (inconsistent experiment records); listed for completeness only, not used as evidence.** https://arxiv.org/abs/2508.06959
- **[69]** Kornblith, Shlens, Le. "Do Better ImageNet Models Transfer Better?" CVPR 2019. https://arxiv.org/abs/1805.08974
- **[70]** Tan, Le. EfficientNet. ICML 2019. https://arxiv.org/abs/1905.11946
- **[71]** Ridnik et al. TResNet. WACV 2021. https://arxiv.org/abs/2003.13630
- **[72]** Ngiam et al. Domain Adaptive Transfer Learning. arXiv 2018. https://arxiv.org/abs/1811.07056
- **[73]** Foret et al. Sharpness-Aware Minimization. ICLR 2021. https://arxiv.org/abs/2010.01412
- **[74]** Touvron et al. DeiT. ICML 2021. https://arxiv.org/abs/2012.12877
- **[75]** Wightman, Touvron, Jégou. "ResNet Strikes Back." arXiv 2021. https://arxiv.org/abs/2110.00476
- **[76]** Kolesnikov et al. Big Transfer (BiT); no Stanford Cars result. https://arxiv.org/abs/1912.11370
- **[77]** Radford et al. CLIP. ICML 2021. https://arxiv.org/abs/2103.00020
- **[78]** Sun et al. EVA-CLIP. arXiv 2023. https://arxiv.org/abs/2303.15389
- **[79]** Oquab et al. DINOv2. TMLR 2024. https://arxiv.org/abs/2304.07193
- **[80]** Siméoni et al. DINOv3. arXiv 2508.10104, 2025. https://arxiv.org/abs/2508.10104
- **[81]** Cherti et al. Reproducible scaling laws for contrastive language-image learning (OpenCLIP). 2022. https://arxiv.org/abs/2212.07143
- **[82]** Jia et al. Visual Prompt Tuning. ECCV 2022. https://arxiv.org/abs/2203.12119
- **[83]** Lian et al. SSF. NeurIPS 2022. https://arxiv.org/abs/2210.08823
- **[84]** V²APT. arXiv 2503.17650, 2025. https://arxiv.org/abs/2503.17650
- **[85]** Zhang et al. Gradient-based Parameter Selection (GPS). CVPR 2024, supplement. https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhang_Gradient-based_Parameter_Selection_CVPR_2024_supplemental.pdf
- **[86]** Frequency-based PEFT. arXiv 2411.19297. https://arxiv.org/abs/2411.19297
- **[87]** Controlled CLIP vs DINO comparison. arXiv 2510.11835, 2025. https://arxiv.org/abs/2510.11835
- **[88]** Stanford Cars fine-grained classification leaderboard mirror (hyper.ai). https://vercel.hyper.ai/en/sota/tasks/fine-grained-image-classification/benchmark/fine-grained-image-classification-on-stanford
- **[89]** timm EVA-02 model card. https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in1k
- **[90]** McGuire et al. "Project: Stanford Cars with fastai v1." fast.ai forum, 2019. https://forums.fast.ai/t/project-stanford-cars-with-fastai-v1/34311?page=2 ; https://forums.fast.ai/t/project-stanford-cars-with-fastai-v1/34311/43
- **[91]** jianshen92. stanford-car-grab-challenge (GitHub). https://www.github.com/jianshen92/stanford-car-grab-challenge

### Vehicle-specific deep recognition

- **[92]** Krause, Jin, Yang, Fei-Fei. "Fine-Grained Recognition without Part Annotations." CVPR 2015. https://openaccess.thecvf.com/content_cvpr_2015/html/Krause_Fine-Grained_Recognition_Without_2015_CVPR_paper.html
- **[93]** Hu, Wang, Li, Shen. Spatially weighted pooling for vehicle recognition. IEEE T-ITS 2017. https://web4.library.adelaide.edu.au/dspace/handle/2440/116890
- **[94]** Valev et al. Systematic comparison of architectures for fine-grained car classification. arXiv 2018. https://arxiv.org/abs/1806.02987
- **[95]** Zeng et al. 3D Perspective Network. AAAI 2020. https://arxiv.org/abs/1903.07916
- **[96]** Yu et al. EP-CNN. IEEE TCSVT 2022. https://mason.gmu.edu/~zyan4/papers/epcnn_tcsvt22.pdf
- **[97]** Manole, Diosan. Multi-task heads for make/model hierarchy. arXiv 2603.01746, 2026. https://arxiv.org/abs/2603.01746
- **[98]** Gunasekaran, Jaiman. Occlusion study on Stanford Cars. arXiv 2304.11779, 2023. https://arxiv.org/abs/2304.11779
- **[99]** Peng, Hoffman, Yu, Saenko. "Fine-to-coarse knowledge transfer for low-res image classification." ICIP 2016. https://arxiv.org/abs/1605.06695
- **[100]** Vázquez-Santiago et al. Open-set vehicle make and model recognition. Math. Comput. Appl. (MDPI) 28(4):80, 2023. https://www.mdpi.com/2297-8747/28/4/80
- **[101]** Veri-Car. arXiv 2411.06864. https://arxiv.org/abs/2411.06864

### Detection

- **[102]** torchvision detection model table. https://docs.pytorch.org/vision/stable/models.html
- **[103]** Ultralytics model documentation (YOLOv8, v9, v10, 11, 12, 26, RT-DETR). https://docs.ultralytics.com/models/yolo26/ ; https://docs.ultralytics.com/models/yolo11/ ; https://docs.ultralytics.com/models/yolo12/ ; https://docs.ultralytics.com/models/yolov10/ ; https://docs.ultralytics.com/models/yolov9/ ; https://docs.ultralytics.com/models/yolov8/ ; https://docs.ultralytics.com/models/rtdetr/
- **[104]** YOLO26 paper, arXiv 2606.03748; review, arXiv 2601.12882. https://arxiv.org/abs/2606.03748 ; https://arxiv.org/abs/2601.12882
- **[105]** Zhao et al. "DETRs Beat YOLOs on Real-time Object Detection" (RT-DETR). CVPR 2024. https://arxiv.org/abs/2304.08069 ; https://github.com/lyuwenyu/RT-DETR
- **[106]** Lv et al. RT-DETRv2. arXiv 2407.17140, 2024. https://arxiv.org/abs/2407.17140
- **[107]** Peng et al. D-FINE. ICLR 2025. https://github.com/Peterande/D-FINE
- **[108]** Roboflow. RF-DETR. arXiv 2511.09554. https://arxiv.org/abs/2511.09554 ; https://github.com/roboflow/rf-detr
- **[109]** Zhang et al. DINO-DETR. ICLR 2023. https://arxiv.org/abs/2203.03605
- **[110]** Zhu et al. Deformable DETR. ICLR 2021. https://arxiv.org/abs/2010.04159
- **[111]** Liu et al. Grounding DINO. ECCV 2024. https://arxiv.org/abs/2303.05499
- **[112]** Cheng et al. YOLO-World. CVPR 2024. https://arxiv.org/abs/2401.17270
- **[113]** COCO detection evaluation. https://cocodataset.org/#detection-eval
- **[114]** Ultralytics export mode documentation. https://docs.ultralytics.com/modes/export/
- **[115]** Ultralytics YOLOv5 tutorial: transfer learning with frozen layers. https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/

### Hybrid classical + deep integration

- **[116]** Su, Maji. Cross-Quality Distillation. BMVC 2017. https://arxiv.org/abs/1604.00433
- **[117]** Dodge, Karam. Understanding how image quality affects deep neural networks. QoMEX 2016. https://arxiv.org/abs/1604.04004
- **[118]** Loh, Chan. ExDark. CVIU 178, 2019. https://arxiv.org/abs/1805.11227
- **[119]** Yin et al. PE-YOLO. arXiv 2307.10953. https://arxiv.org/abs/2307.10953
- **[120]** Low-light two-stage detection study (secondary). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12190514/
- **[121]** Afifi, Brown. White-balance errors and DNNs. ICCV 2019. https://arxiv.org/abs/1912.06960
- **[122]** Geirhos et al. ImageNet-trained CNNs are biased towards texture. ICLR 2019. https://arxiv.org/abs/1811.12231
- **[123]** Nanni, Ghidoni, Brahnam. "Handcrafted vs. non-handcrafted features for computer vision classification." Pattern Recognition 71:158–172, 2017. https://doi.org/10.1016/j.patcog.2017.05.025 ; https://bearworks.missouristate.edu/articles-cob/358
- **[124]** Khoreva et al. "Simple Does It: Weakly Supervised Instance and Semantic Segmentation." CVPR 2017. https://arxiv.org/abs/1603.07485
- **[125]** Selvaraju et al. Grad-CAM. ICCV 2017; IJCV 2020. https://arxiv.org/abs/1610.02391
- **[126]** Wang et al. Score-CAM. CVPRW 2020. https://arxiv.org/abs/1910.01279
- **[127]** Takikawa et al. Gated-SCNN. ICCV 2019. https://arxiv.org/abs/1907.05740
- **[128]** Liu et al. CoordConv. NeurIPS 2018. https://arxiv.org/abs/1807.03247
- **[129]** Edge enhancement for image classification. arXiv 2401.07028. https://arxiv.org/abs/2401.07028

### Robustness

- **[130]** Hendrycks, Dietterich. "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations." ICLR 2019. https://arxiv.org/abs/1903.12261
- **[131]** Yucel, Cinbis, Duygulu. "How robust are discriminatively trained zero-shot learning models?" Image and Vision Computing 2022. https://arxiv.org/abs/2201.10972
- **[132]** Cai, Chen, Qian, Kämäräinen. "Convolutional low-resolution fine-grained classification." arXiv 1703.05393. https://arxiv.org/abs/1703.05393
- **[133]** Vision Mamba Distillation for low-resolution FGVC. arXiv 2411.17980. https://arxiv.org/abs/2411.17980
- **[134]** DeVries, Taylor. Cutout. arXiv 2017. https://arxiv.org/abs/1708.04552
- **[135]** Zhong et al. Random Erasing. AAAI 2020. https://arxiv.org/abs/1708.04896
- **[136]** Singh, Lee. Hide-and-Seek. ICCV 2017. https://arxiv.org/abs/1704.04232
- **[137]** Yun et al. CutMix. ICCV 2019. https://arxiv.org/abs/1905.04899
- **[138]** Shanmugam et al. "Better aggregation in test-time augmentation." ICCV 2021. https://arxiv.org/abs/2011.11156
- **[139]** Wang et al. TENT. ICLR 2021. https://arxiv.org/abs/2006.10726

### Confidence, calibration and open-set recognition

- **[140]** Guo, Pleiss, Sun, Weinberger. "On Calibration of Modern Neural Networks." ICML 2017. https://arxiv.org/abs/1706.04599
- **[141]** Minderer et al. "Revisiting the calibration of modern neural networks." NeurIPS 2021. https://arxiv.org/abs/2106.07998
- **[142]** LeVine et al. Calibration of zero-shot CLIP. arXiv 2303.12748, 2023. https://arxiv.org/abs/2303.12748
- **[143]** Oh et al. "Towards calibrated robust fine-tuning of vision-language models." NeurIPS 2024. https://arxiv.org/abs/2311.01723
- **[144]** Geifman, El-Yaniv. "Selective classification for deep neural networks." NeurIPS 2017. https://arxiv.org/abs/1705.08500
- **[145]** Geifman, El-Yaniv. SelectiveNet. ICML 2019. https://arxiv.org/abs/1901.09192
- **[146]** Geifman, Uziel, El-Yaniv. ICLR 2019. https://arxiv.org/abs/1805.08206
- **[147]** Deng, Krause, Berg, Fei-Fei. "Hedging your bets: Optimizing accuracy-specificity trade-offs in large scale visual recognition." CVPR 2012. https://dblp1.uni-trier.de/rec/conf/cvpr/DengKBF12.html
- **[148]** Goren, Galil, El-Yaniv. "Hierarchical selective classification." arXiv 2405.11533. https://arxiv.org/abs/2405.11533
- **[149]** Mortier et al. Conformal prediction restricted to hierarchy nodes. arXiv 2501.19038. https://arxiv.org/abs/2501.19038
- **[150]** den Hengst et al. "Hierarchical Conformal Classification." arXiv 2508.13288. https://arxiv.org/abs/2508.13288
- **[151]** Liu et al. Energy-based OOD detection. NeurIPS 2020. https://arxiv.org/abs/2010.03759
- **[152]** Wang et al. ViM. CVPR 2022. https://arxiv.org/abs/2203.10807
- **[153]** Sun et al. KNN-OOD. ICML 2022. https://arxiv.org/abs/2204.06507
- **[154]** Vaze, Han, Vedaldi, Zisserman. "Open-Set Recognition: a Good Closed-Set Classifier is All You Need?" ICLR 2022. https://arxiv.org/abs/2110.06207 ; https://ar5iv.labs.arxiv.org/html/2110.06207 ; https://github.com/sgvaze/osr_closed_set_all_you_need
- **[155]** Secondary source for SSB numbers. arXiv 2404.19527. https://arxiv.org/abs/2404.19527

### Evaluation methodology

- **[156]** Anderson, Gwilliam, Teuscher, Merrill, Farrell. "Facing the Hard Problems in FGVC." arXiv 2006.13190, 2020. https://arxiv.org/abs/2006.13190

### Model optimisation and deployment

- **[157]** Krishnamoorthi. "Quantizing deep convolutional networks for efficient inference: A whitepaper." arXiv 1806.08342, 2018. https://arxiv.org/abs/1806.08342
- **[158]** ONNX Runtime quantization documentation. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- **[159]** Luo, Wu. Entropy-based filter pruning. arXiv 1706.05791. https://arxiv.org/abs/1706.05791
- **[160]** ProtoQuant. arXiv 2602.06592. https://arxiv.org/abs/2602.06592
- **[161]** Kozlov et al. NNCF. arXiv 2002.08679. https://arxiv.org/abs/2002.08679
- **[162]** Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." 2015. https://arxiv.org/abs/1503.02531
- **[163]** PyTorch blog. "Practical Quantization in PyTorch." https://pytorch.org/blog/quantization-in-practice/
- **[164]** PyTorch quantization documentation (torchao migration notice). https://docs.pytorch.org/docs/stable/quantization.html
- **[165]** Hugging Face Spaces documentation (overview, GPUs, ZeroGPU). https://huggingface.co/docs/hub/spaces-overview ; https://huggingface.co/docs/hub/spaces-gpus ; https://huggingface.co/docs/hub/spaces-zerogpu

### Works cited in the source reports without a recorded URL

- **[166]** Rother, Kolmogorov, Blake. GrabCut. SIGGRAPH 2004.
- **[167]** Zuiderveld. Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV, 1994.
- **[168]** Hou, Zhang. Saliency detection: a spectral residual approach. CVPR 2007.
- **[169]** Buchsbaum. Gray-world colour constancy. 1980.
- **[170]** Jobson et al. Multiscale Retinex with colour restoration (MSRCR). IEEE TIP 1997.
- **[171]** Hendrycks, Gimpel. Maximum softmax probability baseline for OOD detection. ICLR 2017.
- **[172]** Liang et al. ODIN. ICLR 2018.
- **[173]** Lee et al. Mahalanobis-distance OOD detection. NeurIPS 2018.
- **[174]** Jacob et al. Quantization and training for integer-arithmetic-only inference. CVPR 2018.
- **[175]** Han et al. Deep Compression. ICLR 2016.
- **[176]** Li et al. Pruning filters for efficient ConvNets. ICLR 2017.
- **[177]** Chen et al. Multi-task viewpoint and class learning for vehicles. IEEE Access 2020.
- **[178]** FineR (ICLR 2024) and Enriched-FineR (2025): training-free LVLM-based FGVC.
