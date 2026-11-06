# VMMR literature survey for a Stanford Cars hackathon (written 2026-11-06)

**How to read the numbers.** "Primary" means I read the figure in the paper's own text (PDF or arXiv HTML). "Secondary" means it comes from a survey or abstract page. "(unverified)" means I could not check it against a primary source.

The main secondary source is the survey by Ni & Huttunen, *J. Signal Processing Systems* 2020 ([arXiv 2006.16400](https://arxiv.org/abs/2006.16400)). I extracted its Tables 1, 3 and 4 directly from the PDF.

Several primary sources could not be fetched: ai.stanford.edu refused the connection, and openaccess.thecvf.com, HAL and core.ac.uk returned 403. Some numbers are therefore secondary.

---

## 1. Datasets

| Dataset | Size / classes | Labels | Availability |
|---|---|---|---|
| **Stanford Cars ("car-197", later 196)** — Krause et al., 3dRR@ICCVW 2013 | 16,185 images, 197 classes in the paper (196 released); 8,144 train / 8,041 test | make-model-body-year string, one 2D bbox per image. No viewpoint or part labels. | **The original URL is dead.** torchvision says "the dataset isn't available online anymore" and its `download=True` no longer downloads. Use Kaggle or Hugging Face mirrors. |
| **CompCars** — Yang, Luo, Loy, Tang, CVPR 2015 (journal version on arXiv 1506.08959) | Web-nature part: 136,727 whole-car images plus 27,618 part images, 163 makes, 1,716 models. Surveillance part: 44,481 front-view images, 281 models. 208,826 images in total (primary). | Web part: hierarchy make → model → year, 5 viewpoints (F/R/S/FS/RS), 8 parts, 5 attributes. Surveillance part: bbox, model, colour. | On request from CUHK; research-only licence. |
| **BoxCars21k** — Sochor, Herout, Havel, CVPR 2016 | 21,250 vehicles, 63,750 images, 27 makes, 148 classes (make+model+submodel+model year) (secondary) | 3D bounding box, surveillance views | Downloadable from medusa.fit.vutbr.cz |
| **BoxCars116k** — Sochor, Špaňhel, Herout, IEEE T-ITS 2018/19 | 27,496 vehicles, 116,286 images, 45 makes, **693 classes (make+model+submodel+model year)**, 137 cameras (primary) | 3D bbox, foreground mask, tracks. "Hard" split keeps model years (107 classes); "medium" split merges them. | Same server; code at github.com/JakubSochor/BoxCars |
| **VMMRdb** — Tafazzoli, Frigui, Nishiyama, CVPRW 2017 | 291,752 images, 9,170 classes, model years 1950–2016 (GitHub README) | make, model, year. Unaligned, crowd-sourced (Craigslist-style) photos. Experiments used the VMMRdb-3036 subset. | Dropbox ZIP; the repo is MIT-licensed |
| **Gebru et al. "visual census"** — AAAI 2017 | 712,430 images, 2,657 classes, "nearly all car types produced after 1990" (primary) | make, model, year, trim, grouped into "sets of indistinguishable classes" | Not publicly released, as far as I could find |
| **Car-1000** — Hu, Li, Yan, Shao, Luo; FGVC11 workshop at CVPR 2024; arXiv 2503.12385 (v2, Feb 2026) | 140,267 images, 1,000 models, 166 automakers, "1960s to the latest 2020s"; 60/20/20 split (primary) | 3-tier hierarchy: 7 types → 21 sub-types → 1,000 models. No year or generation labels are described. | github.com/toggle1995/Car-1000 |
| **CaMiT** — Lin, Popescu, Le Borgne et al., NeurIPS 2025 Datasets & Benchmarks track, arXiv 2510.17626 | 787K labelled images (190 models, 2007–2023) plus 5.1M unlabelled (2005–2023) | Time-stamped images, built for studying temporal drift | Released by the authors (CEA List) |
| **VehicleID** — Liu et al., CVPR 2016 (re-ID, for contrast) | 221,763 images of 26,267 vehicles; only a subset has model labels (47,558 + 42,638 images) | identity, partial model labels | On request |
| **VeRi-776** — Liu et al., 2016 (re-ID, for contrast) | About 50k images (49,357 per one source), 776 vehicles, 20 cameras | identity, colour, type, brand | On request |

**Re-ID contrast.** VehicleID and VeRi label vehicle *identity* across cameras, not model taxonomy. They are useful only for metric-learning pretraining or as a source of surveillance-domain images.

**Older small sets** (counts from the Ni survey, Table 1):
- Petrovic–Cootes 2004: 1,132 images, 77 classes.
- Clady 2008: 1,121 images, 50 classes.
- car-types (Stark 2012): 1,904 images, 14 classes.
- BMW-10: 512 images, 10 classes.
- FG3DCar (Lin 2014): 300 images, 30 classes.

**Key primary fact about how Stanford Cars classes were defined.** Krause et al. wrote that "models of cars do not have a different appearance from year to year". They crawled every car type made since 1990, then de-duplicated example images with perceptual hashing "to a limited number of… visually distinct classes", and sampled 197 of those.

So the year in a Stanford Cars label is a *representative year for a visually distinct appearance*, not a precise model year. Year errors should not be scored as "off by N years".

---

## 2. Classical VMMR (pre-deep-learning)

- **Petrovic & Cootes, BMVC 2004**, "Analysis of features for rigid structure vehicle type recognition."
  - Method: a frontal reference patch is located relative to the licence plate. Raw, Harris and gradient-based structure features are compared; gradient features worked best. Nearest-neighbour classification.
  - Result: over 93% recognition and under 5.6% verification EER, on more than 1,000 images of 77 classes (primary abstract).
- **Clady et al. 2008:** oriented-contour-point models from frontal images; 93.1% on 50 classes (secondary, Ni survey).
- **Pearce & Pears, AVSS 2011**, "Automatic make and model recognition from frontal images of cars."
  - Method: a Harris-corner/LNHS frontal descriptor with Naive Bayes.
  - Result: roughly 96% on 262 images and 74 classes (unverified; this comes from another paper's summary table).
- **Psyllos, Anagnostopoulos & Kayafas, IEEE T-ITS 2010:** logo recognition with an enhanced SIFT matching scheme that merges SIFT points across logo samples. 1,200 logos from 10 makes. Reported as 91% (enhanced) vs 88% (standard SIFT), but another paper cites 97%, so treat both as (unverified).
- **Hsieh, Chen & Chen, IEEE T-ITS 2014**, "Symmetrical SURF":
  - A mirrored SURF descriptor finds symmetric matching pairs, which locates the vehicle without background subtraction.
  - Classification uses a grid-wise ensemble of HOG and SURF SVMs.
  - A sparse-representation extension appeared in *Pattern Recognition* 2015.
  - Accuracy figures could not be verified.
- **Llorca et al., ITSC 2014:** recognises rear-view model and trim emblems using HOG features and linear SVMs, assisted by licence-plate localisation. 8 makes, 28 models, **52 classes when generations are counted**. 93.75% overall accuracy (secondary, from a search snippet of the paper PDF).
  - This is one of the few classical papers that explicitly separates *generations*.
- **Boonsim & Prakoonwit, *Pattern Analysis & Applications* 2017:** night-time recognition from rear taillight and licence-plate geometry and shape features. A majority vote over SVM, decision tree and kNN. About 93% (secondary, consistent across sources; the paper is open access).
- **Stark et al., BMVC 2012:** structDPM, a multi-class DPM, plus a part-bank method, on car-types (14 classes): structDPM 93.5% (primary, quoted in Krause 2013).
- **Krause et al., 3dRR 2013** (primary numbers):
  - Method: fits 3D CAD geometry, rectifies patches, and lifts SPM and BubbleBank pooling into 3D (SPM-3D, BB-3D).
  - car-types: BB-3D-G 94.5%.
  - BMW-10: BB-3D-G 66.1% vs BB 58.7% and HOG 28.3%.
  - car-197: **2D SPM 69.5% beat BB-3D-G 67.6%**; BB 63.6%, SPM-3D-L 65.6%.
  - Takeaway: 3D pooling helped on small data but not once there was more data.
- **Lin, Morariu, Hsu & Davis, ECCV 2014**, "Jointly optimizing 3D model fitting and fine-grained classification":
  - Method: DPM gives coarse parts, landmarks are regressed, a 3D model is fitted, and features at the landmarks feed an SVM. The predicted class is fed back to refine the fit.
  - Result: FG3DCar (300 images, 30 classes) 90.0% (secondary).
- **FGComp 2013 (cars track = Stanford Cars):** Inria-Xerox won with late fusion of two Fisher-vector systems, using descriptor filtering and spatial-coordinate coding (Gosselin et al., *Pattern Recognition Letters* 2014; primary claim of winning).
  - Their exact cars accuracy is (unverified). I recall about 82.7–87.8% but could not confirm it.
  - A different FV setup reported elsewhere gives 83.7% (secondary).

---

## 3. Deep VMMR

- **CompCars baselines** (primary, Yang 2015):
  - 431 models, all views: AlexNet 81.9%, Overfeat 87.9%, GoogLeNet 91.2% top-1.
  - Per-view models (Overfeat, half train / half test), model level: F 52.4%, R 43.1%, S 42.8%, FS 56.3%, RS 59.8%, all-view 76.7%.
  - Same per-view models, make level: 71.0% / 52.1% / 50.7% / 68.0% / 65.6% / 82.9%.
  - Parts alone: taillight was best at 68.4%, partly because the model name is printed near it. Voting over all 8 parts: 80.8%.
  - Surveillance front views (rain, fog, night): GoogLeNet 98.4%.
  - **CompCars merges all production years of a model into one class** ("splitting them would make the task harder"; not tested).
- **Krause et al., CVPR 2015**, "Fine-grained recognition without part annotations": co-segmentation, alignment and discriminative parts on a fine-tuned VGG. 92.8% on cars-196 with ground-truth bbox (secondary, from the supplement table).
- **Hu, Wang, Li & Shen, IEEE T-ITS 2017**, spatially weighted pooling: Stanford Cars 92.6% → 93.1% (ResNet-101); CompCars 97.6% (secondary).
- **Xiang et al. 2019**, global topology constraint network: Stanford Cars 94.3%, CompCars 98.5% (secondary, Ni survey).
- **Valev et al., arXiv 2018:** systematic comparison of architectures on Stanford Cars; best 94.6%. They note that "vehicles of different types… can look more similar than vehicles of the same type" when colour and viewpoint match.
- **BoxCars, CVPR 2016:**
  - Inputs added to the CNN: the image "unpacked" via the 3D box, a rasterised low-resolution shape, and the 3D orientation.
  - Accuracy 77.2% → 83.2% (26% error reduction); verification AP 0.378 → 0.785.
- **BoxCars116k, T-ITS 2018** (primary):
  - Adds 3D-box estimation from vanishing points (for when calibration is unavailable), colour augmentation, and random-noise "ImageDrop".
  - Train and test cameras are disjoint, so test viewpoints are unseen.
  - Hard split, single image / whole track: ResNet152 77.68/86.20 → 83.74/91.71 with ground-truth 3D box, 80.87/90.93 with estimated box.
  - Generic fine-grained baselines on 116k (secondary): BCNN 69.6%, Simon & Rodner 75.9%, Gao 70.6%, Sochor 84.1%.
- **Zeng et al., AAAI 2020**, 3D Perspective Network: a 2D global feature fused with a 3D-box feature; vanishing-point loss plus RoI perspective layers. On BoxCars: 81.2% (secondary).
- **Viewpoint-aware methods:**
  - EP-CNN, Yu et al., IEEE TCSVT 2022: a pose-estimation sub-network plus pose-conditioned SE blocks; evaluated on CompCars web and Stanford Cars.
  - Chen et al., IEEE Access 2020: multi-task learning of viewpoint and class; BoxCars116k and CompCars.
  - Accuracy figures for both are (unverified).
- **Hierarchy (make → model):** Manole & Diosan, arXiv 2603.01746 (Mar 2026) compare parallel vs cascaded multi-task heads on Stanford Cars and CompCars. Multi-task helps CNNs "in almost all scenarios"; numbers not checked.
- **Large-scale make/model/year:** Gebru et al. 2017, 2,657 classes (primary):
  - 31.27% on ground-truth boxes, 33.27% on DPM detections.
  - Make 66.38%, model 51.83%.
  - Errors cluster between sedan/coupe and between extended-cab/crew-cab trucks.

---

## 4. Challenges, with the evidence for each

**(a) Year / generation and facelifts**
- Stanford Cars avoids year-level splits unless appearance differs (Krause 2013). CompCars merges years.
- BoxCars116k is the main direct test: its hard (with model year) and medium (without) splits gave "almost identical" accuracy.
- Gebru et al. say differences are "extremely subtle at the year and trim level".
- Llorca 2014 separates generations using rear emblems.
- CaMiT reports that "accuracy declines when models are tested across years".

**(b) Badge-engineered twins**
- I found **no paper that measures rebadge confusion**. Valev 2018 and Gebru 2017 give only general similarity arguments.
- So the twin analysis in §6 is a gap the hackathon could fill.

**(c) Viewpoint**
- CompCars per-view results: front and front-side views are better than rear and side (model level 52.4% vs 42.8%).
- BoxCars: 3D-box unpacking gives up to +12 percentage points and up to 50% error reduction. Testing on unseen cameras is the realistic setting.
- Krause 2013: 3D lifting only helped at small scale.

**(d) Occlusion**
- Gunasekaran & Jaiman, arXiv 2304.11779 (2023): synthetic occlusions added to Stanford Cars; VGG19, ResNet-50/101, GoogLeNet and DenseNet-121 tested. Numbers not checked.
- BoxCars' ImageDrop augmentation targets occlusion.

**(e) Illumination / night**
- CompCars surveillance part includes night, rain and fog: about 98% because the views are fixed frontal ones.
- Boonsim 2017: taillight features at night, about 93%.

**(f) Resolution**
- Peng, Hoffman, Yu & Saenko, ICIP 2016: "staged training", i.e. fine-tuning on progressively downsampled cars and birds to transfer high-resolution knowledge to low-resolution tests.
- Gebru downsampled product shots to match Street View resolution.
- BoxCars notes that surveillance images are "usually small".

**(g) Open-set and temporal drift**
- Car-1000 says Stanford Cars is outdated: only models produced earlier than 2013.
- Vázquez-Santiago et al., *MCA* (MDPI) 2023: VMMR treated as open-set recognition, with new-class discovery by GMM plus clustering on a VMMRdb subset.
- Veri-Car, arXiv 2411.06864: retrieval with a hierarchical multi-similarity loss, aimed at unseen models.
- CaMiT, NeurIPS 2025: time-incremental pretraining and time-incremental classifier-only updates both reduce drift.

---

## 5. Hybrid classical + deep approaches for vehicles

- **Geometry priors fed to CNNs** are the strongest evidence:
  - BoxCars (unpacked image, rasterised shape, orientation): verified gains.
  - Zeng 3DPN (vanishing-point geometric constraint).
  - Lin 2014 (3D fitting coupled with classification).
- **Symmetry:** Hsieh's symmetrical SURF is classical only; I found no deep symmetry-prior VMMR paper.
- **HOG + CNN feature fusion:** only low-tier work turned up, for example HOG + EfficientNet features with an SVM on Nigerian-road data (MYJAS 2023), and a 2016 *Sensors* local-tiled CNN paper.
- **Edge maps as an extra CNN input channel:** **I found no paper doing this for VMMR.** This looks like an open niche, consistent with the evidence that low-resolution shape input helps (BoxCars).

---

## 6. Stanford Cars class-list analysis

Source: the 196 names in `cars_meta` order, taken from the clip-benchmark `classnames.txt` mirror. Index 1 is AM General Hummer SUV 2000; torchvision uses the same order with 0-based labels. I computed the groupings below with a script.

**Label-format pitfalls**
- The year is always the last token.
- Body type is sometimes missing, e.g. "Acura TL Type-S 2008", "Chevrolet Corvette ZR1 2012", "Buick Regal GS 2012", "Jaguar XK XKR 2012", "FIAT 500 Abarth 2012", "Chrysler 300 SRT-8 2010".
- "Bentley Continental Supersports Conv. Convertible 2012" contains an abbreviation and a redundant word.

### (i) Same make + model, differing only in year — 7 pairs (exact match after removing the year)

Generation notes are my automotive domain knowledge, not taken from papers, and should be checked.

| Pair (1-based index) | Generation relationship | Expected difficulty |
|---|---|---|
| Audi S4 Sedan 2012 (23) vs 2007 (24) | B8 vs B7 | moderate |
| Bentley Continental GT Coupe 2012 (42) vs 2007 (43) | 2nd vs 1st generation, evolutionary styling | hard |
| Dodge Caliber Wagon 2012 (83) vs 2007 (84) | **same generation (2007–12)**, only a mild refresh | **near label noise; hardest** |
| Dodge Durango SUV 2012 (94) vs 2007 (95) | 3rd vs 2nd generation | easy–moderate |
| Ford F-150 Regular Cab 2012 (113) vs 2007 (114) | 12th vs 11th generation | moderate |
| Honda Odyssey Minivan 2012 (126) vs 2007 (127) | 4th vs 3rd generation | moderate |
| Volkswagen Golf Hatchback 2012 (190) vs 1991 (191) | Mk6 vs Mk2 | easy |

**Near-year groups where the model name also differs:**
- Silverado 1500 Classic Extended Cab 2007 (70; older-generation carry-over) vs Silverado 1500 Extended Cab 2012 (74).
- Dodge Charger SRT-8 2009 (97) vs Charger Sedan 2012 (96).
- Chevrolet Malibu Sedan 2007 (73) vs Malibu Hybrid Sedan 2010 (67).
- BMW 6 Series Convertible 2007 (31) vs M6 Convertible 2010 (36): same E64 body.
- BMW M5 Sedan 2010 (35) vs ActiveHybrid 5 Sedan 2012 (26).
- Corvette Ron Fellows Z06 2007 (57) vs Corvette Convertible / ZR1 2012 (55, 56): all C6 generation.
- Dodge Ram 3500 Quad Cab 2009 (87) vs Crew Cab 2010 (86).
- Dodge Dakota Club Cab 2007 (91) vs Crew Cab 2010 (90).
- Hyundai Elantra Sedan 2007 (135) vs Elantra Touring 2012 (139).
- The Bentley Continental family: 39, 42, 43, 44.

### (ii) Rebadged and platform twins inside the 196 classes (domain knowledge)

**True rebadges (near-identical body):**
- Chevrolet Express Van 2007 (71) and Express Cargo Van 2007 (64) ↔ GMC Savana Van 2012 (119).
- Dodge Sprinter Cargo Van 2009 (88) ↔ Mercedes-Benz Sprinter Van 2012 (166): same NCV3 body.
- Chevrolet Tahoe Hybrid SUV 2012 (62) ↔ GMC Yukon Hybrid SUV 2012 (120).
- Chrysler Aspen SUV 2009 (76) ↔ Dodge Durango SUV 2007 (95).
- Buick Rainier 2007 (48) ↔ Chevrolet TrailBlazer SS 2009 (68) ↔ Isuzu Ascender 2008 (143): GMT360 platform.
- Chrysler Town and Country 2012 (78) ↔ Ram C/V Cargo Van 2012 (174): same minivan, different fascia.

**Platform siblings with distinct styling:**
- GM Lambda platform: Chevrolet Traverse (58), GMC Acadia (121), Buick Enclave (50).
- Cadillac Escalade EXT 2007 (53) ↔ Chevrolet Avalanche 2012 (65).
- GMC Canyon (122) ↔ HUMMER H3T (124).
- Jeep Patriot (145), Jeep Compass (149), Dodge Caliber (83, 84).
- Chrysler LX platform: Charger (96, 97), 300 SRT-8 (79), Magnum (92), Challenger (93).

**Trim twins:**
- Audi A5 Coupe / S5 Coupe (13, 22).
- Audi TT Hatchback / TTS / TT RS (19, 14, 25).
- Hyundai Sonata / Sonata Hybrid (138, 134).
- The Chevrolet Silverado cluster (54, 69, 70, 74, 75).

**Body-style twins** (same make, model and year; differ in roof or body):
- Aston Martin V8 Vantage (8/9) and Virage (10/11).
- Bugatti Veyron (45/46).
- Ferrari 458 (103/104).
- Spyker C8 (179/180).
- Audi S5 (21/22), BMW 1 Series (27/28), BMW 3 Series sedan/wagon (29/30).
- Honda Accord (128/129), Suzuki SX4 (183/184), Audi 100 (17/18).
- Rolls-Royce Phantom (175/177).

**Twins outside the set (open-set risk):** Geo Metro ≈ Suzuki Swift, Eagle Talon ≈ Mitsubishi Eclipse, Plymouth Neon ≈ Dodge Neon, Mazda Tribute ≈ Ford Escape. Real photos of the partner car will be confidently mislabelled.

### (iii) The 49 makes (computed, with class counts)

AM General (1), Acura (6), Aston Martin (4), Audi (14), BMW (13), Bentley (6), Bugatti (2), Buick (4), Cadillac (3), Chevrolet (22), Chrysler (6), Daewoo (1), Dodge (15), Eagle (1), FIAT (2), Ferrari (4), Fisker (1), Ford (12), GMC (5), Geo (1), HUMMER (2), Honda (4), Hyundai (11), Infiniti (2), Isuzu (1), Jaguar (1), Jeep (5), Lamborghini (4), Land Rover (2), Lincoln (1), MINI (1), Maybach (1), Mazda (1), McLaren (1), Mercedes-Benz (6), Mitsubishi (1), Nissan (4), Plymouth (1), Porsche (1), Ram (1), Rolls-Royce (3), Scion (1), Spyker (2), Suzuki (4), Tesla (1), Toyota (4), Volkswagen (3), Volvo (3), smart (1).

**Multi-word makes:** "AM General", "Aston Martin" and "Land Rover" contain spaces. "Mercedes-Benz" and "Rolls-Royce" are hyphenated. Splitting on the first space gives 52 "makes", which is wrong.

**Make-label quirks that will affect make-level scoring:**
- Casing: "FIAT", "MINI", "HUMMER", "smart".
- Hummer is split: "AM General Hummer SUV 2000" sits under AM General, while the H2 and H3T sit under "HUMMER".
- Ram is split: "Ram C/V" is under make "Ram", while "Dodge Ram Pickup 3500" is under Dodge.

---

## Sources

- Krause et al. 2013, 3D Object Representations (read in full): https://www.cs.princeton.edu/~jiadeng/paper/KrauseStarkDengFei-Fei_3DRR13.pdf ; https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/html/Krause_3D_Object_Representations_2013_ICCV_paper.html
- torchvision StanfordCars source: https://github.com/pytorch/vision/blob/main/torchvision/datasets/stanford_cars.py
- Class names mirror: https://huggingface.co/datasets/clip-benchmark/wds_cars
- CompCars: https://arxiv.org/abs/1506.08959 ; https://openaccess.thecvf.com/content_cvpr_2015/html/Yang_A_Large-Scale_Car_2015_CVPR_paper.html
- BoxCars 2016: https://openaccess.thecvf.com/content_cvpr_2016/html/Sochor_BoxCars_3D_Boxes_CVPR_2016_paper.html
- BoxCars116k: https://arxiv.org/abs/1703.00686 ; https://github.com/JakubSochor/BoxCars
- VMMRdb: https://openaccess.thecvf.com/content_cvpr_2017_workshops/w9/html/Tafazzoli_A_Large_and_CVPR_2017_paper.html ; https://github.com/faezetta/VMMRdb
- Car-1000: https://arxiv.org/abs/2503.12385 ; https://github.com/toggle1995/Car-1000
- CaMiT: https://arxiv.org/abs/2510.17626
- Gebru et al. 2017: https://ar5iv.labs.arxiv.org/html/1709.02480 ; https://ojs.aaai.org/index.php/AAAI/article/view/11174
- Ni & Huttunen survey: https://arxiv.org/abs/2006.16400
- Petrovic & Cootes 2004: https://bmva-archive.org.uk/bmvc/2004/abstracts/abstract_165.html
- Stark et al. 2012: https://bmva-archive.org.uk/bmvc/2012/BMVC/paper036/index.html
- Lin et al. ECCV 2014: https://www.doi.org/10.1007/978-3-319-10593-2_31
- Psyllos et al. 2010: https://www.scilit.net/publications/fa0bd52b7d0c232f04b0a241bd43a868
- Hsieh et al. 2014: https://scholars.ntou.edu.tw/handle/123456789/24744
- Llorca et al. 2014: https://invett.aut.uah.es/sotelo/ITSC2014-VehicleLogo.pdf
- Boonsim & Prakoonwit 2017: https://link.springer.com/article/10.1007/s10044-016-0559-6
- Pearce & Pears 2011: https://www-users.york.ac.uk/~np7/research/bib/avss11pearce-bib.html
- Gosselin et al. (FGComp 2013): https://hal.archives-ouvertes.fr/hal-01056223
- Krause et al. CVPR 2015: https://openaccess.thecvf.com/content_cvpr_2015/html/Krause_Fine-Grained_Recognition_Without_2015_CVPR_paper.html
- Hu et al. 2017: https://web4.library.adelaide.edu.au/dspace/handle/2440/116890
- Valev et al. 2018: https://arxiv.org/abs/1806.02987
- Zeng et al. AAAI 2020: https://arxiv.org/abs/1903.07916
- EP-CNN (TCSVT 2022): https://mason.gmu.edu/~zyan4/papers/epcnn_tcsvt22.pdf
- Manole & Diosan 2026: https://arxiv.org/abs/2603.01746
- Occlusion study: https://arxiv.org/abs/2304.11779
- Low-resolution staged training: https://arxiv.org/abs/1605.06695
- Open-set VMMR: https://www.mdpi.com/2297-8747/28/4/80
- Veri-Car: https://arxiv.org/abs/2411.06864
- Stanford Cars colour-refined labels: https://github.com/morrisfl/stanford_cars_refined
- VehicleID / VeRi collection: https://github.com/layumi/Vehicle_reID-Collection

The class list can be regenerated from `data/stanford_cars/devkit/cars_meta.mat`.
