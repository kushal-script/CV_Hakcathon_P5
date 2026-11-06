# 02 — Critique of `01_synthesis.md` (stage 3: research → synthesise → **critique** → revise → final)

*Written 2026-11-06 by a reviewer who did not write the proposal. It is red-teamed from two viewpoints: a sceptical senior CV researcher, and a hackathon judge applying the P5 rubric. Section references such as "§3.2" or "E07" point to `01_synthesis.md`.*

**Facts I re-checked for this critique** (each tagged **[V]** where it is used below):
- **TResNet paper** (arXiv 2003.13630, read from the PDF):
  - Stanford Cars 96.0 is TResNet-L at 368 px, starting from ImageNet pre-trained checkpoints.
  - It was "fine-tuned … for 80 epochs using 1-cycle policy".
  - The loss was cross-entropy plus a "weighted triplet loss with soft-margin".
  - TResNet-L trains at 345 img/s on a V100 at 224 px with mixed precision.
- **timm model cards**:
  - `tresnet_l.miil_in1k_448`: 56.0 M params, **43.6 GMACs at 448**, Apache-2.0. Its BatchNorm weights were remapped from Inplace-ABN, so `inplace_abn` is not needed.
  - `timm/models/tresnet.py` uses SpaceToDepth, BlurPool2d anti-aliasing, SE and LeakyReLU.
  - `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`: 87.1 M params, **107.1 GMACs**.
- **HF pricing page**: PRO costs **$9/month** and lists "Host ZeroGPU, Gradio & Docker Spaces" as a PRO feature. CPU Basic (2 vCPU / 16 GB) is listed as free hardware.
- **Kaggle limits**: ~30 GPU-h/week and 12 h sessions are corroborated only by third-party pages. One source says ~9 h. **Not confirmed officially.**
- **Hackathon PDF** (re-extracted with `pdftotext -layout`):
  - The rubric lists only **five** criteria, whose "Max" column sums to 100. The prose says "the other five criteria carry 15", which does not add up.
  - **No design-document template** appears anywhere in the PDF.

---

## (a) Verdict

The proposal is unusually honest and well evidenced. Its separation of evidence grades, its "Weak" labels, the val/test hygiene, the back-off design and the risk register are better than most hackathon plans. But it is a research programme of **12 modules and 22 experiments**, aimed at a rubric that pays most for **a live URL that works on a judge's phone, a clear 3-minute demo, and a team that can explain it**.

Four defects matter before anything else:
1. **The URL is unresolved.** A $9 fix exists and is verified.
2. **The bake-off includes an undeployable candidate.** EVA-02-B@448 is ~214 GFLOPs per forward pass, which is >3.5× the default.
3. **The headline "verified 96.0" is tied to a recipe the plan does not run.** It needs 80 epochs plus triplet loss; the plan trains 30 epochs with cross-entropy.
4. **The classical-vs-deep-vs-hybrid comparison is confounded.** The "hybrid" gain will come mostly from YOLO cropping, which is deep, and a viva question will expose this.

Several demo paths (no car, multiple cars, Indian-market modern cars, phone upload time) are under-designed relative to how likely they are on judging day.

**Recommendation: keep the architecture's spine and cut the scope by about 40%.**
- **Spine to keep:** YOLO → crop → strong CNN → calibrated hierarchical back-off, plus visible classical modules.
- **Make the URL live with a skeleton by ~H6.**
- **Commit to one backbone at H0.**

---

## (b) Severity-ranked issues

| ID | Sev. | § | Issue | Evidence | Concrete fix |
|---|---|---|---|---|---|
| C1 | **Critical** | §7, R1, Q1 | **No host has been chosen**, and missing the URL caps UI/UX at 0. The fallback "H2 ZeroGPU, same code, `@spaces.GPU` wrapper only" does not work. ZeroGPU only helps torch-CUDA code, and the deployed image is deliberately PyTorch-free ONNX-CPU. Also, the HF pricing page now lists *hosting* ZeroGPU Spaces as a PRO feature, which conflicts with [05]'s "free accounts ≥30 days". | Rubric deductions; HF pricing page [V] | **Decide today: one member buys HF PRO ($9) [V].** Deploy a Docker CPU-Basic Space. Delete H2 from the plan, or rewrite it as "a torch build for ZeroGPU" (extra work, not recommended). Give judges the direct `*.hf.space` URL, which is lighter on phones than the huggingface.co wrapper page. Pre-warm before judging, because CPU Basic still sleeps after 48 h idle [05]. |
| C2 | **Critical** | §8 timeline | **No minimum viable demo exists before H8, and it depends on the bake-off checkpoint.** If E07 or the host slips, the team arrives at H18 with sophisticated parts and no working URL. | Rubric: Demo 25 + UI 15 = 40% of points depend on the live path | **MVP by H6** containing: COCO YOLO26s zero-shot; *any* fine-tuned classifier (even a 3-epoch TResNet-L); box + label + confidence; four classical panels (CLAHE, Canny, contour, HSV histogram); error states. Keep `main` deployable at all times and redeploy after every milestone. |
| M1 | Major | §1, E07, §3.4 | **EVA-02-B@448 cannot meet the latency budget, so the bake-off cannot change the decision.** Its forward pass is 107.1 GMACs ≈ 214 GFLOPs [V], 3.6× TResNet-L@368. With flip TTA on 2 vCPU FP32 that is ≈ 5–10 s (est.), versus the §1 claim of "~1.5–4 s". Training cost is also under-estimated: ~12–20 img/s on a T4 with checkpointing and no FlashAttention on sm75 (est.) gives ≈ 1–1.7 h for 10 epochs, and 3–5 h for 30 epochs if it wins. The decision rule ("higher val Top-1 wins") ignores latency. | timm card [V]; §7 budget of 0.8–2.0 s for M6 | **Drop EVA and the E07 bake-off.** Commit to TResNet-L at H0. The 1.5–2.5 GPU-h saved goes into a longer TResNet schedule (M2). If a second architecture is wanted for the viva, the CLIP-L linear-probe row (E08) already answers "why not a ViT/foundation model?" |
| M2 | Major | §1, §3.4, §10 | **The "verified ≥96%" default rests on a recipe the plan does not run.** The paper used 80 epochs, 1-cycle, CE + weighted soft-margin triplet, and an unspecified (224 or 448) IN-1k checkpoint [V]. The plan uses 30 epochs with CE + label smoothing. That puts us in the "plain modern backbone" regime [05 §2.1], so the honest expectation is ~93.5–95% (est.), not 96. §10 admits the recipe gap but omits the 80 epochs, and still uses 96.0 as the tie-break argument. | TResNet §4.1 [V]; [05 §2.1] | Restate the expectation as 94–95 (est.). Start from `tresnet_l.miil_in1k_448` [V] (closer to 368, BatchNorm weights). Run **40–60 epochs** if quota allows: 345 img/s on V100 @224 [V] → ~35–45 img/s on a T4 @368 (est.) → ~3 min/epoch → 2–3 h. Treat batch-hard triplet as one optional ablation arm, not a dependency. |
| M3 | Major | §1 table, §7 | **FLOPs and MACs are confused.** "TResNet-L@368 ≈ 29 GFLOPs" is really ≈ 29 GMACs ≈ **59 GFLOPs** (43.6 GMACs @448 [V] × (368/448)²). M6 FP32 with flip is then ≈ 1.2–2.4 s on 2 vCPU (est.). INT8 speed-up on non-VNNI CPUs may be small or negative [05 §2.3]. | timm card [V] | At H6, **export an untrained TResNet-L@368 to ONNX FP32/INT8 and time it on the real host.** If the flip pair takes > 1.5 s, either drop flip TTA from the live path (keep it offline) or serve at 320 px (TResNet is trained multi-res [V]). Record the result in E18. |
| M4 | Major | §6.1, E22, Adv-4 | **The classical / deep / hybrid comparison is confounded.** "Deep = full image, no M2–M4" and "hybrid = full pipeline" differ by (i) the YOLO crop, which is deep, (ii) crop-trained versus full-image weights, and (iii) the classical modules. Any Δ will mostly be the crop (E10), yet it will be credited to "hybrid". A judge asking "what does the classical part add?" gets "≈0" (§3.7 expects β→0) with no clean row to show. The live "deep" prediction would also need a *second* full-image model in the container. | §3.7 expectation; rubric viva band | Define three rows: **Classical** = saliency/GrabCut ROI + handcrafted features + SVM, fully classical, with a "+YOLO crop" variant labelled hybrid-ROI. **Deep** = YOLO crop → student, no M2/M4/M8/back-off. **Hybrid** = Deep + M2 + M4 + M8 fusion + back-off. Report hybrid − deep on clean *and* on corrupted, dark and no-detection subsets, where the classical modules can plausibly help. Keep full-image vs crop as a separate ablation (E10). Live "deep" = z_deep, which costs nothing extra. |
| M5 | Major | §3.2, E04 | **The fine-tuned 1-class detector may fail on demo images.** Stanford Cars has exactly one box per image, with no negatives and unlabelled background cars. Fine-tuning teaches the detector to *suppress secondary cars* and never shows it car-free scenes. Multi-car street photos and no-car images (both likely at judging) may get worse, while val mAP improves. | [03 A2] one box per image; §10 A4 | Add ~300–500 **empty-label negatives** to fine-tuning (Ultralytics accepts background images) and a few multi-car images. Choose the *deployed* detector by behaviour on the web set (multi-car recall, no-car false-positive rate). Report the fine-tuned model for IoU/mAP. If it regresses, deploy the COCO model with a car/truck/bus filter. |
| M6 | Major | §2.1 M3b | **A no-car image still gets a car label.** When no box reaches ≥ 0.25, the saliency + GrabCut fallback produces an ROI, which then flows to the classifier. A cat or a selfie becomes "Bugatti Veyron 2009, 41%". | §2.1 diagram | If the detector finds nothing (optionally also check COCO classes at conf 0.1), show **"No vehicle detected"** with no make/model. Show the classical ROI only as a visual panel, behind a "classify anyway" toggle. Gate every label on the MLS/OOD test, computed on T-scaled z_deep rather than fused logits, whose scales mix. |
| M7 | Major | §4.8, R2 | **The domain of judges' unseen images is mis-specified.** The deliverable asks for **USNs**, which suggests an Indian university, so judges may upload Indian-market cars (Maruti Suzuki, Tata, Mahindra, Kia, MG). None of these are among the 49 makes [02 §6(iii)], so the make back-off cannot rescue them. In-set makes (Hyundai, Honda, Toyota, Suzuki) will appear in models that are not in the set. | [02 §6(iii)]; hackathon deliverables text | The web set must include ≥ 30 such images. Tune τ_ood/τ_make so they land in "unknown model / unknown make", and report the result honestly. Add a one-line UI scope note: "Knows 196 US-market models, 1991–2012". Rehearse this exact failure as a viva talking point. |
| M8 | Major | §7 latency | **The 10 s budget leaves out the slowest real costs:** phone upload of a 3–6 MB photo (2–4 s on 4G, est.), downloading 6+ PNG panels, Gradio queueing when two judges click at once on 2 vCPU, and cold start. §0 says "≤ 4 s p95" while §7 says "3–5 s"; they are inconsistent. | Rubric "first click from phone … ~10 s" | Return box + label + confidence **first** (a Gradio generator `yield`), then panels. Send panels as ≤ 512 px JPEGs inside a collapsed accordion. Set `concurrency_limit=1` with a visible queue. Measure end-to-end from a real phone on mobile data at H8 and H20. Use one target: p95 ≤ 6 s server-side. |
| M9 | Major | §5, §8 | **Scope exceeds team-hours.** 22 experiments, 12 modules, teacher LPs, KD, SSB, CLIP viewpoint bins, an HF-split hash-join, a label-noise audit and two classical tiers, all in an *assumed* 30 h with 4 people (duration and size are unknown, §10). The GPU sum (10–13 h) is optimistic once the M1–M2 numbers are corrected (≈ 12–18 h). The bigger constraint is people and wall-clock: E09 cannot start until E07 ends. | §5 table; my estimates (est.) | Cut list in (d). Run E09 (one long TResNet run) from ~H4 on one account while short arms run on others. **Cache per-stage outputs** (boxes, masks, classical scores, logits) to disk, so fusion, calibration, back-off and ablations E14–E17 rerun in seconds on CPU. |
| M10 | Major | §4.7, E19 | **The robustness suite is unrealistically sized and not end-to-end.** 15 corruptions × 5 severities × 8,041 = ~600k images. Several `imagecorruptions` types (glass/zoom blur) take seconds per image on CPU. The 0.5 GPU-h budget is not credible. It is also run on GT crops, so the detector's degradation (dark, blur) is never measured. M2's triggers are tuned on the *same* corruption families they are later tested on. | §4.7, E15, E19 | Use a stratified **1,960-image test subset (10/class)**, 6–8 relevant corruptions (noise, motion/defocus blur, brightness↓, contrast, JPEG, pixelate, fog) × severities {1, 3, 5}. Run **end-to-end** (corrupt the full image, then detect, crop and classify), plus a GT-crop classifier-only column. Tune M2 on val corruptions and report on a **held-out family** (e.g. speckle noise, Gaussian blur). Drop the HF corrupted splits and the hash-join. |
| m1 | Minor | §3.3, M4 | **Canny on a binary GrabCut mask** just traces the mask boundary, which `findContours` already does. A judge will read this as a checkbox. | §2.1 M4 | Run Canny on the grey ROI with median-based thresholds and **show the edge map** as a panel. Take contours from the mask. Feed edge density into M8 (already planned). |
| m2 | Minor | §2.3, M5 | **Geometric features only feed an SVM whose fusion weight is expected to be ≈ 0**, so the "geometric analysis" requirement looks token. | §3.7 | Make M5 *do* something visible. Use L-R symmetry + aspect ratio → a **viewpoint bin** (front/rear vs side vs ¾). This replaces the CLIP zero-shot viewpoint bins in §4.7, feeds the viewpoint-robustness table, and appears in the UI ("view: front-¾"). Also add a sanity flag ("aspect suggests van/pickup"). |
| m3 | Minor | §1 point 3 | **PEFT is rejected on contradictory evidence.** The cited numbers show SSF at **89.2 beating** full FT at 84.5 under the VTAB protocol [01]. | [01 §5 PEFT table] | Drop the PEFT rationale or restate it ("we full-fine-tune; PEFT was not needed"). The decision stands; the citation does not support it. |
| m4 | Minor | §3.4 E12, §4.6 | **KD from a web-scale teacher** (SigLIP2/CLIP) carries web exposure into the "lowest contamination" student, which undermines the §1 justification. | §1 contamination row | Drop E12. If kept, flag the distilled student as web-exposed. |
| m5 | Minor | §4.2, E03/E04 | **The detector must be trained on the same 90% split.** Otherwise val IoU/mAP and the val "predicted-crop" accuracy are optimistic. §5 says only "train on train boxes". | §4.2 | State it explicitly in E04's config and assert it in code. |
| m6 | Minor | §3.8, §4.2 | **τ_hi "≥ 97% val precision"** is set from ~57 val errors (815 × 7%), so it is a high-variance estimate. Many McNemar tests run without multiplicity control. | §4.2 | Report bootstrap 95% CIs for τ and for coverage. Apply Holm correction across the ablation table. Optional: 5-fold out-of-fold logits for calibration if time allows. |
| m7 | Minor | E10, E13 | **Ablations are not single-factor.** E13 removes three augmentations at once. E10's epoch count is not matched to the 15-epoch control. | §5 | Match E10 to the 15-epoch control. Split E13, or rename it "robustness-aug bundle". |
| m8 | Minor | §3.8, §4.3 | **Make back-off on as-labelled makes splits probability mass** (AM General vs HUMMER, Ram vs Dodge), so a Hummer may fail τ_make. | [02 §6(iii)] | Use display-level make groups (Hummer = AM General ∪ HUMMER; Ram ∪ Dodge) for the back-off decision. Keep the labelled makes for scoring. |
| m9 | Minor | §4.1 | Mirror selection is stale: the data is already local and verified. | Known facts | Reduce E01 to assertions (counts, box bounds, class order). Copy the data to a private Kaggle Dataset once. |
| m10 | Minor | §8, R5 | The Kaggle quota and session limit are unverified (third-party sources say 30 h/week; 12 h, or ~9 h). Accounts are one per person (ToS). GPU and internet access need phone verification. | WebSearch (third-party only) | Each member verifies their own quota panel at H0. Checkpoint every epoch to an HF model repo, and make that repo public so the Docker build needs no token. |
| m11 | Minor | §5, docs | **"README reproduces the results"**: running the full hybrid pipeline on CPU over 8,041 images takes hours. | Rubric docs band | Ship cached per-stage predictions plus `make metrics` (recomputes every table in seconds), and `make eval-subset` (500 images end-to-end, about 20 min on CPU). Document the full rerun command. |
| m12 | Minor | Deliverables | **The design doc must have "every template section"**, but no template exists in the PDF [V], and the proposal does not mention this. | Rubric docs band | Ask the organisers (see e). Default to: problem, requirements traceability, architecture, data, methods, experiments, results, limitations, ethics, deployment, team roles. |
| m13 | Minor | §0, metrics | **Efficiency may be part of the ranked "average of metrics"**: 56 M params, ~59 GFLOPs, plus a ~25 MB FV-SVM. | Rubric metrics row | Report the **deployed INT8** size and latency as headline efficiency. Show size and latency per stage. |
| m14 | Minor | §3.2, §10 | **Licensing and ethics are not fully stated:** AGPL obligations, dataset terms, and handling of uploads. | [03]; dataset terms | Add a "Source" link in the UI footer (AGPL network clause). **Never put Stanford Cars images in the repo or Docker image, or use them as app examples**; use team-shot or CC-licensed examples. Set Gradio `delete_cache` and disable flagging so uploads are not retained. State the US-market 1991–2012 bias and number-plate privacy in the Ethics section. |

---

## (c) Good — keep these

1. **Detect → padded crop → fine-tuned strong CNN**, with box jitter in training (§3.4). This is cheap, evidence-backed, and the right shape for the rubric.
2. **The validation protocol** (§4.2):
   - stratified 10% val;
   - SE stated (~±0.9 pt);
   - paired McNemar tests;
   - a single logged `eval_test.py`;
   - thresholds set only on val;
   - no retraining on train+val.
3. **Calibration and hierarchical back-off** (§3.8): temperature scaling, then class → year-merged family → make → abstain, with risk–coverage/AURC. Year-merging is the best idea in the document, because it turns Krause's "year is a representative appearance" into a UI state.
4. **Honest expectation management**: "Weak" tags, an expected fusion gain of ≤ 1 pt, and reporting β even when it is ≈ 0.
5. **Twin groups** (`year_twins`, `rebadge_twins`) with a twin-confusion rate. This is novel, cheap, and makes good viva material.
6. **The CLIP-L/14@336 linear-probe sanity row** (E08, ~0.3 GPU-h) as the "foundation model" comparison, flagged as web-exposed.
7. **ONNX-Runtime CPU Docker with no PyTorch**, static QDQ S8S8 INT8 for the CNN, and FP32 shipped if the INT8 drop is > 1 pt.
8. **A module ↔ package 1:1 mapping** (§2.3). This makes the architecture diagram match the code, which the rubric checks.
9. **Global descriptors and SIFT-FV as the classical row** (§3.6), with FV giving a credible ~50–65% (est.). Keep FV offline at least; it makes the classical-vs-deep comparison meaningful.
10. **The failure-case shortlist** (§6) and the Anderson taxonomy. TTA flip counts.
11. **Make-name parsing by longest-prefix match** against the 49-make list.

---

## (d) The 5 most important changes, in priority order

1. **Unblock the URL and ship an MVP by ~H6** (C1, C2, M8).
   - Buy HF PRO ($9) or confirm an equivalent host at H0.
   - Deploy a Docker CPU-Basic Space containing: COCO YOLO26s, a quick classifier, four classical panels, and error/loading states.
   - Return the label first and stream the panels.
   - Measure end-to-end latency from a phone at H8.
   - Keep `main` always deployable.
2. **Commit to TResNet-L now and spend the bake-off budget on its schedule** (M1, M2, M3).
   - Drop EVA-02.
   - Start from `tresnet_l.miil_in1k_448` and run 40–60 epochs at 368 (~2–3 GPU-h, est.), starting by ~H4.
   - Time ONNX FP32/INT8 of the architecture on the host at H6; decide flip TTA and 320 vs 368 from measurements.
   - Restate the target as 94–95% (est.).
3. **Make the hybrid claim defensible** (M4, m1, m2).
   - Use the three cleanly defined rows: Classical, Deep (YOLO + student), Hybrid (+ M2/M4/M8/back-off).
   - Report the Δs on clean, corrupted, dark and no-detection subsets.
   - Apply Canny to the image, not the mask.
   - Give M5 geometry a visible job (viewpoint bin and body-type sanity check).
   - In the viva: "classical CV adds robustness, fallback, interpretability and the back-off UX; on clean images it adds ≈ 0 pt top-1, and here is the McNemar test".
4. **Harden the judge-input paths** (M5, M6, M7).
   - Add negatives and multi-car images to detector fine-tuning; choose the deployed detector on web-set behaviour.
   - "No vehicle detected" must never be followed by a make/model.
   - The web set must include Indian-market modern cars and no-car photos.
   - Tune τ_ood/τ_make on val, then check them on that set.
   - Rehearse the failure explanation.
5. **Cut scope to fit the people** (M9, M10, m4). The cuts:
   - **Drop:** EVA bake-off; M7 part-crop *re-classification* (keep a CAM *visual*: for TResNet, compute the CAM from the last feature map × FC weights as an extra ONNX output, with no backward pass and no attention export); E12 KD; SigLIP2 teacher; SSB-Cars; HF-split hash-join; CLIP viewpoint bins; the make-head ablation (E11) unless time remains.
   - **Shrink:** robustness to a 1,960-image end-to-end subset.
   - **Cache:** every stage's outputs.
   - **Optional:** label-noise audit (top-50 only, using student + CLIP-LP disagreement); E02 near-duplicates.

---

## (e) Questions for the user that block decisions

1. **Host and money.** Will a member pay $9 for HF PRO today, or is another host already available? This blocks C1 and the H6 MVP.
2. **Duration, team size, time already used, and Team ID/USNs.** The whole plan assumes 30 h and 4 people. Is the deadline fixed? Late submission costs −5 points per 15 minutes.
3. **Ranking formula.**
   - Which numbers in `eval_P5_<TeamID>.md` enter the "average of performance metrics"? Do model size and latency count?
   - Must the ranked row be the *deployed* system?
   - Which weights are real: 25/20 or 15/15? (The PDF lists only five criteria; its prose implies six.)
4. **External pretraining.** Are web-scale models (CLIP/SigLIP) allowed even for a comparison row? Is ImageNet-1k pretraining (TResNet) explicitly fine?
5. **Design-doc template.** The rubric requires "every template section", but the PDF contains none. Can the organisers supply it?
6. **Judging images.** Do the judges expect Indian-market or recent cars? This decides how much effort goes into OOD/abstention versus top-1.

---

### Sources consulted for verification
- TResNet paper, arXiv 2003.13630 (PDF text, §4.1 and Tables 3 and 7).
- [timm/tresnet_l.miil_in1k_448](https://huggingface.co/timm/tresnet_l.miil_in1k_448); [timm tresnet.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/tresnet.py); [timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k).
- [Hugging Face pricing](https://huggingface.co/pricing).
- Kaggle quota (third-party only): [Notre Dame course page](https://www3.nd.edu/~dchiang/teaching/nlp/2023/kaggle.html), [aimultiple](https://aimultiple.com/free-cloud-gpu).
- `Computer Vision Hackathon 2026.pdf`, re-extracted locally.
