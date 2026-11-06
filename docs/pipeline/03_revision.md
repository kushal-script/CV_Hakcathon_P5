# 03 — Revision: the approach after critique (stage 4: research → synthesise → critique → **revise** → final)

*Written 2026-11-06. Inputs: `01_synthesis.md` (the draft), `02_critique.md` (the critique), `docs/research/01–05`, the hackathon PDF text and `experiments/README.md`. This document stands on its own: it replaces the draft and does not need to be read as a diff. Evidence tags `[01]`–`[05]` and `[P]/[S]` keep their meaning from the draft. **(est.)** marks a planning estimate. **[V]** marks a fact the critic re-checked. **[V2]** marks a fact re-checked for this revision.*

**Facts re-checked for this revision [V2]**
- The timm `tresnet.py` pretrained tags include `tresnet_l.miil_in1k_448`, `tresnet_m.miil_in1k_448`, `tresnet_m.miil_in21k_ft_in1k` and `tresnet_v2_l.miil_in21k_ft_in1k`.
- `tresnet_m.miil_in1k_448` has 31.4 M params and **23.0 GMACs @448**, Apache-2.0.
- `tresnet.py` builds on `timm.layers` SpaceToDepth, BlurPool2d and SEModule with plain LeakyReLU/ReLU. There are no custom autograd functions or JIT-only activations, so there is **no obvious ONNX blocker. This has not been tested**, which is why export is the first gate in §4.
- CPU anchor: timm `resnet50.a1_in1k` (4.1 GMACs @224) runs at **~88 ms p50 in PyTorch on 4 vCPU of an AMD EPYC 7763** (HF optimum-benchmark dataset). No HF CPU-Basic ONNX number was found.

---

## 0. Assumptions used to unblock the plan (defaults + what changes otherwise)

The user has not answered the open questions yet. We proceed on these defaults. Only Q1 and Q2 in §11 block anything.

| # | Default assumption | What changes under the alternative |
|---|---|---|
| A1 | **Host: one member buys HF PRO ($9/month [V])** at H0. We run a Docker Space on free CPU Basic (2 vCPU / 16 GB). | No PRO → the **same Docker image** goes on Google Cloud Run (free tier, needs a billing account; general knowledge, unverified). Last resort: a team laptop plus a Cloudflare quick tunnel. ZeroGPU is dropped in every case (C1). |
| A2 | **Duration 30 h, 4 members.** | **24 h:** drop E09/E10 (ablations fall back to cached test-time arms only), shorten E05 to 30 epochs, freeze at H15. **48 h:** add the deferred-stretch items (§1.3) in priority order. **3 members:** merge the UI/eval role into the others (§9). **5 members:** split UI from eval/docs. |
| A3 | **ImageNet-1k/21k pretraining is allowed. Web-scale CLIP appears only as a flagged comparison row.** | Web-scale models banned → drop the CLIP-LP row; nothing deployed changes. Only IN-1k allowed → the TResNet-M fallback switches from `miil_in21k_ft_in1k` to `miil_in1k_448`. |
| A4 | **Rubric weights = the "Max" column (Metrics 25, Implementation 20, UI 15, Docs 15, Demo 25 = 100).** The prose's "five others × 15" does not sum. The **ranked row is the deployed pipeline on test** (INT8 if shipped). | If efficiency counts in the average, nothing changes, because INT8 size and latency are already headline columns. If the ranked metric is top-1 only, we could spend the H18 slack on an offline multi-scale TTA row, but the ranked row stays the deployed one. |
| A5 | **Team ID/USNs unknown.** The file is `eval_P5_TXX.md` with a placeholder first line. | Blocks only the final submission file, not any work. |
| A6 | Judges upload single prominent cars **and** Indian-market modern cars, multi-car street shots and non-car images. | If judges use only Stanford-like images, the OOD work costs little and stays as limitations material. |

---

## 1. Final component list

### 1.1 Kept (deployed path)

| Module | What it is now | Mandatory requirement satisfied |
|---|---|---|
| M0 Input guard | Decode jpg/png/webp/heic (pillow-heif) and apply EXIF transpose. RGBA is composited on white, grey becomes 3-channel, GIFs use their first frame. Downscale to max side 1600. Reject < 64 px, > 50 MP or undecodable files with a readable message. | Implementation: bad input handled |
| M1 Quality analysis | Luminance mean/percentiles, RMS contrast, gray-world cast, noise σ, blur (variance of Laplacian) → flags | A: preprocessing/normalisation; Adv-2 (illumination) |
| M2 Conditional preprocessing | Gray-world WB if cast; CLAHE (LAB-L) + gamma if dark or low contrast; NLM if noisy. **Off on clean images.** Before/after panel shown. | A: preprocessing; C: classical→DL integration |
| M3 Detector | **YOLO26s fine-tuned as 1 class, with pseudo-labelled background cars and empty negatives** (M5 fix). ONNX @640. COCO zero-shot (car/truck/bus) is the fallback and the reported baseline. YOLO11s if YOLO26 tooling breaks. | B: YOLO detection; C: boxes with labels and confidence; metrics: IoU, mAP |
| M3b Classical ROI | Spectral-residual saliency → GrabCut on a central rect. **Never produces a label on its own** (M6 fix). Used for: the fully-classical row, a visual panel, and the "classify anyway" toggle. | A: ROI extraction |
| M4 ROI refinement | Padded box → GrabCut (5 iterations, ≤ 320 px) → mask → contours from the mask. **Canny runs on the grey ROI** (median-based thresholds) and the edge map is shown (m1 fix). The tight box replaces the detector box only if val IoU improves (E07). | A: edge detection + contours; ROI extraction |
| M5 Geometry | Aspect ratio, extent, solidity, Hu moments, L-R symmetry → **viewpoint bin (front/rear · side · ¾)** shown in the UI, plus a body-shape sanity note. Features also go into M8. (m2 fix) | A: geometric feature analysis; Adv-2 (viewing angles) |
| M6 Deep classifier | **TResNet-L@368** from `tresnet_l.miil_in1k_448`, fine-tuned on jittered padded crops. Latency fallback: **TResNet-M@368** (also the MVP model). Flip TTA live only if the latency gate allows; always offline. | B: CNN FGVC, transfer learning + fine-tuning, 196-way |
| M6c CAM visual | The final feature map is a second ONNX output. CAM = feature map · FC weight of the top-1 class, computed in NumPy: no backward pass, ~free. **Visual only, no re-classification.** | Output 5 (intermediate results); viva explainability |
| M8 Classical classifier | **Live:** global descriptors (fg-mask HSV hist + colour moments, uniform LBP 2×2, HOG, Canny edge-density 4×4, M5 geometry) → PCA → LinearSVC. **Offline:** dense RootSIFT-FV → linear SVM (the strongest classical row). | A: colour + texture features; Adv-4 |
| M9 Late fusion | `z = z_deep + β·log p_cls`, with β fit on val NLL. **Deployed only if** Δ ≥ 0 with Holm-corrected McNemar p < 0.05 on val *or* on a corrupted subset, and the added latency is ≤ 0.6 s. Otherwise it is reported and β is shown. | C: integration; Adv-4/5 |
| M10 Decision policy | Temperature scaling on z_deep. **OOD gate (MLS on T-scaled z_deep, not on fused logits)** → class → year-merged family (189) → **display-make group** (Hummer = AM General ∪ HUMMER; Ram ∪ Dodge, m8 fix) → abstain. | C: low-confidence handling; metric: confidence/threshold; Adv-1 |
| M11 Gradio UI | **Streams** box + label + confidence first, then panels as ≤ 512 px JPEGs in a collapsed accordion. `concurrency_limit=1` with a visible queue. Comparison tab. Scope note. Source link (AGPL). No flagging, `delete_cache` set. | Expected outputs 1–6; UI rubric |

Package layout stays 1:1 with the modules: `p5/{io,quality,preprocess,detect,roi,geometry,classify,cam,classical,fusion,policy}.py` plus `app/app.py`. The architecture diagram is generated from this table at H22.

### 1.2 Cut

| Cut | Why |
|---|---|
| EVA-02-B@448 and the E07 bake-off | 107.1 GMACs [V] cannot meet the latency budget, so the bake-off cannot change the decision (M1). |
| M7 part-crop re-classification | +0–0.5 pt (est.) on cars that fill the frame [03]. Its CPU cost is a second forward pass. The CAM visual stays. |
| Knowledge distillation (old E12) and the SigLIP2 teacher | Uncertain gain, and it puts web exposure into the "clean" student (m4). |
| SSB-Cars, HF corrupted splits + hash-join, CLIP viewpoint bins | Cost without rubric payoff. Viewpoint bins now come from M5. |
| ZeroGPU second URL | It needs a torch-CUDA build; ours is ONNX-CPU (C1). |
| The 600k-image robustness suite | Replaced by a 1,960-image end-to-end subset (M10). |
| Multi-scale TTA in the live path | Latency. |

### 1.3 Deferred stretch (in priority order, only after H18 freeze criteria are met, or with a 48 h duration)

1. CLIP-L/14@336 linear-probe row (~0.3 GPU-h). It is a "why not a foundation model?" sanity row, flagged as web-exposed. It is cheap enough to run as P2 when a GPU is idle, so it is effectively kept.
2. Label-noise audit, top-50 (student + CLIP-LP disagreement).
3. E02 train↔test near-duplicate check.
4. Batch-hard triplet arm (the TResNet paper's recipe ingredient).
5. 49-way make head.
6. Client-side JS downscale before upload, if the H8 phone measurement shows upload > 3 s.
7. 5-fold out-of-fold logits for calibration.

---

## 2. Critique disposition (every issue)

| ID | Verdict | Justification |
|---|---|---|
| **C1** host | **Accept-modified** | We buy PRO ($9 [V]) and deploy a Docker CPU-Basic Space at **H1** (hello-world first), using the direct `*.hf.space` URL. ZeroGPU is deleted. The critic named no non-HF fallback, so we add one: the same image on Cloud Run, else laptop + tunnel. |
| **C2** no MVP | **Accept** | Demo + UI are 40 points and capped by the URL. The MVP at H6 is in §4. Change: the MVP classifier is a TResNet-M short run (E04) that **doubles as the latency fallback**, so it is not throwaway work. |
| **M1** EVA undeployable | **Accept** | 107 GMACs ≈ 214 GFLOPs [V] is ~3.6× TResNet-L@368. On my scaling (§7), one FP32 forward on 2 vCPU is ≈ 2.7–3.7 s. **Partial disagreement:** "commit to one backbone, no hedge" is too brittle given §7. The hedge must be *cheaper* (TResNet-M), not stronger. |
| **M2** recipe gap | **Accept-modified** | The target is restated as **94–95% (est.)**. We start from `tresnet_l.miil_in1k_448` (chosen for its resolution; all timm TResNet weights are BN-remapped, so that is no differentiator). We run **40 epochs**, not 60: at ~3 min/epoch (est.) that is ~2 h on one T4, and quota is better spent on matched ablations. Triplet is deferred-stretch. |
| **M3** MACs vs FLOPs | **Accept, with a stricter number** | TResNet-L@368 = 43.6 × (368/448)² = **29.4 GMACs = 58.8 GFLOPs**. Scaling the ResNet-50 anchor [V2] gives **~0.7–1.3 s per FP32 forward** on 2 vCPU, so the critic's "1.2–2.4 s for the flip pair" is the optimistic end. The latency gate moves **earlier, to H3** (E02), because it decides what the main run trains. |
| **M4** confounded comparison | **Accept-modified** | We adopt the three clean rows. We also add input-matched diagnostic columns (classical on GT crop / YOLO crop / classical ROI) so that a weak classical row is not blamed on classical localisation. See §3. |
| **M5** detector unlearns multi-car/no-car | **Accept-modified** | We add ~500 COCO-val2017 images with no vehicles as empty labels. We also **pseudo-label background cars** (COCO YOLO26 car/truck/bus with conf > 0.5 and IoU < 0.3 to GT) instead of leaving them as implicit negatives. The deployed detector is chosen on **web-dev** (not web-test) multi-car recall and no-car FPR. |
| **M6** no-car still labelled | **Accept** | No box ≥ τ_det means **"No vehicle detected", with no make/model**. "Classify anyway" is an explicit toggle and its output is marked unreliable. The OOD gate uses T-scaled z_deep. |
| **M7** judge domain | **Accept-modified** | We add ≥ 30 Indian-market images and a UI scope note. **Correction:** Maruti *Suzuki* is an in-set make, so "Suzuki, model unknown" is a *correct* back-off. Kia, Tata, Mahindra and MG are out of set. τ_ood cannot be tuned on val (val has no OOD), so the web set is split **40% web-dev (tuning) / 60% web-test (reported once)** to keep "unseen" honest. |
| **M8** latency omissions | **Accept** | One target: **server p95 ≤ 6 s warm, phone end-to-end ≤ 10 s**. Label first via a Gradio generator; JPEG panels; queue; phone measurements at H8 and H20. Budget in §7. |
| **M9** scope > people | **Accept** | 15 experiments (was 22), 11 live modules. The main run starts at ~H3 on its own account. Every stage output is cached (§5.2). |
| **M10** robustness size | **Accept** | 1,960-image stratified test subset (10/class) × 8 corruptions × severities {1, 3, 5}, **end-to-end** plus a GT-crop column. M2 triggers are tuned on val corruptions and reported on **held-out families** (speckle noise, Gaussian blur). |
| **m1** Canny on mask | **Accept** | Canny runs on the grey ROI, the edge map is shown, and contours come from the mask. |
| **m2** token geometry | **Accept-modified** | M5 → viewpoint bin + aspect sanity note. The bin is validated on 100 hand-labelled val images. If accuracy is < 60% it is shown as "experimental" and not used in the robustness table. |
| **m3** PEFT citation | **Accept** | The cited numbers (SSF 89.2 > full FT 84.5) contradict the claim. The rationale is restated: "full fine-tuning of a CNN is cheap and proven on Cars; VTAB-protocol PEFT numbers @224 are not comparable." |
| **m4** KD contamination | **Accept** | KD is dropped. |
| **m5** detector split | **Accept** | The detector trains on the same 90% split, asserted in code (the list of val ids is checked to be disjoint). |
| **m6** threshold variance | **Accept-modified** | Bootstrap 95% CIs for τ and coverage; Holm correction across the ablation table. 5-fold OOF is deferred-stretch (cost). |
| **m7** non-single-factor ablations | **Accept** | Matched 15-epoch arms. E10 is renamed "robustness-aug bundle". |
| **m8** make splits | **Accept** | Display-make groups are used for back-off; labelled makes are used for scoring. |
| **m9** stale mirror step | **Accept** | E01 reduces to assertions + split + crop cache, uploaded once as a private Kaggle Dataset. |
| **m10** Kaggle limits | **Accept** | Each member checks their own quota panel at H0; one account per person. Checkpoints go to a **public** HF model repo every epoch, so the Docker build needs no token. |
| **m11** README reproduction | **Accept** | Cached predictions + `make metrics` (seconds) + `make eval-subset` (500 images end-to-end, ~20 min CPU, est.). The full rerun command is documented. |
| **m12** no doc template | **Accept** | We ask the organisers. Default sections: problem, requirements traceability (§1.1 table), architecture, data, methods, experiments, results, limitations, ethics, deployment, team roles. |
| **m13** efficiency metric | **Accept** | The deployed INT8 size and latency per stage are headline columns. |
| **m14** licence/ethics | **Accept** | AGPL Source link in the footer. **No Stanford Cars images** in the repo, the image or the examples; examples are team-shot or CC-licensed. No upload retention. Ethics: US-market 1991–2012 bias, number plates visible in uploads (not stored, not read). |

**Kept from the critique's "good" list:** all 11 items. Item 6 (the CLIP-LP row) is kept at P2 priority (§1.3) rather than on the critical path. Item 9 (FV) is kept offline as the main classical row, and the live classical path uses the faster global SVM.

---

## 3. Classical vs deep vs hybrid: a comparison that isolates contributions

**Principle.** Each row adds one family of components to the previous row. Where the question is "which features are better?", the rows are compared **on the same input crop**.

### 3.1 Row definitions

| Row | Localisation | Recognition | Policy |
|---|---|---|---|
| **C** Classical (pure) | M3b saliency + GrabCut ROI | M8: FV-SVM (offline) / global SVM (live) | argmax |
| **C+Y** Classical on detected ROI (hybrid-ROI) | M3 YOLO box | same M8 | argmax |
| **D** Deep | M3 YOLO box (padded) | M6 z_deep, no M2, no fusion | argmax |
| **H** Hybrid (deployed) | M3 → (M4 if E07 says so) | M2 conditional → M6 (+ M9 if accepted) | M10 TS + OOD + back-off |
| *D-full* (ablation only) | none, full image | M6 trained on full images | argmax |
| *CLIP-LP* (flagged, P2) | M3 YOLO box | frozen CLIP-L/14@336 + LR | argmax |

Interpretation, stated in the viva:
- **C → C+Y** is the value of deep localisation for classical features.
- **C+Y → D** is deep features vs classical features on identical crops.
- **D → H** is what classical preprocessing, fusion and the policy add.
- **D-full → D** is the value of detection-crop training (E09).

### 3.2 Comparison table skeleton (test, once; also on val during development)

| Row | Top-1 | Top-5 | Make Top-1 | Family Top-1 | Sel. acc @90% cov | AURC | Top-1 corrupted (subset, sev 3) | Top-1 dark | No-det subset handled | Size (MB) | p50/p95 ms (host) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C (FV) | | | | | | | | | | | offline |
| C (global, live) | | | | | | | | | | | |
| C+Y (FV / global) | | | | | | | | | | | |
| D FP32 | | | | | | | | | | | |
| H FP32 | | | | | | | | | | | |
| **H INT8 (ranked)** | | | | | | | | | | | |
| CLIP-LP (web-exposed) | | | | | | | | | | | offline |

Diagnostic columns for the classical rows: Top-1 on **GT crop / YOLO crop / classical ROI**. These separate descriptor weakness from localisation weakness.

**Honest expectation (est.):**
- C (FV) 50–65%; C (global) 10–30%; D 94–95%.
- H − D ≈ 0 on clean Top-1.
- Expected positive Δ on dark/noisy subsets (M2), on selective accuracy/AURC (M10), and on the no-detection subset, where H says "no vehicle" and D would mislabel.
- This is the defensible hybrid claim, with McNemar p-values per subset.

### 3.3 Ablation table skeleton

| # | Component removed / changed (single factor) | Arm backbone | Δ Top-1 val | Δ on target subset (which) | McNemar p (Holm) | Δ latency |
|---|---|---|---|---|---|---|
| A1 | Crop training → full-image training (E09) | TResNet-M 15 ep | | full-image test images | | — |
| A2 | Predicted crop → GT crop at test (cached) | L final | | — | | — |
| A3 | Box jitter off | M 15 ep (stretch) | | pred-crop | | — |
| A4 | Robustness-aug bundle off (E10) | M 15 ep | | Cars-C sev 3/5 | | — |
| A5 | M2 conditional preprocessing off (E12) | L final, cached | | dark / noise, held-out family | | |
| A6 | M4 tight box → detector box (E07) | L final | IoU Δ | — | | |
| A7 | M9 fusion off, β = 0 (E12) | L final, cached | | clean + corrupted | | |
| A8 | Flip TTA off (+ flip counts) | L final, cached | | — | | |
| A9 | Temperature scaling off (ECE, AURC) | cached | ECE Δ | — | — | — |
| A10 | Year-merge / make back-off off (selective acc) | cached | — | web-test | — | — |
| A11 | INT8 → FP32 (E11) | L final | | — | | |
| A12 | Fine-tuned → zero-shot detector | — | IoU/mAP Δ | multi-car/no-car web-dev | | |

Arms A1, A3 and A4 use **TResNet-M at a matched 15 epochs** as a declared proxy backbone. This saves ~3 GPU-h versus L arms, and the limitation is stated. Everything else is test-time and reruns from caches in seconds.

---

## 4. Build order: minimum viable demo first

**Rule:** `main` is always deployable. After every milestone the Space is rebuilt from `main` and the phone smoke test is re-run (one Stanford-like car, one non-car, one 12 MP phone photo).

| Hour | Milestone | Owner |
|---|---|---|
| **H0–1** | PRO bought; Space created with a hello-world Docker Gradio → **URL live at H1**. Repo skeleton, Dockerfile (ORT, opencv-contrib-headless, sklearn, gradio, pillow-heif; no torch). Quota panels checked. | B |
| H0–2 | **E01**: assertions, 90/10 stratified split (seed 42), class parser + display-make groups + `groups.json`, **crop cache** (GT box + 16% pad, max side 448, JPEG q95) and full-image cache (max side 512) → private Kaggle Dataset. | A |
| H1–3 | **E02 latency gate**: export *untrained* TResNet-L@368, L@320 and M@368 plus YOLO26s to ONNX FP32 + INT8 (random-data calibration is fine for timing). ORT parity check vs PyTorch on 10 images. Time on the **real Space**. | B |
| **H3 decision** | Single-forward INT8 L@368 ≤ 0.8 s → main = L@368. ≤ 0.8 s only at 320 → L trained at 368, served at 320 (TResNet is multi-res-robust [V]; verify on val). Otherwise main = **M@368 for 40 epochs**. Flip TTA live only if pair ≤ 1.0 s. | B + C |
| H2–3 | **E03** zero-shot COCO YOLO26s on val (mIoU, mAP). **E04** TResNet-M@368, 6 epochs (~10–15 min T4, est.) → ONNX. | B, C |
| H3 | **E05 main run starts** (account C1, checkpoints every epoch). | C |
| H3–6 | App v1 wiring: M0, M1, M2 panels, COCO-YOLO (car/truck/bus), M6 = E04 model, CAM, 4 classical panels (CLAHE before/after, Canny edge map, mask contour, HSV histogram), no-vehicle and error states, streamed output, queue. | D (UI), A (panels) |
| **H6** | **MVP live: end-to-end, weak (~85%?, est.) but real.** First phone test. | D |
| H6–12 | E06 detector FT (negatives + pseudo-labels); E07 GrabCut IoU; E08 classical SVMs; M5 viewpoint; T-scaling + back-off code on E04 logits (rerun later on E05 logits); web set collection and dev/test split. **H8: phone end-to-end measurement #1.** | A, B, D |
| ~H6–7 | E05 finishes → export, INT8, swap in → **app v2**. E09/E10 ablation arms on accounts C2/C3. | C, B |
| H12–18 | E11 INT8 eval; E12 fusion + M2 on/off from caches; E13 calibration/OOD on val + web-dev; detector choice on web-dev → **app v3**. E14 robustness. | all |
| **H18** | **Freeze**: no training, thresholds fixed. | — |
| H18–22 | E15 final test run (once, logged); web-test; host latency; **H20 phone measurement #2**; failure analysis. | D, B |
| H22–27 | `eval_P5_TXX.md`, design doc, generated diagram, README reproduction from a clean clone. | all |
| H27–30 | 3-min demo script, dry runs with web-test images, viva Q&A per member, pre-warm schedule. | all |

Fallbacks:
- If E04 has not exported by H5, the MVP ships with an ImageNet-pretrained TResNet-M plus a logistic-regression head on pooled features (GPU feature extraction ~5 min, est.).
- If the Space build fails, the H1 hello-world proves the path, and we switch to the A1 fallback host before H6, not at H20.

---

## 5. Revised experiment list

Each experiment is created with `scripts/new_experiment.py --type <type> --title "E0x: …"` and gets its own `config.yaml` and `metrics.json`. Results are tuned on val; test is touched only in E14/E15 via the logged `eval_test.py`. T4 times are estimates.

| ID (was) | Type | Hypothesis | Config delta | Metric (split) | T4 time |
|---|---|---|---|---|---|
| E01 (E01) | data | Local data matches 8,144/8,041/196, labels 0–195 in meta order, all boxes in bounds; parser yields 49 makes | — | assertions, counts | 0 (CPU 20 min) |
| E02 (new) | optimisation | INT8 L@368 single forward ≤ 0.8 s on host; INT8 ≥ 1.5× faster than FP32 (est.) | untrained L@368/320, M@368, YOLO26s; FP32/INT8 | p50/p95 ms on Space, ORT parity | 0 |
| E03 (E03) | detection | Zero-shot COCO YOLO26s (car+truck+bus) mIoU ≥ 0.85 (est.) | no training | mIoU, IoU≥.5/.75, mAP50/50-95 (val) | 0.1 |
| E04 (new) | deep | A 6-epoch TResNet-M@368 reaches ≥ 80% val (est.), enough for the MVP | M, 6 ep, jittered crops | Top-1/5 (val) | 0.25 |
| E05 (E09) | deep | **Main**: TResNet-L@368, 40 ep, reaches 94–95% val (est.) | AdamW, cosine + 2-ep warm-up, LS 0.1, light RandAugment, hflip, RandomErasing 0.25, down-up-scale p 0.2, random CLAHE/gamma p 0.2, AMP, bs 32 | Top-1/5, make, family (val) | 2.0–2.5 |
| E06 (E04) | detection | FT (1 class, 20 ep, 90% split, +500 negatives, +pseudo-labelled background cars) improves mIoU/mAP over E03 **without** worse multi-car recall / no-car FPR on web-dev | train set delta | val IoU/mAP; web-dev recall, FPR | 0.7 |
| E07 (E05) | hybrid | GrabCut tight box beats the E06 box on IoU | M4 on/off | mIoU (val) | 0 (CPU) |
| E08 (E06) | classical | Global SVM 10–30%, FV 50–65% (est.); GT > YOLO > classical-ROI input | input ∈ {GT, YOLO, M3b} | Top-1/5 (val) | 0 (CPU 2–3 h) |
| E09 (E10) | ablation | Crop training ≥ +1 pt over full-image training (est.) | M, 15 ep, full images vs crops (control) | Top-1 (val) on matched inputs | 0.8 (2 arms) |
| E10 (E13) | ablation | Robustness-aug bundle costs < 0.5 pt clean and helps Cars-C sev 3/5 | M, 15 ep, bundle off (shares E09 control) | Top-1 clean + corrupted (val) | 0.4 |
| E11 (E18) | optimisation | Static QDQ S8S8 per-channel INT8 (reduce_range if no VNNI) loses ≤ 1 pt, ~4× smaller; YOLO `quantize=8` | FP32/FP16/INT8 | Top-1, MB, p50/p95 on host (val) | 0.2 |
| E12 (E15+E16) | hybrid | M2 helps dark/noise sev ≥ 3 (incl. held-out family) and is neutral on clean; fusion β > 0 helps only on corrupted subsets | M2 on/off; β grid | Top-1 per subset, NLL, McNemar-Holm (val) | 0 (cached) |
| E13 (E17+E20) | calibration | TS lowers ECE; τ_hi/τ_make/τ_ood with bootstrap CIs; MLS ≥ energy on web-dev OOD AUROC | — | ECE/MCE, AURC, acc@cov, AUROC/FPR95 (val, web-dev) | 0 (cached) |
| E14 (E19) | robustness | End-to-end degradation is driven by the detector at dark/blur sev 5 | 1,960 test subset × 8 corruptions × {1,3,5}; resolution 368→112/56/28; occlusion 10/25/50%; M5 viewpoint bins | Top-1 e2e + GT-crop column, relative mCE (test, frozen) | 0.5 |
| E15 (E22+E21) | hybrid / failure-analysis | **Final comparison §3.2 + ablation §3.3**; twin-confusion rates; web-test back-off distribution; ≥ 3 failure cases | frozen | all §6 metrics (test, once; web-test once) | 0.3 |
| P2 (E08) | deep | CLIP-L/14@336 LP ≈ 91.5 ± 3 on crops (pipeline sanity) | frozen + LR | Top-1 (val → test) | 0.3 |

**Total ≈ 6–7 T4-hours (est.)**, about half the draft's corrected figure, spread over ≥ 3 accounts.

The `metrics.json` keys added in the draft are kept: `family_top1`, `rebadge_twin_confusion`, `aurc`, `fpr95`, `tta_flips`, `robustness.<name>_s<k>_top1`. We also add `latency_ms.{stage}.{p50,p95}` and `input` (gt/yolo/roi/full).

### 5.2 Caching of stage outputs

`cache/<split>/<stage>.<ext>`, keyed by image id, with each file's producing experiment folder recorded in its header:

| Stage | Contents | Used by |
|---|---|---|
| `det_zs.parquet`, `det_ft.parquet` | all boxes, conf (conf 0.001, max_det 100) | IoU/mAP, crop selection, E14 |
| `roi_classical.parquet`, `mask_rle.parquet`, `tightbox.parquet` | M3b ROI, GrabCut masks, M4 boxes | E07, E08, C rows |
| `quality_flags.parquet` | M1 stats/flags | E12 subsets |
| `cls_feats_{global,fv}.npz`, `cls_scores_*.npz` | descriptors, Platt-scaled SVM scores | E08, E12 |
| `logits_{model}_{prec}_{input}_{flip}.npz` | 196 logits for FP32/INT8 × GT/pred/full × flip | E11–E15, all test-time ablations |
| `corrupt/<name>_s<k>/…` | e2e logits and boxes on the 1,960 subset | E14 |

`make metrics` rebuilds every table from these caches. `make eval-subset` regenerates them for 500 images end-to-end.

---

## 6. Evaluation plan (order required by `eval_P5_<TeamID>.md`)

The first line holds the USNs, comma-separated. Every table states its split, resolution, input crop and precision.

1. **Top-1/Top-5**: rows §3.2; make, family, year-twin and rebadge-twin confusion; crop matrix (full / GT / predicted).
2. **IoU**: mean IoU, IoU ≥ 0.5 / ≥ 0.75 for zero-shot, fine-tuned, GrabCut-refined and classical ROI.
3. **mAP**: single-class mAP50 and mAP50-95 (pycocotools, max_det 100). Class-aware mAP50 with score = det_conf × p_cls.
4. **Efficiency**: params, MB on disk, GMACs (stated as MACs), p50/p95 per stage and end-to-end on the host, cold start, FP32 vs INT8 (headline = deployed INT8).
5. **Robustness**: E14 tables; M2 on/off; viewpoint bins.
6. **Confidence**: reliability diagrams, ECE/MCE before/after TS, risk–coverage + AURC, τ's with bootstrap CIs, back-off level distribution (test, web-test), OOD AUROC/FPR95, flip counts.

The ablation table (§3.3, Holm-corrected) and the failure cases follow.

Failure-case shortlist:
- Express ↔ Savana rebadge;
- Caliber 2007/2012 year twin;
- a post-2013 in-set-make car confidently mapped to its ancestor;
- an Indian-market out-of-set make;
- a multi-car / off-centre image;
- extreme low light.

---

## 7. Compute and latency (corrected units)

### 7.1 Model compute

| Model | Input | GMACs | ≈ GFLOPs (= 2×MACs) | Params | INT8 size (est.) |
|---|---|---|---|---|---|
| TResNet-L | 448 | 43.6 [V] | 87.2 | 56.0 M [V] | — |
| **TResNet-L** | **368** | **29.4** | **58.8** | 56.0 M | ~57 MB |
| TResNet-L | 320 | 22.2 | 44.5 | 56.0 M | ~57 MB |
| TResNet-M | 368 | 15.5 | 31.0 | 31.4 M [V2] | ~32 MB |
| EVA-02-B (cut) | 448 | 107.1 [V] | 214 | 87.1 M | — |
| YOLO26s | 640 | — | ~20 (Ultralytics reports FLOPs; est. from YOLO11s 21.5) | ~10 M (est.) | ~10 MB |

### 7.2 CPU latency estimate

The anchor is ResNet-50 at 4.1 GMACs taking ~88 ms in PyTorch on 4 vCPU [V2].
- Assume 2 vCPU is 1.6–2× slower, and ORT FP32 is 1.2–1.5× faster than PyTorch. That gives **~25–45 ms per GMAC** on 2 vCPU (est.).
- TResNet-L@368 FP32 is then **0.7–1.3 s per forward**, and 1.4–2.6 s with flip.
- INT8 static gives ~1.5–2.5× speed-up with VNNI and possibly ~1× without it [05 §2.3]: **0.35–0.9 s (est.)**.
- TResNet's GPU-oriented design (SpaceToDepth, wide early stages) may be less CPU-efficient per MAC than ResNet-50. That uncertainty is exactly why E02 measures on the host at H3.

### 7.3 Latency budget (default: INT8 L@368, no live flip, global classical SVM)

| Stage | Budget (est.) | Notes |
|---|---|---|
| Phone upload of 3–6 MB on 4G | 1–4 s | **client side**; iOS Safari usually transcodes HEIC → JPEG |
| Queue wait | 0 (alone); +one job (~3–4 s) if two judges click at once | `concurrency_limit=1`, visible position |
| Decode + EXIF + resize + M1 | 0.1–0.3 s (HEIC 0.3–0.5) | |
| M2 conditional | 0–0.15 s | only when flagged |
| M3 YOLO26s @640 | 0.15–0.3 s | |
| M6 INT8 single forward + CAM | 0.35–0.9 s | flip only if the E02 gate passes |
| M10 policy + render box | 0.05 s | **first `yield`: label at ≈ 0.7–1.7 s server** |
| M4 GrabCut ≤ 320 px + Canny + contours + M5 | 0.25–0.45 s | |
| M8 global features + SVM | 0.1–0.2 s | FV offline only |
| Up to 2 extra cars (single forward each) | +0.35–0.9 s each | multi-car only |
| Panels as ≤ 512 px JPEG + transfer | 0.1–0.3 s | collapsed accordion |
| **Server total p95 (single car)** | **≈ 1.5–3.5 s; target ≤ 6 s; hard cap 8 s** | at 6 s: skip extra cars, then M4 panels ("fast mode") |
| **Phone end-to-end** | **≈ 3–8 s; target ≤ 10 s** | measured at H6, H8, H20 |
| Cold start (container + model load) | 20–60 s | pre-warm at T−30 min and T−5 min before judging; CPU Basic sleeps after 48 h idle [05] |

### 7.4 Host plan

| | Default | Fallback 1 | Fallback 2 |
|---|---|---|---|
| Host | HF Docker Space, CPU Basic (PRO $9) | Google Cloud Run, same image, 2 vCPU / 4 GB, min-instances 1 during judging | Team laptop + Cloudflare quick tunnel |
| Trigger | — | Space build fails or PRO not bought by H1 | No billing account available |
| URL given to judges | direct `https://<space>.hf.space` | `*.run.app` | `*.trycloudflare.com` (changes on restart; keep the laptop awake) |

Weights are pulled from a public HF model repo at build time. No secrets.

---

## 8. Judge-input handling (exact UI behaviour)

Banner wording is fixed below. Every path returns a page, never a stack trace (a top-level try/except logs the error id).

| Input | Detection / policy outcome | Exact UI |
|---|---|---|
| Stanford-like car, high confidence (p ≥ τ_hi) | class level | Box + **"Audi S4 Sedan 2012 — 93%"**, top-5 bar chart, view "front-¾", panels |
| Year-twin ambiguity | family level (Σ over the twin pair ≥ τ_fam) | **"Dodge Caliber Wagon — 2007 or 2012 (88%)"** + note "these model years look nearly identical" |
| Modern car of an in-set make (2020 Audi A4, 2019 Honda City) | MLS < τ_ood but make mass ≥ τ_make → make level; else abstain | **"Audi — exact model not in our catalogue (closest known: Audi S4 Sedan 2012, low confidence)"**. Scope note always visible: *"Recognises 196 US-market models from 1991–2012."* |
| Maruti Suzuki (Swift, Baleno) | likely make = Suzuki if mass ≥ τ_make, else abstain | "Suzuki — exact model not in our catalogue", or the abstain text below |
| Out-of-set makes: Tata, Mahindra, Kia, MG | MLS < τ_ood and no make ≥ τ_make | **"Vehicle detected — make/model not in our catalogue."** Collapsed "Closest known models (not a prediction)" with top-3 |
| Multiple cars | all boxes ≥ τ_det (max 5) drawn; primary = argmax(conf × √area) gets the full pipeline; up to 2 others get a single forward | Primary box thick with full label; others thin with short labels; "3 vehicles found — details for the largest; select another:" dropdown re-runs on that box |
| No car (cat, selfie, landscape) | no box ≥ τ_det (0.25) | **"No vehicle detected."** No make/model. Shows the M1 panel + saliency ROI panel. Checkbox **"Classify anyway"** → result labelled "Forced classification — unreliable" |
| Weak detection (0.10 ≤ conf < 0.25) | classify with warning | Banner **"Possible vehicle (detection confidence 0.18) — result may be unreliable"** |
| Close-up / partial car (box > 90% of frame or truncated) | normal pipeline; truncation flag | Note "Car is cropped or very close — accuracy is lower for partial views" |
| Phone 12 MP JPEG / HEIC | M0 EXIF-transpose, downscale to 1600 | Normal; the spinner shows "Analysing…" at once (Gradio loading state) |
| Night / dark photo | M1 dark flag → M2 CLAHE + gamma | Banner **"Low light detected — image enhanced (see panel); confidence may be lower"** + before/after panel |
| Blurry / noisy | M1 blur/noise flag (NLM if noisy) | Banner "Image is blurry/noisy — confidence may be lower" |
| Tiny (< 64 px), > 50 MP, corrupt, non-image | M0 reject | "Image too small (min 64 px)" / "Image too large" / "Couldn't read this file — please upload JPG, PNG, WebP or HEIC" |
| Greyscale | 3-channel copy | Note "Greyscale image — colour features unavailable"; the colour panel is hidden |
| Two users at once | queue | "You are #2 in the queue (~4 s)" |

All thresholds (τ_det, τ_hi, τ_fam, τ_make, τ_ood) are set on val and web-dev and frozen at H18. Behaviour on web-test is reported once, with the back-off level distribution.

---

## 9. Team split (default 4)

| Member | Owns and explains |
|---|---|
| A — Data + Classical | E01, E07, E08; M0–M2, M3b, M4, M5, M8; classical panels |
| B — Detection + Optimisation + Deploy | H1 Space, E02, E03, E06, E11; Dockerfile, host, latency, pre-warm |
| C — Deep | E04, E05, E09, E10, P2; training recipe, ONNX export, CAM |
| D — Policy + UI + Eval + Docs | E12–E15; M9–M11; Gradio, judge-input table, `eval_P5`, diagram, README |

With 3 members, D's UI work goes to A and D's evaluation goes to B. With 5, D splits into UI/deploy and eval/docs.

---

## 10. Risk register

| # | Risk | L | Impact | Mitigation |
|---|---|---|---|---|
| R1 | No host / Space build fails | Med | Critical (UI → 0) | PRO at H0, hello-world URL at H1, Cloud Run/tunnel fallback decided before H6 |
| R2 | TResNet-L too slow on 2 vCPU (§7.2 says it is plausible) | Med-High | UI band | E02 gate at H3 *before* the main run; serve at 320 or switch the main run to M@368; flip offline only |
| R3 | ONNX export or INT8 breaks on TResNet (SpaceToDepth/BlurPool untested; SE/LeakyReLU sensitivity; non-VNNI saturation) | Low-Med | Metrics/latency | Export smoke test at H2 with ORT parity; S8S8 + reduce_range; `qdq_loss_debug` keeps sensitive layers FP32; ship FP32 if Δ > 1 pt |
| R4 | Judge's modern/Indian car gets a confident wrong label | High | Demo | OOD gate on web-dev, make/family back-off, scope note, rehearsed explanation, web-test reported honestly |
| R5 | Fine-tuned detector regresses on multi-car/no-car | Med | Demo | Negatives + pseudo-labels; choose on web-dev; COCO zero-shot fallback |
| R6 | Main run underperforms (< 92% val) or crashes mid-quota | Low-Med | Metrics | Checkpoint every epoch to HF; E04/M fallback model; resume script; one account dedicated to E05 |
| R7 | Kaggle quota/session shorter than assumed (~9–12 h, unverified) | Med | Schedule | E05 ≈ 2–2.5 h is well under any session limit; ablations on separate accounts |
| R8 | Phone upload/latency > 10 s | Med | UI band | Streaming label, JPEG panels, measurements at H6/H8/H20, client-side downscale stretch |
| R9 | Cold start at judging | Med | UI band | Pre-warm schedule; Cloud Run min-instances 1 |
| R10 | Test leakage via repeated eval | Low-Med | Credibility | Single logged `eval_test.py`; τ's from val/web-dev only; web-test touched once |
| R11 | Scope creep after MVP | Med | Docs/viva time | Freeze at H18; deferred-stretch list is explicit and ordered |
| R12 | A member cannot explain their part | Med | Demo/viva (tie-break) | Ownership table §9; each owner writes their design-doc section and rehearses 2 viva questions |
| R13 | AGPL / dataset-terms objection | Low | Docs | AGPL repo + Source link; RF-DETR-Nano (Apache-2.0) swap-in; no Stanford images distributed |

---

## 11. Open questions (only these block)

1. **Will a member buy HF PRO ($9) at H0?** This blocks the H1 URL. Default: yes. If no, we need a Cloud Run billing account or accept the laptop tunnel.
2. **What are the real deadline and team size?** These set the freeze hour and whether E09/E10 survive (see A2). Default: 30 h, 4 people.

Non-blocking, needed before submission:
- Team ID + USNs (file name and first line).
- The design-doc template, if the organisers have one.
- Whether efficiency and the deployed row enter the ranking (default A4).
- Whether web-scale weights are allowed for the comparison row (default A3).
