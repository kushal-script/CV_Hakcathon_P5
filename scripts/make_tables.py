#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def pct(v, digits: int = 2) -> str:
    return "n/a" if v is None else f"{100 * v:.{digits}f}"


def main() -> int:
    p = argparse.ArgumentParser(description="Build eval_P5_<TeamID>.md from the logged experiment outputs")
    p.add_argument("--team", default="TeamID")
    p.add_argument("--usns", default="USN1,USN2,USN3,USN4")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    test = load(RUNS / "e15_test" / "test_report.json")
    fail = load(RUNS / "e15_test" / "failures" / "failures.json")
    quant = load(RUNS / "remote" / "work-runs" / "e05_tresnet_l_368" / "export_int8" / "quantization.json")
    calib = load(RUNS / "e13_calibration" / "calibration.json")
    lat = load(RUNS / "latency.json")
    abl = load(RUNS / "e09_e10_ablation.json")
    fusion = load(RUNS / "e12_fusion_fp32" / "fusion.json")
    classical = load(RUNS / "e08_classical" / "metrics.json")
    det_val = load(RUNS / "remote" / "work-runs" / "e06_detector" / "metrics.json")
    neg = [json.loads(l) for l in (RUNS / "remote" / "work-runs" / "e15_pipeline" / "coco_neg_test_pipeline.jsonl").read_text().splitlines()]

    L = [args.usns, "", f"# eval_P5_{args.team}: Intelligent Fine-Grained Vehicle Recognition and Analysis System", "",
         f"All numbers are on the Stanford Cars **test split (n = {test['n']:,})** unless marked val. Models, thresholds and the "
         "enhancement rule were frozen on train/validation before test labels were read; test labels are read only by "
         "`scripts/eval_test.py` and every call is logged in `experiments/test_log.csv`.", "",
         "**Deployed system:** YOLO26s (fine-tuned, 1 class) → crop → TResNet-L@368 (ImageNet-1k pretrained, fine-tuned, INT8) → "
         "temperature-scaled decision policy with family/make back-off; classical quality analysis, enhancement with detector-confidence "
         "selection, GrabCut/Canny/contour ROI, geometry and a colour/texture/shape SVM for the intermediate panels and the comparison.", ""]

    L += ["## 1. Classification performance: Top-1 and Top-5", "",
          "| Approach | Input | Top-1 | Top-5 | Make Top-1 | Family Top-1 |", "|---|---|---|---|---|---|"]
    rows = [("Classical only (HOG, colour, LBP, edges, geometry → LinearSVC)", "saliency ROI", test.get("classical_roi")),
            ("Classical on detector crop", "YOLO crop", test.get("classical_pred")),
            ("Deep only, no detection", "full image", test.get("deep_full", {}).get("single")),
            ("Deep on detector crop (FP32)", "YOLO crop", test.get("deep_pred", {}).get("single")),
            ("Deep on detector crop (INT8)", "YOLO crop", test.get("deep_int8_pred", {}).get("single")),
            ("**Hybrid: deployed system (INT8)**", "**raw upload**", test.get("deployed_pipeline")),
            ("Deep on GT crop (oracle localisation, diagnostic)", "GT crop", test.get("deep_gt", {}).get("single")),
            ("I-JEPA ViT-H/14, fine-tuned (research row)", "YOLO crop", test.get("extra_ijepa_finetune_pred")),
            ("I-JEPA ViT-H/14, frozen linear probe (research row)", "GT crop", test.get("extra_ijepa_probe_gt"))]
    for name, inp, r in rows:
        if r:
            bold = "**" if "deployed" in name else ""
            L.append(f"| {name} | {inp} | {bold}{pct(r['top1'])}{bold} | {pct(r['top5'])} | {pct(r['make_top1'])} | {pct(r['family_top1'])} |")
    dp = test["deployed_pipeline"]
    L += ["", f"Visually similar classes (deployed system): year-twin confusion {pct(dp['year_twin_confusion'], 1)}%, "
          f"rebadge-twin confusion {pct(dp['rebadge_twin_confusion'], 1)}%. Make-level accuracy {pct(dp['make_top1'])}%.", ""]

    L += ["## 2. Object localisation: IoU", "", "| Detector | Mean IoU | IoU ≥ 0.5 | IoU ≥ 0.75 |", "|---|---|---|---|"]
    for tag, name in (("zs", "YOLO26s, COCO zero-shot (car, bus, truck)"), ("ft", "**YOLO26s, fine-tuned (deployed)**")):
        d = test[f"detection_{tag}"]
        L.append(f"| {name} | {d['mean_iou']:.4f} | {pct(d['iou_ge_0_5'])} | {pct(d['iou_ge_0_75'])} |")
    L += ["", "## 3. Object detection: mAP", "", "| Detector | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 |", "|---|---|---|---|"]
    for tag, name in (("zs", "YOLO26s zero-shot"), ("ft", "**YOLO26s fine-tuned (deployed)**")):
        d = test[f"detection_{tag}"]
        L.append(f"| {name} | {d['map_50']:.4f} | {d['map_75']:.4f} | {d['map_50_95']:.4f} |")
    nv = sum(1 for r in neg if r["level"] == "no_vehicle")
    L += ["", f"No-vehicle images: {nv}/{len(neg)} held-out COCO test images without vehicles correctly return \"no vehicle\". "
          f"Test images with no detection: {pct(dp['no_vehicle_rate'])}%.", ""]

    L += ["## 4. Model efficiency: size and inference time", "",
          "| Classifier | Size (MB) | Val Top-1 | Test Top-1 (detector crop) | Latency p50, 2 threads, AMD EPYC 7542 | Latency p50, 2 threads, Apple M3 |",
          "|---|---|---|---|---|---|"]
    if quant:
        L.append(f"| FP32 | {quant['fp32']['size_mb']} | {pct(quant['fp32']['top1'])} | {pct(test['deep_pred']['single']['top1'])} | "
                 f"{quant['fp32']['latency_p50_ms']:.0f} ms | {lat.get('fp32', {}).get('classify', {}).get('p50', float('nan')):.0f} ms |")
        L.append(f"| **INT8 (static, QDQ, per-channel)** | **{quant['int8']['size_mb']}** | {pct(quant['int8']['top1'])} | "
                 f"{pct(test['deep_int8_pred']['single']['top1'])} | {quant['int8']['latency_p50_ms']:.0f} ms | "
                 f"{lat.get('int8', {}).get('classify', {}).get('p50', float('nan')):.0f} ms |")
    L += ["", "Detector: YOLO26s ONNX, 38 MB. INT8 is 3.9× smaller and 3.2× faster on CPUs with integer dot-product "
          "instructions (ARM, x86 VNNI); on the AMD EPYC 7542 (no VNNI) it is slower than FP32.", ""]
    if lat:
        L += [f"End-to-end latency of the full product pipeline (all panels and comparison rows), {lat['threads']} threads, "
              f"{lat['n']} test images, {lat['cpu']}:", "", "| Stage | INT8 p50 (ms) | INT8 p95 (ms) | FP32 p50 (ms) | FP32 p95 (ms) |", "|---|---|---|---|---|"]
        for k in lat["int8"]:
            f32 = lat.get("fp32", {}).get(k, {})
            L.append(f"| {k} | {lat['int8'][k]['p50']:.0f} | {lat['int8'][k]['p95']:.0f} | {f32.get('p50', float('nan')):.0f} | {f32.get('p95', float('nan')):.0f} |")
        L.append("")

    L += ["## 5. Robustness: performance under varying conditions", "",
          "1,960 random test images, end-to-end Top-1 (%). \"Deployed\" applies the classical enhancement and keeps whichever of "
          "the original or enhanced image gives the more confident detection (rule chosen on validation). Speckle noise and Gaussian "
          "blur were never used for tuning.", "",
          "| Condition | Raw | Always enhance | **Deployed** | GT crop | Detection rate raw → deployed |", "|---|---|---|---|---|---|"]
    for k, v in test.get("robustness", {}).items():
        L.append(f"| {k} | {pct(v['e2e'], 1)} | {pct(v['e2e_m2'], 1)} | **{pct(v['e2e_select'], 1)}** | {pct(v['gt'], 1)} | "
                 f"{pct(v['detected'], 0)}% → {pct(v['detected_select'], 0)}% |")
    L.append("")

    c = test["confidence"]
    L += ["## 6. Confidence and threshold analysis", "",
          f"* Temperature T = {calib['temperature']:.3f}, fitted on validation (deployed INT8 model on detector crops).",
          f"* Expected calibration error on test: {pct(c['ece_before'])}% before, **{pct(c['ece_after'])}%** after temperature scaling.",
          f"* AURC {c['aurc']:.4f}. Accuracy at 80 / 90 / 95 / 100% coverage: " +
          " / ".join(pct(c["accuracy_at_coverage"][k]) for k in ("0.8", "0.9", "0.95", "1.0")) + ".",
          "* Thresholds (validation, 95% bootstrap CI): " + ", ".join(
              f"{k} = {v:.3f} [{calib['bootstrap_ci_95'][k][0]:.2f}, {calib['bootstrap_ci_95'][k][1]:.2f}]" for k, v in calib["thresholds"].items()) +
          "; detector τ_lo = 0.10, τ_det = 0.25.", "",
          "| Decision level (deployed, test) | Share of images | Correct at that level |", "|---|---|---|"]
    names = {"class": "Make, model and year", "family": "Model family, year uncertain", "make": "Make only",
             "ood_make": "Possibly ⟨make⟩, not one of the 196 models", "no_vehicle": "No vehicle detected"}
    for k in ("class", "family", "make", "ood_make", "no_vehicle"):
        if k in dp["levels"]:
            L.append(f"| {names[k]} | {pct(dp['levels'][k])}% | {pct(dp['level_accuracy'].get(k)) + '%' if k in dp['level_accuracy'] else '—'} |")
    L += ["", f"Across all levels, {pct(dp['answered_correct_rate'])}% of test images receive an answer that is correct at the level given.", ""]

    L += ["## 7. Comparison: classical vs deep vs hybrid", "",
          "| Approach | Test Top-1 | What it adds |", "|---|---|---|",
          f"| Classical only (saliency ROI) | {pct(test['classical_roi']['top1'])} | Fully hand-crafted pipeline, no learning beyond a linear SVM |",
          f"| Classical + deep detector | {pct(test['classical_pred']['top1'])} | Deep localisation; recognition still hand-crafted |",
          f"| Deep only (full image) | {pct(test['deep_full']['single']['top1'])} | Fine-tuned CNN without detection |",
          f"| Deep detector + deep classifier | {pct(test['deep_pred']['single']['top1'])} | Detection crop feeds the fine-grained classifier |",
          f"| **Hybrid deployed system** | **{pct(dp['top1'])}** | + classical enhancement with detector selection, calibrated back-off, classical panels |", "",
          "Robustness is where the hybrid differs most (section 5): Gaussian noise s1 81.9 → 90.9, speckle s3 78.9 → 88.3, contrast s5 83.9 → 91.2.", ""]

    L += ["## 8. Ablation study", "", "| Factor (one change at a time) | Split | Top-1 with | Top-1 without | Δ | Significance |", "|---|---|---|---|---|---|"]
    if abl:
        L.append(f"| Detection crop vs full image (TResNet-M, 15 epochs) | val | {pct(abl['e09_control_crops']['top1'])} | "
                 f"{pct(abl['e09_full_image']['top1'])} | {100 * -abl['e09_full_image']['delta_vs_control']:+.2f} | McNemar p = {abl['e09_full_image']['p_holm']:.3f} (Holm) |")
        L.append(f"| Robustness augmentation bundle | val | {pct(abl['e09_control_crops']['top1'])} | {pct(abl['e10_no_aug']['top1'])} | "
                 f"{100 * -abl['e10_no_aug']['delta_vs_control']:+.2f} | p = {abl['e10_no_aug']['p_holm']:.3f} (n.s. on clean images) |")
    L.append(f"| Detection crop vs full image (TResNet-L, deployed model) | test | {pct(test['deep_pred']['single']['top1'])} | "
             f"{pct(test['deep_full']['single']['top1'])} | {100 * (test['deep_pred']['single']['top1'] - test['deep_full']['single']['top1']):+.2f} | — |")
    L.append(f"| Fine-tuned vs zero-shot detector (mean IoU) | test | {test['detection_ft']['mean_iou']:.4f} | {test['detection_zs']['mean_iou']:.4f} | "
             f"{test['detection_ft']['mean_iou'] - test['detection_zs']['mean_iou']:+.4f} | — |")
    if fusion:
        L.append(f"| Classical score fusion (β = {fusion['best_beta']}) | val | {pct(fusion['fused']['top1'])} | {pct(fusion['deep']['top1'])} | "
                 f"{100 * (fusion['fused']['top1'] - fusion['deep']['top1']):+.2f} | McNemar p = {fusion['mcnemar_fused_vs_deep']['p_value']:.2f} (rejected) |")
    L.append(f"| INT8 quantization | val | {pct(quant['int8']['top1'])} | {pct(quant['fp32']['top1'])} | {100 * (quant['int8']['top1'] - quant['fp32']['top1']):+.2f} | 3.9× smaller |")
    rob = test.get("robustness", {})
    for cond in ("gaussian_noise_s1", "speckle_noise_s3", "contrast_s5", "lowlight_s1"):
        if cond in rob:
            v = rob[cond]
            L.append(f"| Classical enhancement with detector selection, {cond} | test subset | {pct(v['e2e_select'], 1)} | {pct(v['e2e'], 1)} | "
                     f"{100 * (v['e2e_select'] - v['e2e']):+.1f} | n = 1,960 |")
    if classical:
        b = classical["block_ablation_top1"]
        L.append(f"| Classical features: HOG block | val | {pct(classical['all']['top1'])} | {pct(b['without_hog'])} | "
                 f"{100 * (classical['all']['top1'] - b['without_hog']):+.2f} | shape carries the classical signal |")
        L.append(f"| Classical features: HSV colour block | val | {pct(classical['all']['top1'])} | {pct(b['without_hsv'])} | "
                 f"{100 * (classical['all']['top1'] - b['without_hsv']):+.2f} | |")
    L.append("")

    if fail:
        L += ["## 9. Failure-case analysis", "",
              f"{fail['n_wrong_top1']} of {test['n']} test images are wrong at Top-1 ({100 * fail['n_wrong_top1'] / test['n']:.2f}%).", "",
              "| Category | Images |", "|---|---|"]
        for k, v in sorted(fail["categories"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {k} | {v} |")
        L += ["", "Most frequent confusions:", "", "| True class | Predicted | Count |", "|---|---|---|"]
        for t in fail["top_confusions"][:10]:
            L.append(f"| {t['true']} | {t['predicted']} | {t['count']} |")
        L += ["", "Worked cases (figures in `experiments/2026-11-06_16-31-27/figures/`):", "",
              "1. **Trim twin:** a windowless Express Cargo Van predicted as Express Van at 100%. The CAM sits on the identical front fascia; the discriminative cue (side windows) is ignored.",
              "2. **Same-generation year twin:** Dodge Caliber 2007 predicted as 2012. Both years share one generation; the decision policy's family back-off is the correct output.",
              "3. **Probable label error:** labelled HUMMER H3T, predicted GMC Canyon at 96%; the truck carries a Chevrolet Z71 badge (Colorado, same platform as Canyon and H3T).",
              "4. **Missed detection in clutter:** a Lamborghini on a show stand behind a crowd; three small cars, none detected.",
              "5. **Rejected input:** 6 test images smaller than 64 px were refused by the input guard (since lowered to 32 px; the numbers above keep the original guard).", ""]

    pp = lambda v, d=1: f"{pct(v, d)}%"
    india = RUNS / "e18_test" / "test_report.json"
    if india.exists():
        ir = load(india)
        main_arm, base_arm = "e18v3_india_e05init", "e18v3_india_imagenet"
        a, b = ir[main_arm], ir[base_arm]
        L += ["## 10. Extension: Indian-market catalogue (E18)", "",
              "A second catalogue of car models sold or made in India, built from free-licensed Wikimedia Commons photos of models that Wikidata "
              "and Wikipedia link to the Indian market. Photos were cropped with the COCO detector and split 70/15/15 so that no photographer "
              f"appears in two splits. Test split: **{a['n']:,} photos, {a['classes_in_test']} models**, read once by `scripts/eval_india.py`.", "",
              "| Model (INT8, single view) | Top-1 (%) | 95% CI | Top-5 (%) | Macro Top-1 (%) | Make Top-1 (%) | ECE after temperature (%) |",
              "|---|---|---|---|---|---|---|"]
        for name, r in (("TResNet-L, started from the Stanford Cars model (deployed)", a), ("TResNet-L, started from ImageNet", b)):
            L.append(f"| {name} | {pct(r['top1'])} | {pct(r['top1_ci95'][0])}–{pct(r['top1_ci95'][1])} | {pct(r['top5'])} | {pct(r['macro_top1'])} | "
                     f"{pct(r['make_top1'])} | {pct(r['ece'])} |")
        m = ir.get("mcnemar")
        if m:
            L += ["", f"Starting from the Stanford Cars model is right on {m['a_only']} photos where the ImageNet start is wrong, and wrong on "
                      f"{m['b_only']} where it is right (McNemar p = {m['p_value']:.2g})."]
        us = ir.get("us_catalogue_on_india_test")
        if us:
            L += ["", "**Why a second catalogue:** the Stanford Cars model names a specific model for "
                      f"{pp(us['us_answers_a_model'])} of these Indian photos and flags {pp(us['us_flags_out_of_catalogue'])} as outside its catalogue. "
                      f"On the {us['images_with_brand_in_us_catalogue']:,} photos whose make also exists in Stanford Cars, it names the make correctly "
                      f"{pp(us['us_make_top1_on_shared_brands'])} of the time, against {pp(us['india_make_top1_on_shared_brands'])} for the Indian model."]
        lv = a["levels"]
        tot = sum(lv.values())
        L += ["", f"Decision levels on the test split with thresholds fitted on validation: model named {pp(lv.get('class', 0) / tot)} of the time "
                  f"(correct {pp(a['level_accuracy'].get('class', 0))}), make only {pp(lv.get('make', 0) / tot)}, not sure "
                  f"{pp(lv.get('abstain', 0) / tot)}, outside the catalogue {pp((lv.get('ood', 0) + lv.get('ood_make', 0)) / tot)}.", ""]
    hunt = ROOT / "configs" / "hunt_model.json"
    if hunt.exists():
        hm = json.loads(hunt.read_text())
        r = hm["report"]
        L += ["## 11. Extension: vehicle hunt match model (E19)", "",
              f"Match probability between a target photo and a camera track, fitted on {hm['n_vehicles']} Stanford Cars validation vehicles with simulated second-camera views "
              "and evaluated on a held-out half.", "",
              "| AUC | TPR at 1% FPR | TPR at 5% FPR (deployed threshold) | TPR at 10% FPR | FPR on another vehicle of the same model |",
              "|---|---|---|---|---|",
              f"| {r['auc_eval']:.3f} | {pp(r['tpr_eval_at_fpr']['0.01'])} | {pp(r['tpr_eval_at_fpr']['0.05'])} | {pp(r['tpr_eval_at_fpr']['0.1'])} | "
              f"{pp(r['fpr_eval_by_kind']['same model, other vehicle'])} |", ""]

    out = args.out or ROOT / f"eval_P5_{args.team}.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
