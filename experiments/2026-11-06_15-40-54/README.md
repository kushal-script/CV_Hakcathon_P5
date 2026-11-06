# E11: ONNX export and static INT8 quantization

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_15-40-54` |
| Type | optimisation |
| Author | Kushal |
| Started | 2026-11-06 14:40 IST (folder created 15:40) |
| Status | done |
| Compute | JarvisLabs instance CPU (AMD EPYC 7542), run `r_72de10db`; also local Apple M3 |

## Hypothesis
Static INT8 quantization of the E05 classifier loses ≤ 1 point Top-1, shrinks the model about 4×, and speeds up CPU inference.

## Setup
- Export: PyTorch → ONNX opset 17 with two outputs (`logits`, `features` for the CAM). Parity with PyTorch: max |Δlogit| = 6.6e-7.
- Quantization: ONNX Runtime `quantize_static`, QDQ format, signed INT8 weights and activations, per-channel weights, percentile calibration on 256 training crops (seed 42), after `quant_pre_process`.
- Evaluation: all 815 val GT crops, batch 1, 2 intra-op threads (to mimic a small CPU host).
- Local calibration on the Mac (16 GB) was killed by the OS: percentile calibration keeps every activation of every calibration image. It was moved to the 112 GB instance.

## How to reproduce
```bash
bash scripts/remote/quantize.sh e05_tresnet_l_368
```

## Results
Split: **val (n = 815), GT crops**

| Precision | Top-1 | Make Top-1 | Size (MB) | Latency p50, 2 threads, AMD EPYC 7542 | Latency p50, 2 threads, Apple M3 |
|---|---|---|---|---|---|
| FP32 | 95.34 | 99.02 | 206.1 | 354 ms | 413 ms |
| INT8 | **94.97** (−0.37) | 99.02 | **52.9** (3.9× smaller) | 434 ms | **125 ms** (3.3× faster) |

On detector crops (the deployed input) the INT8 model reaches 95.09% val Top-1.

## Observations
- Accuracy cost is small (−0.37 points) and make-level accuracy is unchanged.
- Speed depends on the CPU: the EPYC 7542 (Zen 2) has no AVX-512 or VNNI integer instructions, so INT8 runs slower than FP32 there, while the ARM M3 gets a 3.3× speed-up. This matches the ONNX Runtime documentation's warning about older x86 CPUs.

## Conclusion & next step
- Both precisions are packaged. The deployed precision is chosen by measuring on the actual hosting CPU: INT8 where VNNI or ARM dot-product instructions exist, otherwise FP32. Either is far inside the 10 s budget.

## Artifacts
`runs/remote/work-runs/e05_tresnet_l_368/export_int8/` (`classifier_int8.onnx`, `quantization.json`, val logits for both precisions).
