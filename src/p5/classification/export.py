from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from p5.classification.model import build


class WithFeatures(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        f = self.model.forward_features(x)
        return self.model.forward_head(f), f


def load_model(ckpt: Path) -> tuple[nn.Module, dict]:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build(state["backbone"], state["num_classes"], pretrained=False)
    model.load_state_dict(state["model"])
    return model.eval(), state


def export(ckpt: Path, out_dir: Path, opset: int = 17) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, state = load_model(ckpt)
    wrapped = WithFeatures(model).eval()
    size = state["size"]
    x = torch.randn(2, 3, size, size)
    onnx_path = out_dir / "classifier.onnx"
    torch.onnx.export(wrapped, x, onnx_path, input_names=["input"], output_names=["logits", "features"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
                      opset_version=opset, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref_logits, ref_feat = wrapped(x)
    got_logits, got_feat = sess.run(None, {"input": x.numpy()})
    diff = float(np.abs(got_logits - ref_logits.numpy()).max())
    if diff > 1e-3:
        raise AssertionError(f"ONNX logits differ from PyTorch by {diff}")

    head = model.get_classifier()
    np.savez(out_dir / "head.npz", fc_weight=head.weight.detach().numpy(), fc_bias=head.bias.detach().numpy())
    meta = {"backbone": state["backbone"], "size": size, "pad": 0.10, "mean": state["mean"], "std": state["std"],
            "interpolation": "bilinear", "feature_dim": int(got_feat.shape[1]), "feature_hw": list(got_feat.shape[2:]),
            "num_classes": state["num_classes"], "precision": "fp32", "parity_max_abs_diff": diff,
            "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest()}
    (out_dir / "classifier_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export a trained classifier to ONNX with logits and feature map")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    print(json.dumps(export(args.ckpt, args.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
