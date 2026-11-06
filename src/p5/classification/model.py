from __future__ import annotations

import re

import torch
import torch.nn as nn

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HF_PREFIX = "hf-ijepa:"


class IJepaClassifier(nn.Module):
    def __init__(self, model_id: str, num_classes: int, pretrained: bool = True, grad_ckpt: bool = True):
        super().__init__()
        from transformers import IJepaConfig, IJepaModel
        if model_id == "tiny":
            cfg = IJepaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                              image_size=224, patch_size=14)
            self.encoder = IJepaModel(cfg, add_pooling_layer=False)
            mean, std = IMAGENET
        else:
            if pretrained:
                self.encoder = IJepaModel.from_pretrained(model_id, add_pooling_layer=False)
            else:
                self.encoder = IJepaModel(IJepaConfig.from_pretrained(model_id), add_pooling_layer=False)
            try:
                from transformers import AutoImageProcessor
                proc = AutoImageProcessor.from_pretrained(model_id)
                mean, std = tuple(proc.image_mean), tuple(proc.image_std)
            except Exception:
                mean, std = IMAGENET
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size
        self.num_layers = self.encoder.config.num_hidden_layers
        self.num_features = hidden
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.pretrained_cfg = {"mean": mean, "std": std, "interpolation": "bicubic"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(1)

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.pool(h)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    def get_classifier(self) -> nn.Module:
        return self.head

    def layer_id(self, name: str) -> int:
        if name.startswith("encoder.embeddings"):
            return 0
        m = re.match(r"encoder\.layers\.(\d+)\.", name)
        if m:
            return int(m.group(1)) + 1
        return self.num_layers + 1


def build(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if backbone.startswith(HF_PREFIX):
        return IJepaClassifier(backbone[len(HF_PREFIX):], num_classes, pretrained)
    import timm
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def data_config(model: nn.Module) -> dict:
    if isinstance(model, IJepaClassifier):
        return dict(model.pretrained_cfg)
    import timm
    cfg = timm.data.resolve_model_data_config(model)
    return {"mean": tuple(cfg["mean"]), "std": tuple(cfg["std"]), "interpolation": cfg.get("interpolation", "bilinear")}
