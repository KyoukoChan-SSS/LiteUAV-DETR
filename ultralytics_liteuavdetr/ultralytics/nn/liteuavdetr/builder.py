# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Standalone LiteUAVDETR builders and D-FINE checkpoint compatibility.

The detector construction follows D-FINE (https://github.com/Peterande/D-FINE).
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .hgnetv2 import HGNetv2
from .liteuavdetr import LiteUAVDETR
from .liteuavdetr_decoder import LiteUAVDETRTransformer
from .liteuavdetr_encoder import LiteUAVDETREncoder
from .liteuavdetr_postprocessor import LiteUAVDETRPostProcessor


def build_liteuavdetr(cfg: dict[str, Any], nc: int | None = None) -> LiteUAVDETR:
    """Build a standalone LiteUAVDETR network from an Ultralytics model dictionary."""
    cfg = deepcopy(cfg)
    num_classes = int(nc if nc is not None else cfg.get("nc", 80))
    eval_spatial_size = list(cfg.get("eval_spatial_size", [640, 640]))

    backbone_cfg = deepcopy(cfg["liteuav_backbone"])
    backbone_cfg.setdefault("pretrained", False)
    backbone = HGNetv2(**backbone_cfg)

    encoder_cfg = deepcopy(cfg["liteuav_encoder"])
    encoder_cfg["eval_spatial_size"] = eval_spatial_size
    encoder = LiteUAVDETREncoder(**encoder_cfg)

    decoder_cfg = deepcopy(cfg["liteuav_decoder"])
    decoder_cfg["num_classes"] = num_classes
    decoder_cfg["eval_spatial_size"] = eval_spatial_size
    decoder = LiteUAVDETRTransformer(**decoder_cfg)

    return LiteUAVDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def build_liteuavdetr_postprocessor(
    cfg: dict[str, Any], nc: int | None = None
) -> LiteUAVDETRPostProcessor:
    """Build the exact D-FINE/LiteUAVDETR top-k postprocessor."""
    post_cfg = deepcopy(cfg.get("liteuav_postprocessor", {}))
    post_cfg["num_classes"] = int(nc if nc is not None else cfg.get("nc", 80))
    return LiteUAVDETRPostProcessor(**post_cfg)


def get_dfine_checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract the model state dictionary from a released D-FINE-style checkpoint."""
    if not isinstance(checkpoint, dict):
        raise TypeError("A D-FINE checkpoint must contain a dictionary.")

    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        state = ema["module"]
    elif isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif all(isinstance(key, str) for key in checkpoint):
        state = checkpoint
    else:
        raise KeyError("Checkpoint has neither ema.module nor model state.")

    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_dfine_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load an original LiteUAVDETR/D-FINE checkpoint without rewriting it."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = get_dfine_checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=strict)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            f"Strict checkpoint load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint


def infer_dfine_variant(checkpoint_path: str | Path) -> str:
    """Infer UltraFast or Fast from the HGNetV2 stem stored in a checkpoint."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    state = get_dfine_checkpoint_state(checkpoint)
    stem = state.get("backbone.stem.stem1.conv.weight")
    if stem is None:
        raise KeyError("Cannot infer LiteUAVDETR variant: HGNetV2 stem weight is missing.")
    if stem.shape[0] == 16:
        return "ultrafast"
    if stem.shape[0] == 24:
        return "fast"
    raise ValueError(f"Unsupported HGNetV2 stem width {stem.shape[0]}.")
