#!/usr/bin/env python3
"""
LiteUAVDETR precision-aware ONNX exporter.

Adapted from D-FINE's deployment flow:
https://github.com/Peterande/D-FINE
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig  # noqa: E402
import src.zoo  # noqa: F401,E402


class DeployModel(nn.Module):
    def __init__(self, cfg: YAMLConfig) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor) -> Any:
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--dynamic-batch", action="store_true", default=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_state(path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            return ckpt["ema"]["module"]
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def main() -> None:
    args = parse_args()
    cfg = YAMLConfig(args.config)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    incompatible = cfg.model.load_state_dict(load_state(args.resume), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint does not match config:\n"
            f"missing_keys={incompatible.missing_keys[:20]} total={len(incompatible.missing_keys)}\n"
            f"unexpected_keys={incompatible.unexpected_keys[:20]} total={len(incompatible.unexpected_keys)}"
        )

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = DeployModel(cfg).to(device).eval()
    image_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    if args.precision == "fp16":
        model.half()

    images = torch.rand(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        dtype=image_dtype,
        device=device,
    )
    sizes = torch.tensor(
        [[args.image_size, args.image_size]] * args.batch_size,
        dtype=torch.int64,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "images": {0: "N"},
            "orig_target_sizes": {0: "N"},
            "labels": {0: "N"},
            "boxes": {0: "N"},
            "scores": {0: "N"},
        }

    torch.onnx.export(
        model,
        (images, sizes),
        str(args.output),
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,
    )

    import onnx

    onnx_model = onnx.load(str(args.output))
    onnx.checker.check_model(onnx_model)
    print(f"saved {args.precision} ONNX: {args.output}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
