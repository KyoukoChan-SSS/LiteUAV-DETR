#!/usr/bin/env python3
"""Historical wall-clock B/C latency protocol for LiteUAVDETR.

This intentionally coexists with ``benchmarkonly.py``. It reproduces the
measurement semantics of our earlier wall-clock latency scripts:

* deploy the model and postprocessor before timing;
* B preprocesses every image outside one forward+postprocess wall-clock window;
* C times one continuous Image.open-to-postprocess wall-clock window;
* synchronize CUDA immediately before and after each timed window;
* report the population standard deviation over the complete sorted image set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse the historical protocol arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("B", "C"), required=True)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("-u", "--update", nargs="+", default=None)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument(
        "--input-size", nargs=2, type=int, default=(640, 640), metavar=("H", "W")
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--expected-images", type=int, default=548)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--tag", default="liteuavdetr_legacy")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Hash one input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_repo(repo: Path) -> Path:
    """Put the selected D-FINE-style repository first on the import path."""
    root = repo.expanduser().resolve()
    if not (root / "src").is_dir():
        raise FileNotFoundError(f"Repository does not contain src/: {root}")
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    importlib.invalidate_caches()
    return root


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    """Extract the inference state from a D-FINE checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return checkpoint
    if (
        "ema" in checkpoint
        and isinstance(checkpoint["ema"], dict)
        and "module" in checkpoint["ema"]
    ):
        return checkpoint["ema"]["module"]
    for key in ("model", "state_dict", "module"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Load and strictly deploy the selected LiteUAVDETR checkpoint."""
    setup_repo(args.repo)
    from src.core import YAMLConfig, yaml_utils

    importlib.import_module("src.zoo")
    updates = yaml_utils.parse_cli(args.update) if args.update else {}
    cfg = YAMLConfig(
        str(args.config), resume=str(args.weights) if args.weights else None, **updates
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model
    missing: Sequence[str] = ()
    unexpected: Sequence[str] = ()
    if args.weights:
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        incompatible = model.load_state_dict(
            checkpoint_state(args.weights), strict=False
        )
        missing = tuple(incompatible.missing_keys)
        unexpected = tuple(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={list(missing)[:20]}, unexpected={list(unexpected)[:20]}"
            )

    if hasattr(model, "deploy"):
        model = model.deploy()
    model.to(device).eval()

    postprocessor = cfg.postprocessor
    if hasattr(postprocessor, "deploy"):
        postprocessor = postprocessor.deploy()
    postprocessor.to(device).eval()
    metadata = {
        "checkpoint_missing_keys": len(missing),
        "checkpoint_unexpected_keys": len(unexpected),
        "model_class": type(model).__name__,
        "postprocessor_class": type(postprocessor).__name__,
        "postprocessor_deploy_mode": bool(getattr(postprocessor, "deploy_mode", False)),
    }
    return model, postprocessor, metadata


def list_images(folder: Path) -> list[Path]:
    """Return the deterministic recursive image set used by B and C."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def percentile(values: Sequence[float], quantile: float) -> float:
    """Compute one linearly interpolated percentile."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Summarize the full per-image latency population."""
    mean = statistics.fmean(values)
    return {
        "mean_ms": mean,
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "fps": 1000.0 / mean if mean > 0 else 0.0,
    }


def synchronize(device: torch.device) -> None:
    """Close a wall-clock timing window on CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_b(
    model: nn.Module,
    postprocessor: nn.Module,
    paths: Sequence[Path],
    transform: T.Compose,
    input_size: tuple[int, int],
    warmup: int,
    device: torch.device,
) -> list[float]:
    """Measure deployed forward and postprocess with preprocessing outside."""
    height, width = input_size
    warm_input = torch.randn(1, 3, height, width, device=device)
    warm_orig = torch.tensor([[width, height]], device=device)
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            postprocessor(model(warm_input), warm_orig)
        synchronize(device)

        for path in paths:
            image = Image.open(path).convert("RGB")
            width_orig, height_orig = image.size
            tensor = transform(image).unsqueeze(0).to(device)
            orig = torch.tensor([[width_orig, height_orig]], device=device)
            synchronize(device)
            start = time.perf_counter()
            postprocessor(model(tensor), orig)
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def measure_c(
    model: nn.Module,
    postprocessor: nn.Module,
    paths: Sequence[Path],
    transform: T.Compose,
    warmup: int,
    device: torch.device,
) -> list[float]:
    """Measure one continuous image-open through postprocess window."""
    warmup_path = paths[0]
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            image = Image.open(warmup_path).convert("RGB")
            width_orig, height_orig = image.size
            tensor = transform(image).unsqueeze(0).to(device)
            orig = torch.tensor([[width_orig, height_orig]], device=device)
            postprocessor(model(tensor), orig)
        synchronize(device)

        for path in paths:
            synchronize(device)
            start = time.perf_counter()
            image = Image.open(path).convert("RGB")
            width_orig, height_orig = image.size
            tensor = transform(image).unsqueeze(0).to(device)
            orig = torch.tensor([[width_orig, height_orig]], device=device)
            postprocessor(model(tensor), orig)
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def main() -> None:
    """Run one historical B or C pass and save every timing sample."""
    args = parse_args()
    args.repo = args.repo.resolve()
    args.config = args.config.resolve()
    args.image_folder = args.image_folder.resolve()
    if args.weights:
        args.weights = args.weights.resolve()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.image_folder.is_dir():
        raise FileNotFoundError(args.image_folder)

    paths = list_images(args.image_folder)
    if not paths:
        raise RuntimeError(f"No images found under {args.image_folder}")
    if args.expected_images and len(paths) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} images, found {len(paths)}"
        )

    device = torch.device(args.device)
    model, postprocessor, metadata = load_model(args, device)
    height, width = int(args.input_size[0]), int(args.input_size[1])
    transform = T.Compose([T.Resize((height, width)), T.ToTensor()])
    if args.mode == "B":
        samples = measure_b(
            model, postprocessor, paths, transform, (height, width), args.warmup, device
        )
        window = "forward + postprocess; preprocessing outside"
    else:
        samples = measure_c(model, postprocessor, paths, transform, args.warmup, device)
        window = "Image.open + transform + H2D + forward + postprocess"

    result = {
        "protocol": "legacy_wallclock_bc_v1",
        "mode": args.mode,
        "window": window,
        "tag": args.tag,
        "family": "liteuavdetr",
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "weights": str(args.weights) if args.weights else "",
        "weights_sha256": sha256(args.weights) if args.weights else "",
        "device": str(device),
        "input_size": [height, width],
        "warmup": args.warmup,
        "num_images": len(paths),
        "deploy": True,
        "metadata": metadata,
        "latency": summarize(samples),
        "per_image_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    latency = result["latency"]
    print(
        f"[{args.mode} legacy] mean={latency['mean_ms']:.3f}ms std={latency['std_ms']:.3f}ms "
        f"p50={latency['p50_ms']:.3f}ms p95={latency['p95_ms']:.3f}ms "
        f"fps={latency['fps']:.1f} n={len(samples)}"
    )
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
