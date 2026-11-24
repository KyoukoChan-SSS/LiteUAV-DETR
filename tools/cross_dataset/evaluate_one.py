#!/usr/bin/env python3
"""Evaluate one LiteUAVDETR checkpoint/dataset pair and write stable JSON.

This is an additive cross-dataset entry point.  It deliberately calls the same
YAMLConfig, solver setup, checkpoint loader, evaluation engine, postprocessor,
and CocoEvaluator as ``train.py --test-only`` instead of introducing a second
accuracy implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LITE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LITE_ROOT))

COCO_BBOX_METRICS = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)


def _metric_dict(stats: list[float]) -> dict[str, float | None]:
    """Map COCO's fixed 12-value vector and represent undefined -1 as null."""
    if len(stats) != len(COCO_BBOX_METRICS):
        raise RuntimeError(
            f"Expected {len(COCO_BBOX_METRICS)} COCO bbox statistics, got {len(stats)}"
        )
    return {
        name: (None if float(value) < 0 else float(value))
        for name, value in zip(COCO_BBOX_METRICS, stats)
    }


def _atomic_json_dump(data: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def evaluate_one(args: argparse.Namespace) -> dict[str, Any]:
    # Keep heavyweight imports out of --help and the orchestration/test process.
    import torch

    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS
    from src.solver.det_engine import evaluate

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    started = time.perf_counter()
    try:
        overrides = {
            "resume": str(args.checkpoint),
            "device": args.device,
            "output_dir": str(args.work_dir),
            "val_dataloader": {
                "total_batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "dataset": {
                    "img_folder": str(args.images),
                    "ann_file": str(args.annotations),
                },
            },
        }
        cfg = YAMLConfig(str(args.config), **overrides)

        actual_num_classes = int(cfg.yaml_cfg.get("num_classes", -1))
        if actual_num_classes != args.expected_num_classes:
            raise RuntimeError(
                "Cross-dataset category contract mismatch: "
                f"manifest expects num_classes={args.expected_num_classes}, "
                f"but config has num_classes={actual_num_classes}. Use a VisDrone release config."
            )
        if cfg.yaml_cfg.get("remap_mscoco_category", False):
            raise RuntimeError(
                "Cross-dataset annotations already use VisDrone IDs; "
                "remap_mscoco_category must be false."
            )

        # Match train.py's checkpoint behavior: do not fetch backbone weights
        # when a full release checkpoint is being restored.
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        module = solver.ema.module if solver.ema else solver.model
        test_stats, _ = evaluate(
            module,
            solver.criterion,
            solver.postprocessor,
            solver.val_dataloader,
            solver.evaluator,
            solver.device,
            epoch=-1,
            use_wandb=False,
        )
        bbox_stats = [float(value) for value in test_stats["coco_eval_bbox"]]
        result = {
            "schema_version": 1,
            "status": "ok",
            "dataset": args.dataset_name,
            "protocol": args.protocol,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "device": args.device,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "images": str(args.images.resolve()),
            "annotations": str(args.annotations.resolve()),
            "coco_bbox_stats": bbox_stats,
            "metrics": _metric_dict(bbox_stats),
            "versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        }
        _atomic_json_dump(result, args.output)
        return result
    finally:
        dist_utils.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one COCO-format cross-dataset target with LiteUAVDETR."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--expected-num-classes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-method", default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    evaluate_one(args)


if __name__ == "__main__":
    main()
