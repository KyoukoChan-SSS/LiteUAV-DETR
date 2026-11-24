# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train LiteUAVDETR OpenBox from a standard Ultralytics detection dataset.

OpenBox changes only the dataset interface. The model, loss, optimizer, EMA,
augmentation stages, and checkpoint policy remain the released LiteUAVDETR recipe.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics-config"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(REPO_ROOT))

from ultralytics.models.liteuavdetr.openbox import LiteUAVDETROpenBox  # noqa: E402


RELEASE_RECIPES = {
    "ultrafast": {
        "epochs": 220,
        "workers": 12,
        "lr0": 0.0008,
        "weight_decay": 0.0001,
    },
    "fast": {
        "epochs": 200,
        "workers": 8,
        "lr0": 0.0004,
        "weight_decay": 0.000065,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse beginner-facing fields while keeping release settings as defaults."""
    parser = argparse.ArgumentParser(
        description="Train LiteUAVDETR OpenBox with standard images/labels folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True, help="Standard Ultralytics data.yaml.")
    parser.add_argument("--variant", choices=RELEASE_RECIPES, default="ultrafast")
    parser.add_argument("--device", default="0", help="Ultralytics device: 0, 0,1, or cpu.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, help="Optional smoke-test override; omit for the release recipe.")
    parser.add_argument("--workers", type=int, help="Optional worker override; omit for the release recipe.")
    parser.add_argument(
        "--project",
        type=Path,
        default=REPO_ROOT / "runs" / "train_openbox",
        help="Directory in which Ultralytics creates the run folder.",
    )
    parser.add_argument("--name", help="Run directory name; an automatic OpenBox name is used by default.")
    return parser.parse_args()


def main() -> None:
    """Run one LiteUAVDETR OpenBox training job."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")

    variant = args.variant
    recipe = RELEASE_RECIPES[variant]
    model_cfg = REPO_ROOT / "ultralytics" / "cfg" / "models" / "liteuavdetr" / f"liteuavdetr-{variant}.yaml"
    name = args.name or f"liteuavdetr-{variant}-openbox-640"

    os.chdir(REPO_ROOT)
    model = LiteUAVDETROpenBox(model_cfg)
    metrics = model.train(
        data=data,
        epochs=args.epochs or recipe["epochs"],
        batch=args.batch,
        imgsz=640,
        workers=recipe["workers"] if args.workers is None else args.workers,
        optimizer="AdamW",
        lr0=recipe["lr0"],
        weight_decay=recipe["weight_decay"],
        nbs=8,
        amp=False,
        patience=0,
        val=True,
        save=True,
        save_period=-1,
        close_mosaic=0,
        multi_scale=0.0,
        compile=False,
        channels_last=False,
        deterministic=True,
        seed=0,
        pretrained=True,
        fraction=1.0,
        cache=False,
        plots=False,
        device=args.device,
        project=args.project.expanduser().resolve(),
        name=name,
        exist_ok=False,
    )
    print(
        "LITEUAVDETR_OPENBOX_TRAIN_DONE",
        "variant",
        variant,
        "map",
        format(metrics.box.map, ".15f") if metrics else None,
        "map50",
        format(metrics.box.map50, ".15f") if metrics else None,
        "save_dir",
        model.trainer.save_dir,
    )


if __name__ == "__main__":
    main()
