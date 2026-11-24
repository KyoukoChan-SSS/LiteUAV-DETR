# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train the released LiteUAVDETR recipes through the Ultralytics API.

The defaults are the public counterparts of the launch settings used for the
validated UltraFast run. The model, loss, and training recipe are derived from
D-FINE (https://github.com/Peterande/D-FINE).
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

from ultralytics import LiteUAVDETR  # noqa: E402


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
    """Parse the portable fields around the fixed release recipes."""
    parser = argparse.ArgumentParser(
        description="Train LiteUAVDETR UltraFast or Fast with the released D-FINE-aligned recipe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", choices=RELEASE_RECIPES, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Local VisDrone dataset YAML.")
    parser.add_argument("--device", default="0", help="Ultralytics device: 0, 0,1, or cpu.")
    parser.add_argument(
        "--project",
        type=Path,
        default=REPO_ROOT / "runs" / "train",
        help="Directory in which Ultralytics creates the run folder.",
    )
    parser.add_argument("--name", help="Run directory name; an automatic variant-specific name is used by default.")
    return parser.parse_args()


def main() -> None:
    """Run one fixed LiteUAVDETR release recipe."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")

    project = args.project.expanduser().resolve()
    variant = args.variant
    recipe = RELEASE_RECIPES[variant]
    model_cfg = REPO_ROOT / "ultralytics" / "cfg" / "models" / "liteuavdetr" / f"liteuavdetr-{variant}.yaml"
    name = args.name or f"liteuavdetr-{variant}-visdrone-640"

    # Match the working launcher's current directory so HGNetV2 stage-1 weights
    # are downloaded to the repository's ignored weight/hgnetv2 directory.
    os.chdir(REPO_ROOT)
    model = LiteUAVDETR(model_cfg)
    metrics = model.train(
        data=data,
        epochs=recipe["epochs"],
        batch=8,
        imgsz=640,
        workers=recipe["workers"],
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
        project=project,
        name=name,
        exist_ok=False,
    )
    print(
        "LITEUAVDETR_TRAIN_DONE",
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
