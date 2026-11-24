# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Public Ultralytics interface for LiteUAVDETR.

LiteUAVDETR is derived from D-FINE (https://github.com/Peterande/D-FINE).
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.nn.liteuavdetr import LiteUAVDETRDetectionModel, infer_dfine_variant
from ultralytics.utils import ROOT

from .predict import LiteUAVDETRPredictor
from .train import LiteUAVDETRTrainer
from .val import LiteUAVDETRValidator


class LiteUAVDETR(Model):
    """Ultralytics-native interface for LiteUAVDETR UltraFast and Fast."""

    def __init__(self, model: str | Path | Model = "liteuavdetr-ultrafast.yaml", verbose=False):
        if isinstance(model, Model):
            super().__init__(model=model, task="detect", verbose=verbose)
            return

        model_path = Path(model)
        if model_path.suffix == ".pth":
            variant = infer_dfine_variant(model_path)
            cfg = ROOT / "cfg" / "models" / "liteuavdetr" / f"liteuavdetr-{variant}.yaml"
            super().__init__(model=str(cfg), task="detect", verbose=verbose)
            self.model.load_dfine(model_path, strict=True)
            self.ckpt_path = str(model_path)
            self.model.pt_path = str(model_path)
            self.model_name = str(model_path)
            self.overrides["pretrained"] = str(model_path)
        else:
            super().__init__(model=str(model), task="detect", verbose=verbose)

    @property
    def task_map(self):
        """Map detection modes to LiteUAVDETR-specific Ultralytics components."""
        return {
            "detect": {
                "model": LiteUAVDETRDetectionModel,
                "trainer": LiteUAVDETRTrainer,
                "validator": LiteUAVDETRValidator,
                "predictor": LiteUAVDETRPredictor,
            }
        }

    def train(self, trainer=None, **kwargs):
        """Train with the final D-FINE-derived recipe while allowing explicit overrides."""
        recipe = self.model.yaml["liteuav_train"]
        variant = self.model.yaml.get("variant", "ultrafast")
        defaults = {
            "data": str(ROOT / "cfg" / "datasets" / "VisDrone-LiteUAVDETR.yaml"),
            "epochs": int(recipe["epochs"]),
            "batch": 8,
            "imgsz": 640,
            "workers": 12 if variant == "ultrafast" else 8,
            "optimizer": "AdamW",
            "lr0": float(recipe["lr"]),
            "weight_decay": float(recipe["weight_decay"]),
            "nbs": 8,
            "amp": False,
            "patience": 0,
            "val": True,
            "save": True,
            "close_mosaic": 0,
            "multi_scale": 0.0,
            "compile": False,
            "channels_last": False,
            "seed": 0,
        }
        if "pretrained" not in self.overrides:
            # Match the released D-FINE recipe when training from a model YAML.
            # Full LiteUAVDETR checkpoints set their own pretrained path and do
            # not reload the HGNetV2 stage-1 initialization.
            defaults["pretrained"] = True
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        return super().train(trainer=trainer, **kwargs)

    def export(self, **kwargs):
        """Export a deployed one-input LiteUAVDETR graph with embedded global top-k."""
        kwargs.setdefault("imgsz", 640)
        kwargs.setdefault("opset", 17)
        kwargs.setdefault("dynamic", False)
        kwargs.setdefault("nms", False)
        kwargs.setdefault("simplify", False)
        if str(kwargs.get("format", "")).lower() == "engine":
            # LiteUAVDETR: the validated TensorRT path is explicit FP16 ONNX plus a strongly-typed network.
            kwargs.setdefault("quantize", 16)
        previous_export = self.model.export
        previous_end2end = self.model.end2end
        self.model.export = True
        self.model.end2end = True
        try:
            return super().export(**kwargs)
        finally:
            self.model.export = previous_export
            self.model.end2end = previous_end2end
