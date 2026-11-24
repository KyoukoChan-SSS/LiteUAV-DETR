# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ultralytics task wrapper for the D-FINE-derived LiteUAVDETR detector.

LiteUAVDETR integration Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from ultralytics.nn.tasks import BaseModel, yaml_model_load
from ultralytics.utils import LOGGER

from .builder import build_liteuavdetr, build_liteuavdetr_postprocessor, load_dfine_checkpoint
from .liteuavdetr_criterion import LiteUAVDETRCriterion
from .matcher import HungarianMatcher


class LiteUAVDETRDetectionModel(BaseModel):
    """Wrap the released LiteUAVDETR network in the Ultralytics BaseModel contract."""

    def __init__(self, cfg="liteuavdetr-ultrafast.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        if ch != 3:
            raise ValueError(f"LiteUAVDETR currently requires three-channel RGB input, got ch={ch}.")

        self.yaml = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        self.nc = int(nc if nc is not None else self.yaml.get("nc", 80))
        self.yaml["nc"] = self.nc
        core = build_liteuavdetr(self.yaml, nc=self.nc)
        core.i, core.f, core.type = 0, -1, "LiteUAVDETR"
        core.np = sum(parameter.numel() for parameter in core.parameters())
        self.model = nn.Sequential(core)
        self.postprocessor = build_liteuavdetr_postprocessor(self.yaml, nc=self.nc)
        self.save = []
        configured_names = self.yaml.get("names")
        self.names = (
            {int(index): str(name) for index, name in configured_names.items()}
            if isinstance(configured_names, dict)
            else {index: str(index) for index in range(self.nc)}
        )
        self.stride = torch.tensor([32.0])
        self.task = "detect"
        self.args = {"task": "detect"}
        self.export = False
        self.end2end = False
        self.liteuavdetr = True
        if verbose:
            LOGGER.info(
                f"LiteUAVDETR {self.yaml.get('variant', 'custom')}: "
                f"{core.np:,} parameters, {self.nc} classes"
            )

    @property
    def core(self):
        """Return the checkpoint-compatible LiteUAVDETR module."""
        return self.model[0]

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """Run the exact LiteUAVDETR forward path and return its prediction dictionary."""
        if augment:
            LOGGER.warning("LiteUAVDETR does not support augment=True; using single-scale inference.")
        if embed is not None:
            raise NotImplementedError("LiteUAVDETR feature embedding output is not implemented.")
        outputs = self.core(x, targets=batch)
        return self._export_predictions(outputs) if self.export else outputs

    def _export_predictions(self, outputs):
        """Return normalized ``cxcywh, score, class`` rows for ONNX/TensorRT."""
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        num_classes = logits.shape[-1]
        topk = min(self.postprocessor.num_top_queries, logits.shape[1] * num_classes)
        scores, flat_index = torch.topk(logits.sigmoid().flatten(1), topk, dim=-1)
        labels = flat_index - (flat_index // num_classes) * num_classes
        query_index = flat_index // num_classes
        boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
        return torch.cat((boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)), dim=-1)

    def load_dfine(self, weights: str | Path, strict: bool = True):
        """Load an original D-FINE-style .pth checkpoint into the core network."""
        checkpoint = load_dfine_checkpoint(self.core, weights, strict=strict)
        LOGGER.info(f"Loaded D-FINE checkpoint with strict={strict}: {weights}")
        return checkpoint

    def load(self, weights, verbose=True):
        """Load either a released D-FINE .pth or a native Ultralytics model."""
        if isinstance(weights, (str, Path)) and Path(weights).suffix == ".pth":
            self.load_dfine(weights, strict=True)
            return
        return super().load(weights, verbose=verbose)

    def init_criterion(self):
        """Build the released D-FINE Hungarian matcher and fine-grained criterion."""
        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        return LiteUAVDETRCriterion(
            matcher=matcher,
            weight_dict={
                "loss_vfl": 1,
                "loss_bbox": 5,
                "loss_giou": 2,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            losses=["vfl", "boxes", "local"],
            alpha=0.75,
            gamma=2.0,
            num_classes=self.nc,
            reg_max=32,
        )

    @staticmethod
    def _build_targets(batch):
        """Convert an Ultralytics detection batch to D-FINE's per-image target list."""
        batch_idx = batch["batch_idx"].long().view(-1)
        targets = []
        for image_index in range(batch["img"].shape[0]):
            selected = batch_idx == image_index
            targets.append(
                {
                    "labels": batch["cls"][selected].long().view(-1),
                    "boxes": batch["bboxes"][selected].to(dtype=batch["img"].dtype),
                }
            )
        return targets

    def loss(self, batch, preds=None):
        """Compute the complete weighted D-FINE loss, including auxiliary and denoising terms."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        targets = self._build_targets(batch)
        if preds is None:
            preds = self.predict(batch["img"], batch=targets)
        loss_dict = self.criterion(preds, targets)
        total = sum(loss_dict.values())

        loss_items = {}
        for name in ("loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf"):
            values = [value.detach() for key, value in loss_dict.items() if key == name or key.startswith(f"{name}_")]
            loss_items[name.removeprefix("loss_")] = torch.stack(values).sum()
        return total, loss_items

    def fuse(self, verbose=True, imgsz=640):
        """Deploy only the export copy; normal evaluation remains bit-exact and unfused."""
        if self.export:
            self.deploy()
        return self

    def deploy(self):
        """Explicitly convert supported blocks to deployment form for export/benchmarking."""
        self.core.deploy()
        return self
