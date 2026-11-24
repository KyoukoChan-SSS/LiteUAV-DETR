# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Standard Ultralytics-label validation for LiteUAVDETR OpenBox.

The model output and global top-k postprocess remain identical to LiteUAVDETR.
OpenBox reports regular Ultralytics detection metrics because no COCO JSON is required.
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

import torch

from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import colorstr

from .openbox_dataset import LiteUAVDETROpenBoxDataset
from .val import LiteUAVDETRValidator


class LiteUAVDETROpenBoxValidator(LiteUAVDETRValidator):
    """Validate LiteUAVDETR against standard zero-indexed Ultralytics labels."""

    def init_metrics(self, model):
        """Initialize normal Ultralytics metrics without requiring COCO JSON output."""
        RTDETRValidator.init_metrics(self, model)
        self.is_coco = False
        self.is_lvis = False
        self.args.save_json = False
        native_model = getattr(model, "model", None)
        self.num_top_queries = getattr(getattr(native_model, "postprocessor", None), "num_top_queries", 350)

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build the OpenBox validation dataset from standard images and labels folders."""
        return LiteUAVDETROpenBoxDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=32,
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task="detect",
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
        )

    def scale_preds(self, predn, pbatch):
        """Scale normalized predictions to native pixels when optional TXT output is requested."""
        height, width = pbatch["ori_shape"]
        scale = torch.tensor((width, height, width, height), device=predn["bboxes"].device)
        return {**predn, "bboxes": predn["bboxes"].clone() * scale}

    def get_stats(self):
        """Use ordinary Ultralytics mAP50-95 as OpenBox checkpoint fitness."""
        return RTDETRValidator.get_stats(self)

    def eval_json(self, stats):
        """Return standard metrics because OpenBox deliberately has no COCO annotations."""
        return stats
