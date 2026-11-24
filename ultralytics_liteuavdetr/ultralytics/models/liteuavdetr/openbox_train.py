# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Beginner-friendly standard-label trainer for LiteUAVDETR OpenBox.

The optimizer, loss, EMA, warmup, augmentation stages, and checkpoint policy are
inherited unchanged from the D-FINE-derived LiteUAVDETR release trainer.
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

from copy import copy

from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .openbox_dataset import LiteUAVDETROpenBoxDataset
from .openbox_val import LiteUAVDETROpenBoxValidator
from .train import LiteUAVDETRTrainer


class LiteUAVDETROpenBoxTrainer(LiteUAVDETRTrainer):
    """Train the released LiteUAVDETR recipe from standard Ultralytics labels."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build a YOLO-label dataset without changing the LiteUAVDETR training recipe."""
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return LiteUAVDETROpenBoxDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task="detect",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            stop_epoch=self._recipe()["stop_epoch"],
        )

    def get_validator(self):
        """Return the standard-label OpenBox validator for training and final evaluation."""
        return LiteUAVDETROpenBoxValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
