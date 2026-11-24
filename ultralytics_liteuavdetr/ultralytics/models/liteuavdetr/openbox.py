# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Public beginner interface for LiteUAVDETR OpenBox.

LiteUAVDETR is derived from D-FINE (https://github.com/Peterande/D-FINE).
OpenBox adds standard Ultralytics dataset ergonomics without changing the model.
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

from ultralytics.nn.liteuavdetr.task import LiteUAVDETRDetectionModel

from .model import LiteUAVDETR
from .openbox_train import LiteUAVDETROpenBoxTrainer
from .openbox_val import LiteUAVDETROpenBoxValidator
from .predict import LiteUAVDETRPredictor


class LiteUAVDETROpenBox(LiteUAVDETR):
    """Run LiteUAVDETR with a standard Ultralytics ``data.yaml`` and YOLO labels."""

    @property
    def task_map(self):
        """Map training and validation to OpenBox while reusing the released model and predictor."""
        return {
            "detect": {
                "model": LiteUAVDETRDetectionModel,
                "trainer": LiteUAVDETROpenBoxTrainer,
                "validator": LiteUAVDETROpenBoxValidator,
                "predictor": LiteUAVDETRPredictor,
            }
        }

    def train(self, trainer=None, **kwargs):
        """Require an explicit beginner-style data YAML, then run the released training recipe."""
        if not kwargs.get("data") and not self.overrides.get("data"):
            raise ValueError("LiteUAVDETR OpenBox requires data='/path/to/data.yaml'.")
        return super().train(trainer=trainer, **kwargs)
