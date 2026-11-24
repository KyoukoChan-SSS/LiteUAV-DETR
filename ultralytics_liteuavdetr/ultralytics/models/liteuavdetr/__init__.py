# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ultralytics-native interface for the D-FINE-derived LiteUAVDETR model family.

LiteUAVDETR integration Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from .model import LiteUAVDETR
from .predict import LiteUAVDETRPredictor
from .train import LiteUAVDETRTrainer
from .val import LiteUAVDETRValidator

__all__ = (
    "LiteUAVDETR",
    "LiteUAVDETRPredictor",
    "LiteUAVDETRTrainer",
    "LiteUAVDETRValidator",
)
