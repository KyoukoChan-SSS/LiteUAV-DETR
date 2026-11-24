# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D-FINE-derived LiteUAVDETR network, loss, postprocessing, and task model.

LiteUAVDETR integration Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from .builder import (
    build_liteuavdetr,
    build_liteuavdetr_postprocessor,
    get_dfine_checkpoint_state,
    infer_dfine_variant,
    load_dfine_checkpoint,
)
from .liteuavdetr import LiteUAVDETR
from .liteuavdetr_criterion import LiteUAVDETRCriterion
from .liteuavdetr_decoder import LiteUAVDETRTransformer
from .liteuavdetr_encoder import LiteUAVDETREncoder
from .liteuavdetr_postprocessor import LiteUAVDETRPostProcessor
from .matcher import HungarianMatcher
from .task import LiteUAVDETRDetectionModel

__all__ = (
    "LiteUAVDETR",
    "LiteUAVDETRCriterion",
    "LiteUAVDETRDetectionModel",
    "LiteUAVDETREncoder",
    "LiteUAVDETRPostProcessor",
    "LiteUAVDETRTransformer",
    "HungarianMatcher",
    "build_liteuavdetr",
    "build_liteuavdetr_postprocessor",
    "get_dfine_checkpoint_state",
    "infer_dfine_variant",
    "load_dfine_checkpoint",
)
