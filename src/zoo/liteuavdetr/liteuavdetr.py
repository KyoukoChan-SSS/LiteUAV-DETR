"""
LiteUAVDETR model entry point, adapted from the D-FINE detector architecture.
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
Modified from D-FINE (https://github.com/Peterande/D-FINE)
---------------------------------------------------------------------------------
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register

__all__ = [
    "LiteUAVDETR",
]


@register()
class LiteUAVDETR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
