"""
LiteUAVDETR encoder with integrated StripFact Res fusion.
- Integrates the released StripFact Res path directly in LiteUAVDETREncoder.
- Removes experimental neck/MoE selectors while preserving checkpoint keys.
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
Modified from D-FINE (https://github.com/Peterande/D-FINE)
---------------------------------------------------------------------------------
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

__all__ = ["LiteUAVDETREncoder", "StripFactResNeck"]


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = get_activation(activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and optional SiLU activation."""

    def __init__(self, ch_in, ch_out, kernel_size=1, stride=1, groups=1, act=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            return self.act(self.conv_bn_fused(x))
        return self.act(self.norm(self.conv(x)))

    def convert_to_deploy(self):
        """Fold batch normalization into the convolution for inference."""
        if hasattr(self, "conv_bn_fused"):
            return

        conv, bn = self.conv, self.norm
        std = (bn.running_var + bn.eps).sqrt()
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        ).to(conv.weight.device, conv.weight.dtype)
        fused.weight.data.copy_(conv.weight * scale)
        fused.bias.data.copy_(bn.bias - bn.running_mean * bn.weight / std)
        self.conv_bn_fused = fused
        del self.conv
        del self.norm


class MBConvLite(nn.Module):
    """Compact MBConv refinement block with a residual connection."""

    def __init__(self, channels, expand=2):
        super().__init__()
        mid_channels = channels * expand
        self.pw1 = ConvBNAct(channels, mid_channels, kernel_size=1)
        self.dw = ConvBNAct(
            mid_channels,
            mid_channels,
            kernel_size=3,
            groups=mid_channels,
        )
        self.pw2 = ConvBNAct(mid_channels, channels, kernel_size=1, act=False)

    def forward(self, x):
        return x + self.pw2(self.dw(self.pw1(x)))


class StripFactFusion(nn.Module):
    """Fuse adjacent pyramid levels with factorized spatial messages."""

    def __init__(self, channels, k=7, diag_k=3, expand=2):
        super().__init__()
        if k % 2 != 1:
            raise ValueError(f"k must be odd; got {k}")
        if diag_k % 2 != 1:
            raise ValueError(f"diag_k must be odd; got {diag_k}")

        self.row_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, k),
            padding=(0, k // 2),
            groups=channels,
            bias=False,
        )
        self.col_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
            groups=channels,
            bias=False,
        )
        self.diag_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=diag_k,
            padding=diag_k // 2,
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(3 * channels)
        self.act = nn.SiLU(inplace=True)
        self.mixer = ConvBNAct(3 * channels, channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.full((channels, 1, 1), 0.1))
        self.refine = MBConvLite(channels, expand=expand)

    def forward(self, low, high_up):
        messages = torch.cat(
            [
                self.row_conv(high_up),
                self.col_conv(high_up),
                self.diag_conv(high_up),
            ],
            dim=1,
        )
        message = self.mixer(self.act(self.norm(messages)))
        return self.refine(low + self.alpha * message)

    def convert_to_deploy(self):
        """Fold the post-concatenation BN into the depthwise branches."""
        if isinstance(self.norm, nn.Identity):
            return

        bn = self.norm
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        bias = bn.bias - bn.running_mean * scale
        channels = self.row_conv.out_channels
        for branch_idx, conv in enumerate(
            (self.row_conv, self.col_conv, self.diag_conv)
        ):
            channel_slice = slice(
                branch_idx * channels,
                (branch_idx + 1) * channels,
            )
            conv.weight.data.mul_(scale[channel_slice].reshape(-1, 1, 1, 1))
            conv.bias = nn.Parameter(bias[channel_slice].detach().clone())
        self.norm = nn.Identity()


class StripFactNeck(nn.Module):
    """Top-down StripFact feature-pyramid fusion."""

    def __init__(
        self,
        hidden_dim,
        num_levels,
        width=64,
        k=7,
        diag_k=3,
        expand=2,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.reduce = nn.ModuleList(
            [ConvBNAct(hidden_dim, width, kernel_size=1) for _ in range(num_levels)]
        )
        self.fuse = nn.ModuleList(
            [
                StripFactFusion(width, k=k, diag_k=diag_k, expand=expand)
                for _ in range(num_levels - 1)
            ]
        )
        self.expand = nn.ModuleList(
            [
                ConvBNAct(width, hidden_dim, kernel_size=1, act=False)
                for _ in range(num_levels)
            ]
        )

    def forward(self, feats):
        reduced = [self.reduce[i](feats[i]) for i in range(self.num_levels)]
        top_down = list(reduced)
        for idx in range(self.num_levels - 2, -1, -1):
            high_up = F.interpolate(
                top_down[idx + 1],
                size=reduced[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            top_down[idx] = self.fuse[idx](reduced[idx], high_up)
        return [self.expand[i](top_down[i]) for i in range(self.num_levels)]


class StripFactResNeck(nn.Module):
    """Final StripFact neck with the weak-path residual highway."""

    def __init__(
        self,
        hidden_dim,
        num_levels,
        width=64,
        k=7,
        diag_k=3,
        expand=2,
        init_scale=0.05,
    ):
        super().__init__()
        self.base_neck = StripFactNeck(
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            width=width,
            k=k,
            diag_k=diag_k,
            expand=expand,
        )
        self.num_levels = num_levels
        self.weak_proj = nn.ModuleList(
            [
                ConvBNAct(hidden_dim, hidden_dim, kernel_size=1, act=False)
                for _ in range(num_levels - 1)
            ]
        )
        self.scales = nn.ParameterList(
            [
                nn.Parameter(torch.full((hidden_dim, 1, 1), init_scale))
                for _ in range(num_levels - 1)
            ]
        )

    def forward(self, feats):
        outputs = list(self.base_neck(feats))
        for idx in range(self.num_levels - 1):
            weak = self.weak_proj[idx](feats[idx + 1])
            weak_up = F.interpolate(
                weak,
                size=outputs[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            outputs[idx] = outputs[idx] + self.scales[idx] * weak_up
        return outputs


class StripFactResidualMerge(nn.Module):
    """Fixed learnable gated residual merge used by the released N/M models."""

    def __init__(self, num_levels):
        super().__init__()
        self.level_weights = nn.Parameter(torch.zeros(num_levels))

    def forward(self, base_feats, stripfact_feats):
        if len(base_feats) != len(stripfact_feats):
            raise ValueError("StripFact output level count does not match encoder input")
        return [
            base_feat + torch.sigmoid(self.level_weights[idx]) * stripfact_feat
            for idx, (base_feat, stripfact_feat) in enumerate(
                zip(base_feats, stripfact_feats)
            )
        ]


class LiteUAVDETREncoder(nn.Module):
    """Final LiteUAVDETR encoder with StripFact Res integrated directly."""

    __share__ = ["eval_spatial_size"]

    def __init__(
        self,
        in_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32),
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=(2,),
        num_encoder_layers=1,
        pe_temperature=10000,
        eval_spatial_size=None,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        self.num_feature_levels = len(self.in_channels)

        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    in_channel,
                                    hidden_dim,
                                    kernel_size=1,
                                    bias=False,
                                ),
                            ),
                            ("norm", nn.BatchNorm2d(hidden_dim)),
                        ]
                    )
                )
                for in_channel in self.in_channels
            ]
        )

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in self.use_encoder_idx]
        )

        # The child name keeps released checkpoint keys unchanged. It is not a
        # registry or architecture selector: this Sequential has one fixed path.
        self.neck = nn.Sequential(
            OrderedDict(
                [
                    (
                        "inner_model",
                        StripFactResNeck(
                            hidden_dim=hidden_dim,
                            num_levels=self.num_feature_levels,
                            width=64,
                            k=7,
                            diag_k=3,
                            expand=2,
                            init_scale=0.05,
                        ),
                    )
                ]
            )
        )
        self.neck_merger = StripFactResidualMerge(self.num_feature_levels)
        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(
        w,
        h,
        embed_dim=256,
        temperature=10000.0,
    ):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()],
            dim=1,
        )[None, :, :]

    def forward(self, feats):
        if len(feats) != len(self.in_channels):
            raise ValueError(
                f"Expected {len(self.in_channels)} backbone levels, got {len(feats)}"
            )

        proj_feats = [
            self.input_proj[idx](feat)
            for idx, feat in enumerate(feats)
        ]

        if self.num_encoder_layers > 0:
            for encoder_idx, feat_idx in enumerate(self.use_encoder_idx):
                height, width = proj_feats[feat_idx].shape[2:]
                src = proj_feats[feat_idx].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        width,
                        height,
                        self.hidden_dim,
                        self.pe_temperature,
                    ).to(device=src.device, dtype=src.dtype)
                else:
                    # LiteUAVDETR: D-FINE creates this fixed tensor outside the module buffer registry, so explicitly
                    # follow the feature dtype as well as its device for native FP16 inference.
                    pos_embed = getattr(self, f"pos_embed{feat_idx}").to(device=src.device, dtype=src.dtype)
                memory = self.encoder[encoder_idx](src, pos_embed=pos_embed)
                proj_feats[feat_idx] = (
                    memory.permute(0, 2, 1)
                    .reshape(-1, self.hidden_dim, height, width)
                    .contiguous()
                )

        stripfact_feats = self.neck(proj_feats)
        return self.neck_merger(proj_feats, stripfact_feats)
