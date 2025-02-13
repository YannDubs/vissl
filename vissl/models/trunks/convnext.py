# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Initial code copied from the official ConvNext repository:
https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
Then modified to support VISSL format.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from vissl.config import AttrDict
from vissl.models.model_helpers import DropPath, trunc_normal_
from vissl.models.trunks import register_model_trunk

class INPUT_CHANNEL(int, Enum):
    lab = 1
    bgr = 3
    rgb = 3
    rgba = 4

class Block(nn.Module):
    r"""
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv
        -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last)
        -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, norm_type="layernorm"):
        super().__init__()
        self.norm_type = norm_type

        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=(7, 7), padding=(3, 3), groups=dim
        )  # depthwise conv

        if self.norm_type == "layernorm":
            self.norm = LayerNorm(dim, eps=1e-6)
        elif self.norm_type == "batchnorm":
            self.norm = nn.BatchNorm2d(dim)
        else:
            raise ValueError(f"Invalid norm type: {self.norm_type}")

        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)

        if self.norm_type == "layernorm":
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
        elif self.norm_type == "batchnorm":
            # batchnorm expects (N, C, H, W)
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""
    LayerNorm that supports two data formats: channels_last (default) or
    channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds
    to inputs with shape (batch_size, height, width, channels) while
    channels_first corresponds to inputs with shape
    (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):

        # make sure fp32 to avoid nan
        x = x.float()
        weight = self.weight.float()
        bias = self.bias.float()

        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                weight,
                bias,
                self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = weight[:, None, None] * x + bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        depths: Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims: Feature dimension at each stage. Default: [96, 192, 384, 768]
        in_chans: Number of input image channels. Default: 3
        drop_path_rate: Stochastic depth rate. Default: 0.
        layer_scale_init_value: Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        depths: List[int],
        dims: List[int],
        in_chans: int = 3,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        norm_type="layernorm",
        is_skip_last_norm=False,
    ):
        super().__init__()
        self.norm_type = norm_type

        if self.norm_type == "layernorm":
            Norm = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        elif self.norm_type == "batchnorm":
            Norm = nn.BatchNorm2d
        else:
            raise ValueError(f"Invalid norm type: {self.norm_type}")

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers

        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            Norm(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                Norm(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        norm_type=norm_type
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # final norm layer
        if is_skip_last_norm:
            self.norm = nn.Identity()
        elif self.norm_type == "layernorm":
            self.norm = LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        elif self.norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(dims[-1])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        x = self.forward_features(x)
        return [x]


@register_model_trunk("convnext")
def ConvNeXtTrunk(model_config: AttrDict, model_name: str) -> nn.Module:
    trunk_config = model_config.TRUNK.CONVNEXT
    return ConvNeXt(
        depths=trunk_config.DEPTH,
        dims=trunk_config.DIMS,
        drop_path_rate=trunk_config.DROP_PATH_RATE,
        in_chans=INPUT_CHANNEL[model_config.INPUT_TYPE],
        norm_type=trunk_config.NORM_TYPE,
        is_skip_last_norm=trunk_config.IS_SKIP_LAST_NORM,
    )