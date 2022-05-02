# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import List

import math
import einops
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
from vissl.config import AttrDict
from vissl.data.collators.collator_helper import MultiDimensionalTensor
from vissl.models.model_helpers import (
    Flatten,
    _get_norm,
    get_trunk_forward_outputs,
    get_tunk_forward_interpolated_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk


# For more depths, add the block config here
BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 24, 36, 3),
}


class SUPPORTED_DEPTHS(int, Enum):
    RN50 = 50
    RN101 = 101
    RN152 = 152
    RN200 = 200


class INPUT_CHANNEL(int, Enum):
    lab = 1
    bgr = 3
    rgb = 3


class SUPPORTED_L4_STRIDE(int, Enum):
    one = 1
    two = 2


@register_model_trunk("resnet")
class ResNeXt(nn.Module):
    """
    Wrapper for TorchVison ResNet Model to support different depth and
    width_multiplier. We provide flexibility with LAB input, stride in last
    ResNet block and type of norm (BatchNorm, LayerNorm)
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(ResNeXt, self).__init__()
        self.model_config = model_config
        logging.info(
            "ResNeXT trunk, supports activation checkpointing. {}".format(
                "Activated"
                if self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
                else "Deactivated"
            )
        )

        self.trunk_config = self.model_config.TRUNK.RESNETS
        self.depth = SUPPORTED_DEPTHS(self.trunk_config.DEPTH)
        self.width_multiplier = self.trunk_config.WIDTH_MULTIPLIER
        self._norm_layer = _get_norm(self.trunk_config)
        self.groups = self.trunk_config.GROUPS
        self.zero_init_residual = self.trunk_config.ZERO_INIT_RESIDUAL
        self.width_per_group = self.trunk_config.WIDTH_PER_GROUP
        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )

        (n1, n2, n3, n4) = BLOCK_CONFIG[self.depth]
        logging.info(
            f"Building model: ResNeXt"
            f"{self.depth}-{self.groups}x{self.width_per_group}d-"
            f"w{self.width_multiplier}-{self._norm_layer.__name__}"
        )

        model = models.resnet.ResNet(
            block=Bottleneck,
            layers=(n1, n2, n3, n4),
            zero_init_residual=self.zero_init_residual,
            groups=self.groups,
            width_per_group=self.width_per_group,
            norm_layer=self._norm_layer,
        )

        model.inplanes = 64 * self.width_multiplier
        dim_inner = 64 * self.width_multiplier
        # some tasks like Colorization https://arxiv.org/abs/1603.08511 take input
        # as L channel of an LAB image. In that case we change the input channel
        # and re-construct the conv1
        self.input_channels = INPUT_CHANNEL[self.model_config.INPUT_TYPE]

        model_conv1 = nn.Conv2d(
            self.input_channels,
            model.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model_bn1 = self._norm_layer(model.inplanes)
        model_relu1 = model.relu
        model_maxpool = model.maxpool
        model_avgpool = model.avgpool
        model_layer1 = model._make_layer(Bottleneck, dim_inner, n1)
        model_layer2 = model._make_layer(Bottleneck, dim_inner * 2, n2, stride=2)
        model_layer3 = model._make_layer(Bottleneck, dim_inner * 4, n3, stride=2)

        # For some models like Colorization https://arxiv.org/abs/1603.08511,
        # due to the higher spatial resolution desired for pixel wise task, we
        # support using a different stride. Currently, we know stride=1 and stride=2
        # behavior so support only those.
        safe_stride = SUPPORTED_L4_STRIDE(self.trunk_config.LAYER4_STRIDE)
        model_layer4 = model._make_layer(
            Bottleneck, dim_inner * 8, n4, stride=safe_stride
        )

        k_size = self.trunk_config.AVG_POOL_KERNEL_SIZE
        if k_size == -1:
            pass
        elif k_size is None:
            model_avgpool = torch.nn.Identity()
        else:
            model_avgpool = torch.nn.AvgPool2d([k_size, k_size], 1)

        Z_dim_old_tmp_new = self.trunk_config.Z_DIM_OLD_TMP_NEW
        if Z_dim_old_tmp_new is not None:

            old_nchan, tmp_nchan, new_nchan = Z_dim_old_tmp_new
            assert old_nchan % old_nchan == 0
            resizer = BottleneckExpand(old_nchan,
                                       tmp_nchan,
                                       expansion=new_nchan // old_nchan)

        else:
            resizer = torch.nn.Identity()

        # we mapped the layers of resnet model into feature blocks to facilitate
        # feature extraction at various layers of the model. The layers for which
        # to extract features is controlled by requested_feat_keys argument in the
        # forward() call.
        self._feature_blocks = nn.ModuleDict(
            [
                ("conv1", model_conv1),
                ("bn1", model_bn1),
                ("conv1_relu", model_relu1),
                ("maxpool", model_maxpool),
                ("layer1", model_layer1),
                ("layer2", model_layer2),
                ("layer3", model_layer3),
                ("layer4", model_layer4),
                ("resizer", resizer),
                ("avgpool", model_avgpool),
                ("flatten", Flatten(1)),
            ]
        )

        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
            "conv1": "conv1_relu",
            "res1": "maxpool",
            "res2": "layer1",
            "res3": "layer2",
            "res4": "layer3",
            "res5": "layer4",
            "resizer": "resizer",
            "res5avg": "avgpool",
            "flatten": "flatten",
        }

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:

        if isinstance(x, MultiDimensionalTensor):
            out = get_tunk_forward_interpolated_outputs(
                input_type=self.model_config.INPUT_TYPE,
                interpolate_out_feat_key_name="res5",
                remove_padding_before_feat_key_name="avgpool",
                feat=x,
                feature_blocks=self._feature_blocks,
                feature_mapping=self.feat_eval_mapping,
                use_checkpointing=self.use_checkpointing,
                checkpointing_splits=self.num_checkpointing_splits,
            )
        else:
            model_input = transform_model_input_data_type(
                x, self.model_config.INPUT_TYPE
            )
            out = get_trunk_forward_outputs(
                feat=model_input,
                out_feat_keys=out_feat_keys,
                feature_blocks=self._feature_blocks,
                feature_mapping=self.feat_eval_mapping,
                use_checkpointing=self.use_checkpointing,
                checkpointing_splits=self.num_checkpointing_splits,
            )
        return out

class BottleneckExpand(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            expansion=8,
            norm_layer=None,
            is_residual=True
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.expansion = expansion
        self.is_residual = is_residual
        out_channels = in_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, bias=False, padding=1)
        self.bn2 = norm_layer(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_residual:
            identity = einops.repeat(identity, 'b c h w -> b (tile c) h w', tile=self.expansion)
            out += identity

        out = self.relu(out)

        return out

    def reset_parameters(self) :
        weights_init(self)
        # using Johnson-Lindenstrauss lemma for initialization of the projection matrix
        torch.nn.init.normal_(self.conv1.weight,
                              std=1 / math.sqrt(self.conv1.weight.shape[0]))


def weights_init(module):
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    init_std_modules(module)  # in case you gave a standard module

    # loop over direct children (not grand children)
    for m in module.children():

        if init_std_modules(m):
            pass
        elif hasattr(m, "reset_parameters"):
            # if has a specific reset
            # Imp: don't go in grand children because you might have specific weights you don't want to reset
            m.reset_parameters()
        else:
            weights_init(m)  # go to grand children

def init_std_modules(module):
    """Initialize standard layers and return whether was initialized."""
    # all standard layers
    if isinstance(module, nn.modules.conv._ConvNd):
        variance_scaling_(module.weight)
        try:
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        try:
            nn.init.zeros_(module.bias)
        except AttributeError: # no bias
            pass

    elif isinstance(module, nn.modules.batchnorm._NormBase):
        if module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    else:
        return False

    return True

# taken from https://github.com/rwightman/pytorch-image-models/blob/d5ed58d623be27aada78035d2a19e2854f8b6437/timm/models/layers/weight_init.py
def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='truncated_normal'):
    """Initialization by scaling the variance."""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        nn.init.trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")