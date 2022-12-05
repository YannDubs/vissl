# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.utils.fsdp_utils import fsdp_auto_wrap_bn, fsdp_wrapper


@register_model_head("mlp")
class MLP(nn.Module):
    """
    This module can be used to attach combination of {Linear, BatchNorm, Relu, Dropout}
    layers and they are fully configurable from the config file. The module also supports
    stacking multiple MLPs.

    Examples:
        Linear
        Linear -> BN
        Linear -> ReLU
        Linear -> Dropout
        Linear -> BN -> ReLU -> Dropout
        Linear -> ReLU -> Dropout
        Linear -> ReLU -> Linear -> ReLU -> ...
        Linear -> Linear -> ...
        ...

    Accepts a 2D input tensor. Also accepts 4D input tensor of shape `N x C x 1 x 1`.
    """

    def __init__(
        self,
        model_config: AttrDict,
        dims: List[int],
        use_bn: bool = False,
        use_ln: bool = False,
        use_relu: bool = False,
        use_gelu: bool = False,
        use_dropout: bool = False,
        use_bias: bool = True,
        skip_last_layer_relu_bn: bool = True,
        is_JL_init : bool=False,
        is_residual: bool=False,
        is_cosine: bool=False
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            use_bn (bool): whether to attach BatchNorm after Linear layer
            use_relu (bool): whether to attach ReLU after (Linear (-> BN optional))
            use_dropout (bool): whether to attach Dropout after
                                (Linear (-> BN -> relu optional))
            use_bias (bool): whether the Linear layer should have bias or not
            dims (int): dimensions of the linear layer. Example [8192, 1000] which
                        attaches `nn.Linear(8192, 1000, bias=True)`
            skip_last_layer_relu_bn (bool): If the MLP has many layers, we check
                if after the last MLP layer, we should add BN / ReLU or not. By
                default, skip it. If user specifies to not skip, then BN will be
                added if use_bn=True, ReLU will be added if use_relu=True
        """
        super().__init__()
        self.is_residual = is_residual
        self.is_cosine = is_cosine
        assert not(use_gelu and use_relu), "Cannot use both ReLU and GELU"

        if self.is_residual:
            assert dims[0] == dims[-1]

        layers = []
        last_dim = dims[0]
        for i, dim in enumerate(dims[1:]):

            Model = CosineLayer if self.is_cosine else nn.Linear
            layers.append(Model(last_dim, dim, bias=use_bias))

            if i == len(dims) - 2 and skip_last_layer_relu_bn:
                break
            if use_bn:
                layers.append(
                    nn.BatchNorm1d(
                        dim,
                        eps=model_config.HEAD.BATCHNORM_EPS,
                        momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
                    )
                )
            if use_ln:
                layers.append( LayerNorm( dim)  )
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            if use_gelu:
                layers.append(nn.GELU())
            if use_dropout:
                layers.append(nn.Dropout())
            last_dim = dim
        self.clf = nn.Sequential(*layers)
        # we use the default normal or uniform initialization for the layers
        # and allow users to scale the initialization.
        self.scale_weights(model_config, is_JL_init)

    def scale_weights(self, model_config, is_JL_init):
        params_multiplier = model_config.HEAD.PARAMS_MULTIPLIER
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data *= params_multiplier
                if m.bias is not None:
                    m.bias.data *= params_multiplier

                if is_JL_init and (m.weight.shape[0] < m.weight.shape[1]):
                    # Initialization for low dimension projection => johnson lindenstrauss lemma.
                    torch.nn.init.normal_(m.weight, std=1 / math.sqrt(m.weight.shape[0]))

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 2D torch tensor or 4D tensor of shape `N x C x 1 x 1`
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "MLP input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        if batch.ndim > 2:
            assert all(
                d == 1 for d in batch.shape[2:]
            ), f"MLP expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
            batch = batch.reshape((batch.size(0), batch.size(1)))

        out = self.clf(batch)

        if self.is_residual:
            out = out + batch

        return out


class CosineLayer(nn.Module):
    """Cosine similarity between inputs and weights."""
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        assert not bias
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.linear = nn.utils.weight_norm(self.linear)
        self.linear.weight_g.data.fill_(1)  # unit norm
        self.linear.weight_g.requires_grad = False  # don't optimize norm
        self._is_forward = False

    def forward(self, X):
        if not self._is_forward:
            self.linear.weight_g.data.fill_(1) # in case vissl reinitalizes somewhere at some point
        self._is_forward = True

        unit_X = F.normalize(X, dim=-1, p=2)
        return self.linear(unit_X)

@register_model_head("identity")
class Identity(nn.Identity):
    ...


@register_model_head("batchnorm")
class Batchnorm(nn.Module):
    def __init__(
        self,
        model_config: AttrDict,
        dim: int,
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(
                        dim,
                        eps=model_config.HEAD.BATCHNORM_EPS,
                        momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
                    )

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 2D torch tensor or 4D tensor of shape `N x C x 1 x 1`
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "Batchnorm input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        if batch.ndim > 2:
            assert all(
                d == 1 for d in batch.shape[2:]
            ), f"Batchnorm expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
            batch = batch.reshape((batch.size(0), batch.size(1)))
        out = self.norm(batch)
        return out


@register_model_head("mlp_fsdp")
def MLP_FSDP(
    model_config: AttrDict,
    dims: List[int],
    use_bn: bool = False,
    use_relu: bool = False,
    use_dropout: bool = False,
    use_bias: bool = True,
    skip_last_layer_relu_bn: bool = True,
):
    mlp = MLP(
        model_config,
        dims,
        use_bn,
        use_relu,
        use_dropout,
        use_bias,
        skip_last_layer_relu_bn,
    )
    mlp = fsdp_auto_wrap_bn(mlp)
    return fsdp_wrapper(mlp, **model_config.FSDP_CONFIG)


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
