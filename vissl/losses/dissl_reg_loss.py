# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import logging
import pprint
import torch.nn.functional as F

import torch
from classy_vision.losses import ClassyLoss, register_loss
from vissl.config import AttrDict
from .dstl_issl_loss import DstlISSLLoss, DstlISSLCriterion


@register_loss("dissl_reg_loss")
class DisslRegLoss(DstlISSLLoss):
    def __init__(self, loss_config: AttrDict, device: str = "gpu"):
        super().__init__()

        self.loss_config = loss_config
        self.dstl_criterion = DsslRegCriterion(
            n_Mx = self.loss_config.n_Mx,
            temperature_assign = self.loss_config.temperature_assign,
            temperature_pred = self.loss_config.temperature_pred,
            num_crops = self.loss_config.num_crops,
            crops_for_assign = self.loss_config.crops_for_assign,
            beta_H_MlZ = self.loss_config.beta_H_MlZ,
            beta_pM_unif = self.loss_config.beta_pM_unif,
            ema_weight_marginal = self.loss_config.ema_weight_marginal,
            beta_reg=self.loss_config.beta_reg,
        )

class DsslRegCriterion(DstlISSLCriterion):

    def __init__(self, beta_reg, **kwargs):
        super().__init__(**kwargs)
        self.beta_reg = beta_reg

    def forward(self, output: List[torch.Tensor]):
        breakpoint()

        discriminative = super().forward(output[1:])

        Z = output[0]
        inv_reg = discriminative

        return discriminative + self.beta_reg * inv_reg.mean()

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "n_Mx": self.n_Mx,
            "temperature_assign": self.temperature_assign,
            "temperature_pred": self.temperature_pred,
            "num_crops": self.num_crops,
            "crops_for_assign": self.crops_for_assign,
            "beta_pM_unif": self.beta_pM_unif,
            "beta_H_MlZ": self.beta_H_MlZ,
            "ema_weight_marginal": self.ema_weight_marginal,
            "beta_reg" : self.beta_reg
        }
        return pprint.pformat(repr_dict, indent=2)