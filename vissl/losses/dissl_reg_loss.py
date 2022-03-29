# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import logging
import pprint
import torch.nn.functional as F

import torch
from classy_vision.losses import ClassyLoss, register_loss
from .dstl_issl_loss import DstlISSLLoss, DstlISSLCriterion

@register_loss("dissl_reg_loss")
class DisslRegLoss(DstlISSLLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.distance = torch.nn.SmoothL1Loss(reduction="sum")

    def forward(self, output: list[torch.Tensor]):
        discriminative = super().forward(output[1:])

        batch_size = output[1].shape[0] // len(self.crops_for_assign)
        n_crops_for_reg = output[0].shape[0] // batch_size
        all_Z = output[0].chunk(n_crops_for_reg)

        n_reg = 0
        inv_reg = 0
        for i, Z in enumerate(all_Z):
            for Z_aug in all_Z[i+1:]:
                n_reg += 1
                # sum over dimension but mean over batch
                inv_reg = inv_reg + self.distance(Z, Z_aug) / batch_size
        inv_reg = inv_reg / n_reg
        return discriminative + self.beta_reg * inv_reg

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