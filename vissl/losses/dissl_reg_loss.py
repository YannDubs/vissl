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
            is_rel_reg=self.loss_config.is_rel_reg,
        )

class DsslRegCriterion(DstlISSLCriterion):

    def __init__(self, beta_reg=0.1, is_rel_reg=False, **kwargs):
        super().__init__(**kwargs)
        self.beta_reg = beta_reg
        self.is_rel_reg = is_rel_reg

    def forward(self, output: list[torch.Tensor]):
        discriminative = super().forward(output[1:])

        batch_size = output[1].shape[0] // len(self.crops_for_assign)
        n_crops_for_reg = output[0].shape[0] // batch_size
        all_Z = output[0].chunk(n_crops_for_reg)

        n_reg = 0
        inv_reg = 0
        for i, Z in enumerate(all_Z):
            if i not in self.crops_for_assign:
                continue  # only compare to the assignment ones (decreases compute)

            for Z_aug in all_Z[i+1:]:
                n_reg += 1
                if self.is_rel_reg:
                    inv_reg = inv_reg + rel_distance(Z, Z_aug).mean()
                else:
                    inv_reg = inv_reg + F.smooth_l1_loss(Z, Z_aug, reduction="mean")

        inv_reg = inv_reg / n_reg

        if self.num_iteration % 200 == 0 and self.dist_rank == 0:
            sffx = "Rel." if self.is_rel_reg else "Huber"
            logging.info(f"{sffx} Reg.: {inv_reg}")

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
            "beta_reg" : self.beta_reg,
            "is_rel_reg": self.is_rel_reg
        }
        return pprint.pformat(repr_dict, indent=2)

def rel_distance(x1, x2, **kwargs):
    """
    Return the relative distance of positive examples compaired to negative.
    ~0 means that positives are essentially the same compared to negatives.
    ~1 means that positives and negatives are essentially indistinguishable.
    """
    batch_size = x1.shape[0]
    dist = torch.cdist(x1, x2, **kwargs)
    dist_no_diag = dist * (1 - torch.eye(*dist.size(), out=torch.empty_like(dist)))
    dist_neg_row = dist_no_diag.sum(0) / (batch_size - 1)
    dist_neg_col = dist_no_diag.sum(1) / (batch_size - 1)
    dist_neg = (dist_neg_row + dist_neg_col) / 2
    dist_pos = dist.diag()
    dist_rel = dist_pos / (dist_neg + 1e-5)
    return dist_rel