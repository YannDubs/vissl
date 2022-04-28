# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import logging
import pprint
import torch.nn.functional as F
import operator
from functools import reduce

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
            mode=self.loss_config.mode,
            z_dim=self.loss_config.z_dim,
        )

class DsslRegCriterion(DstlISSLCriterion):

    def __init__(self, beta_reg=0.1, mode=False, z_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.beta_reg = beta_reg
        self.mode = mode
        self.z_dim = z_dim

        if self.mode == "etf":
            self.regularizer = ETFRegularizer(self.z_dim)
        elif self.mode == "effdim":
            self.regularizer = EffdimRegularizer(self.z_dim)


    def forward(self, output):
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
                if self.mode == "huber":
                    inv_reg = inv_reg + F.smooth_l1_loss(Z, Z_aug, reduction="mean")
                elif self.mode == "etf":
                    inv_reg = inv_reg + self.regularizer(Z, Z_aug)
                elif self.mode == "effdim":
                    inv_reg = inv_reg + self.regularizer(Z, Z_aug)

        inv_reg = inv_reg / n_reg

        if self.num_iteration % 200 == 0 and self.dist_rank == 0:
            logging.info(f"{self.mode} Reg.: {inv_reg}")

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
            "mode": self.mode,
            "z_dim": self.z_dim
        }
        return pprint.pformat(repr_dict, indent=2)

class ETFRegularizer(torch.nn.Module):
    """Increases effective dimensionality by each dimension independent.

    Parameters
    ---------
    z_dim : list or int
        Shape of representation.
    """

    def __init__(
        self,
        z_dim,
        is_exact_etf=True,
    ) :
        super().__init__()
        self.is_exact_etf = is_exact_etf
        self.z_dim = z_dim
        self.bn = torch.nn.BatchNorm1d(self.z_dim, affine=False)

    def get_etf_rep(self, z):
        return F.normalize(self.bn(z), dim=-1, p=2)

    def forward(self, zx, za):
        z_dim = zx.shape[1]

        zx = self.get_etf_rep(zx)
        za = self.get_etf_rep(za)

        MtM = zx @ za.T
        if self.is_exact_etf:
            pos_loss = (MtM.diagonal() - 1).pow(2).mean()  # want it to be 1
            neg_loss = (1 / z_dim + MtM.masked_select(~eye_like(MtM).bool())).pow(2).mean()  # want it to be - 1 /dim
        else:
            pos_loss = - MtM.diagonal().mean()  # directly maximize
            neg_loss = MtM.masked_select(~eye_like(MtM).bool()).mean()  # directly minimize

        return pos_loss + neg_loss


class EffdimRegularizer(torch.nn.Module):
    """Increases effective dimensionality by each dimension independent.

    Parameters
    ---------
    z_dim : int
        Shape of representation.

    is_use_unit : bool, optional
        Whether to normalize before dot prod.
    """

    def __init__(
        self,
        z_dim,
        is_use_unit=True,
    ) -> None:
        super().__init__()
        self.corr_coef_bn = torch.nn.BatchNorm1d(z_dim, affine=False)
        self.is_use_unit = is_use_unit

    def forward(self, z_x, z_a) :

        batch_size, dim = z_x.shape

        z_x = self.corr_coef_bn(z_x)
        z_a = self.corr_coef_bn(z_a)

        if self.is_use_unit:
            z_x = F.normalize(z_x, dim=0, p=2)
            z_a = F.normalize(z_a, dim=0, p=2)

        corr_coeff = (z_x.T @ z_a) / batch_size

        pos_loss = (corr_coeff.diagonal() - 1).pow(2).mean()
        neg_loss = corr_coeff.masked_select(~eye_like(corr_coeff).bool()).view(dim, dim - 1).pow(2).mean()

        return pos_loss + neg_loss


def eye_like(x):
    """Return an identity like `x`."""
    return torch.eye(*x.size(), out=torch.empty_like(x))
