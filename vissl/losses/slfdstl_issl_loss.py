# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import logging
import pprint
import torch.nn.functional as F

import numpy as np
import torch
from torch.distributions import Categorical
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict
from vissl.utils.distributed_utils import gather_from_all


@register_loss("slfdstl_issl_loss")
class SlfdstlISSLLoss(ClassyLoss):
    """
    This is the contrastive loss which was proposed in ISSL <IRL> paper.
    See the paper for the details on the loss.

    Config params:
        n_Mx (int):                     number of maximal invariants.
        temperature_assign (float):     the temperature to be applied on the logits for assigning M(X).
        temperature_pred (float):       the temperature to be applied on the logits for predicting M(X)
        num_crops (int):                number of crops used
        crops_for_assign (List[int]):   what crops to use for assignment
        beta_pM_unif (float):           scaling to use for the entropy.
        ema_weight_marginal (float):    ema to use for the prior. If None no ema.
    """

    def __init__(self, loss_config: AttrDict, device: str = "gpu"):
        super(SlfdstlISSLLoss, self).__init__()

        self.loss_config = loss_config
        self.slfdstl_criterion = SlfdstlISSLCriterion(
            self.loss_config.n_Mx,
            self.loss_config.temperature_assign,
            self.loss_config.temperature_pred,
            self.loss_config.num_crops,
            self.loss_config.crops_for_assign,
            self.loss_config.beta_pM_unif,
            self.loss_config.ema_weight_marginal,
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates SlfdstlISSLLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            SlfdstlISSLLoss instance.
        """
        return cls(loss_config)

    def forward(self, output, target):
        loss = self.slfdstl_criterion(output)
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name(), "slfdstl_criterion": self.slfdstl_criterion}
        return pprint.pformat(repr_dict, indent=2)


class SlfdstlISSLCriterion(nn.Module):

    def __init__(self,
                 n_Mx : int = 50000,
                 temperature_assign : float = 0.05,
                 temperature_pred : float = 0.1,
                 num_crops : int = 2,
                 crops_for_assign : list[int] = [0,1],
                 beta_H_MlZ : float = 1.0,
                 beta_pM_unif: float = 1.7,
                 ema_weight_marginal : float = 0.7,
                ):
        super(SlfdstlISSLCriterion, self).__init__()
        assert beta_H_MlZ >= 1

        self.n_Mx = n_Mx
        self.temperature_assign = temperature_assign
        self.temperature_pred = temperature_pred
        self.num_crops = num_crops
        self.crops_for_assign = crops_for_assign
        self.beta_pM_unif = beta_pM_unif
        self.beta_H_MlZ = beta_H_MlZ
        self.ema_weight_marginal = ema_weight_marginal

        if self.ema_weight_marginal is not None:
            # initialize running means with uniform
            uniform = torch.ones(self.n_Mx) / self.n_Mx
            self.running_means = nn.ModuleList([RunningMean(uniform,
                                                            alpha_use=self.ema_weight_marginal)
                                                for _ in range(len(self.crops_for_assign))])

    def forward(self, output: List[torch.Tensor]):
        logits_assign, logits_predict = output
        logits_assign = logits_assign.float() / self.temperature_assign

        all_p_Mlz = F.softmax(logits_assign, dim=-1
                              ).chunk(len(self.crops_for_assign))

        all_log_p_Mlz = F.log_softmax(logits_assign, dim=-1
                                      ).chunk(len(self.crops_for_assign))

        all_log_q_Mlz = F.log_softmax(logits_predict.float() / self.temperature_pred, dim=-1
                                     ).chunk(self.num_crops)

        CE_pMlz_qMlza = 0
        H_M = 0
        CE_pMlz_pMlza = 0
        n_CE_pq = 0
        n_CE_pp = 0
        for i_p, p_Mlz in enumerate(all_p_Mlz):

            ##### Ensure maximality #####
            # current marginal estimate p(M). batch shape: [] ; event shape: []
            p_M = p_Mlz.mean(0, keepdim=True)
            p_M = self.gather_marginal(p_M)  # avg marginal across all gpus

            if self.ema_weight_marginal is not None:
                p_M = self.running_means[i_p](p_M)

            # D[\hat{p}(M) || Unif(\calM)]. shape: []
            # for unif prior same as maximizing entropy. Could be computed once per GPU, but fast so ok
            H_M = H_M + Categorical(probs=p_M).entropy()
            #############################

            ##### Ensure invariance and determinism of assignement #####
            if self.beta_H_MlZ > 1:
                for i_log_p, log_p_Mlza in enumerate(all_log_p_Mlz):
                    if i_p == i_log_p:
                        continue
                    CE_pMlz_pMlza = CE_pMlz_pMlza - (p_Mlz * log_p_Mlza).sum(-1).mean(0)
                    n_CE_pp += 1
            #########################

            for i_q, log_q_Mlza in enumerate(all_log_q_Mlz):
                if i_p == i_q:
                    # we skip cases where student and teacher operate on the same view
                    continue

                # KL = - H[M|Z] - E_{p(M|Z)}[log q(M|Z)]. As you want to have a deterministic
                # p(M|Z) you want to min H[M|Z]. So min KL + H[M|Z] = - E_{p(M|Z)}[log q(M|Z)]
                CE_pMlz_qMlza = CE_pMlz_qMlza - (p_Mlz * log_q_Mlza).sum(-1).mean(0)
                n_CE_pq += 1

        CE_pMlz_qMlza /= n_CE_pq
        H_M /= len(all_p_Mlz)
        CE_pMlz_pMlza /= n_CE_pp

        if self.ema_weight_marginal is not None:
            # try to balance the scaling in gradients due to running mean
            H_M = H_M / self.ema_weight_marginal

        delta_H_MlZ = self.beta_H_MlZ - 1  # the first beta is due to KL -> CE
        loss = CE_pMlz_qMlza + self.beta_pM_unif * H_M + delta_H_MlZ * CE_pMlz_pMlza

        return loss

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
        }
        return pprint.pformat(repr_dict, indent=2)

    @staticmethod
    def gather_marginal(marginal: torch.Tensor):
        """
        Do a gather over all marginals, so we can compute the entropy.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            marginal = gather_from_all(marginal)
        else:
            marginal = marginal
        return marginal.mean(0)

class RunningMean(nn.Module):
    """Keep track of an exponentially moving average"""
    def __init__(self, init: torch.tensor, alpha_use: float=0.5, alpha_store: float=0.1):
        super().__init__()

        assert 0.0 <= alpha_use <= 1.0
        assert 0.0 <= alpha_store <= 1.0
        self.alpha_use = alpha_use
        self.alpha_store = alpha_store
        self.register_buffer('running_mean', init.double())

    def forward(self, x):
        out = self.alpha_use * x + (1 - self.alpha_use) * self.running_mean.float()
        # don't keep all the computational graph to avoid memory++
        self.running_mean = (self.alpha_store * x.detach().double() + (1 - self.alpha_store) * self.running_mean).detach().double()
        return out