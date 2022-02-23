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
                 beta_pM_unif : float = 1.7,
                 ema_weight_marginal : float = 0.7,
                ):
        super(SlfdstlISSLCriterion, self).__init__()

        self.n_Mx = n_Mx
        self.temperature_assign = temperature_assign
        self.temperature_pred = temperature_pred
        self.num_crops = num_crops
        self.crops_for_assign = crops_for_assign
        self.beta_pM_unif = beta_pM_unif
        self.ema_weight_marginal = ema_weight_marginal

        if self.ema_weight_marginal is not None:
            # initialize running means with uniform
            uniform = torch.ones(self.n_Mx) / self.n_Mx
            self.running_means = nn.ModuleList([RunningMean(uniform,
                                                            alpha_use=self.ema_weight_marginal)
                                                for _ in range(len(self.crops_for_assign))])

    def forward(self, output: List[torch.Tensor]):
        logits_assign, logits_predict = output

        all_p_Mlz = F.softmax(logits_assign.float() / self.temperature_assign, dim=-1
                              ).chunk(len(self.crops_for_assign))

        all_log_q_Mlz = F.log_softmax(logits_predict.float() / self.temperature_pred, dim=-1
                                     ).chunk(self.num_crops)

        total_cross_ent = 0
        total_marginal_ent = 0
        n_cross_ent = 0
        for i_p, p_Mlz in enumerate(all_p_Mlz):

            # current marginal estimate p(M). batch shape: [] ; event shape: []
            p_M = p_Mlz.mean(0, keepdim=True)
            p_M = self.gather_marginal(p_M)  # avg marginal across all gpus

            if self.ema_weight_marginal is not None:
                p_M = self.running_means[i_p](p_M)

            # D[\hat{p}(M) || Unif(\calM)]. shape: []
            # for unif prior same as maximizing entropy. Could be computed once per GPU, but fast so ok
            total_marginal_ent -= Categorical(probs=p_M).entropy()

            for i_q, log_qMlz in enumerate(all_log_q_Mlz):
                if i_p == i_q:
                    # we skip cases where student and teacher operate on the same view
                    continue

                # KL = - H[M|Z] - E_{p(M|Z)}[log q(M|Z)]. As you want to have a deterministic
                # p(M|Z) you want to min H[M|Z]. So min KL + H[M|Z] = - E_{p(M|Z)}[log q(M|Z)]
                cross_ent = torch.sum(-p_Mlz * log_qMlz, dim=-1)

                total_cross_ent += cross_ent.mean(0)
                n_cross_ent += 1

        total_cross_ent /= n_cross_ent
        total_marginal_ent /= len(all_p_Mlz)

        if self.ema_weight_marginal is not None:
            # try to balance the scaling in gradients due to running mean
            total_marginal_ent = total_marginal_ent / self.ema_weight_prior

        loss = total_cross_ent + self.beta_pM_unif * total_marginal_ent

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