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
from classy_vision.generic.distributed_util import get_rank
from torch import nn
from vissl.config import AttrDict
from vissl.utils.distributed_utils import gather_from_all


@register_loss("dstl_issl_loss")
class DstlISSLLoss(ClassyLoss):
    """
    This is the contrastive loss which was proposed in ISSL <IRL> paper.
    See the paper for the details on the loss.

    Config params:
        n_Mx (int):                     number of maximal invariants.
        temperature_assign (float):     the temperature to be applied on the logits for assigning M(X).
        temperature_pred (float):       the temperature to be applied on the logits for predicting M(X)
        num_crops (int):                number of crops used
        crops_for_teacher (List[int]):  what crops to use for teacher (including invariance)
        crops_for_Mx (List[int]):      what crops to use as M(X) for student. By default same as teacher.
        beta_pM_unif (float):           scaling to use for the entropy.
    """

    def __init__(self, loss_config: AttrDict, device: str = "gpu"):
        super(DstlISSLLoss, self).__init__()

        self.loss_config = loss_config
        self.dstl_criterion = DstlISSLCriterion(
            n_Mx = self.loss_config.n_Mx,
            temperature_assign = self.loss_config.temperature_assign,
            temperature_pred = self.loss_config.temperature_pred,
            num_crops = self.loss_config.num_crops,
            crops_for_teacher = self.loss_config.crops_for_teacher,
            crops_for_Mx=self.loss_config.crops_for_Mx,
            beta_H_MlZ = self.loss_config.beta_H_MlZ,
            beta_pM_unif = self.loss_config.beta_pM_unif,
            warmup_teacher_iter = self.loss_config.warmup_teacher_iter,
            warmup_beta_unif_iter = self.loss_config.warmup_beta_unif_iter,
            warmup_beta_CE_iter = self.loss_config.warmup_beta_CE_iter
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates DstlISSLLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            DstlISSLLoss instance.
        """
        return cls(loss_config)

    def forward(self, output, target):
        loss = self.dstl_criterion(output)
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name(), "dstl_criterion": self.dstl_criterion}
        return pprint.pformat(repr_dict, indent=2)


class DstlISSLCriterion(nn.Module):

    def __init__(self,
                 n_Mx : int = 16384,
                 temperature_assign : float = 0.5,
                 temperature_pred : float = 1,
                 num_crops : int = 2,
                 crops_for_teacher : list[int] = [0,1],
                 crops_for_Mx : list[int] = None,
                 beta_H_MlZ : float = 0.5,
                 beta_pM_unif: float = 1.9,
                 warmup_beta_unif_iter: int = None,  # haven't tried but might be worth
                 warmup_teacher_iter: int= None,
                warmup_beta_CE_iter: int=None
                ):
        super(DstlISSLCriterion, self).__init__()


        self.n_Mx = n_Mx
        self.temperature_assign = temperature_assign
        self.temperature_pred = temperature_pred
        self.num_crops = num_crops
        self.crops_for_teacher = crops_for_teacher
        self.crops_for_Mx = crops_for_Mx or self.crops_for_teacher
        self.beta_pM_unif = beta_pM_unif
        self.beta_H_MlZ = beta_H_MlZ
        self.warmup_beta_unif_iter = warmup_beta_unif_iter
        self.warmup_teacher_iter = warmup_teacher_iter
        self.warmup_beta_CE_iter = warmup_beta_CE_iter
        self.dist_rank = get_rank()
        self.register_buffer("num_iteration", torch.zeros(1, dtype=int))

        if self.beta_pM_unif >= self.beta_H_MlZ + 1:
            logging.info(f"Theory suggests beta_pM_unif >= beta_H_MlZ + 1, which doesn't currently hold {beta_pM_unif} < {beta_H_MlZ + 1}.")

    def forward(self, output: List[torch.Tensor]):

        self.num_iteration += 1

        logits_assign, logits_predict = output
        logits_assign = logits_assign.float() / self.temperature_assign

        all_p_Mlz = F.softmax(logits_assign, dim=-1
                              ).chunk(len(self.crops_for_teacher))

        all_log_p_Mlz = F.log_softmax(logits_assign, dim=-1
                                      ).chunk(len(self.crops_for_teacher))

        all_log_q_Mlz = F.log_softmax(logits_predict.float() / self.temperature_pred, dim=-1
                                     ).chunk(self.num_crops)

        CE_pMlz_qMlza = 0
        H_M = 0
        CE_pMlz_pMlza = 0
        n_CE_pq = 0
        n_CE_pp = 0
        n_H = 0
        for i_p, p_Mlz in zip(self.crops_for_teacher, all_p_Mlz):
            if i_p not in self.crops_for_Mx:
                continue  # skip crops that are not used for expectation


            ##### Ensure maximality #####
            # current marginal estimate p(M). batch shape: [] ; event shape: []
            p_M = p_Mlz.mean(0, keepdim=True)
            p_M = self.gather_marginal(p_M)  # avg marginal across all gpus

            # D[\hat{p}(M) || Unif(\calM)]. shape: []
            # for unif prior same as maximizing entropy. Could be computed once per GPU, but fast so ok
            H_M = H_M + Categorical(probs=p_M).entropy()
            n_H += 1
            #############################

            ##### Ensure invariance and determinism of assignement #####
            for i_log_p, log_p_Mlza in zip(self.crops_for_teacher, all_log_p_Mlz):
                if i_p == i_log_p:
                    continue
                CE_pMlz_pMlza = CE_pMlz_pMlza - (p_Mlz * log_p_Mlza).sum(-1)
                n_CE_pp += 1
            #########################

            ##### Distillation #####
            for i_q, log_q_Mlza in enumerate(all_log_q_Mlz):
                # TODO this currently assumes that the first crops of student are the teacher ones

                if i_p == i_q:
                    # we skip cases where student and teacher operate on the same view
                    continue

                # KL = - H[M|Z] - E_{p(M|Z)}[log q(M|Z)]. As you want to have a deterministic
                # p(M|Z) you want to min H[M|Z]. So min KL + H[M|Z] = - E_{p(M|Z)}[log q(M|Z)]
                CE_pMlz_qMlza = CE_pMlz_qMlza - (p_Mlz * log_q_Mlza).sum(-1).mean(0)
                n_CE_pq += 1
            #########################

        CE_pMlz_qMlza /= n_CE_pq
        H_M /= n_H
        CE_pMlz_pMlza /= n_CE_pp

        fit_pM_Unif = - H_M  # want to max entropy

        if self.warmup_beta_unif_iter is not None and self.num_iteration < self.warmup_beta_unif_iter:
            start_beta = self.beta_H_MlZ + 1
            final_beta = self.beta_pM_unif
            warming_factor = (1 + self.num_iteration) / self.warmup_beta_unif_iter
            beta_pM_unif = start_beta + (final_beta - start_beta) * warming_factor
        else:
            beta_pM_unif = self.beta_pM_unif

        if self.warmup_beta_CE_iter is not None and self.num_iteration < self.warmup_beta_CE_iter:
            start_beta = self.beta_pM_unif
            final_beta = self.beta_H_MlZ
            warming_factor = (1 + self.num_iteration) / self.warmup_beta_CE_iter
            beta_H_MlZ = start_beta + (final_beta - start_beta) * warming_factor
        else:
            beta_H_MlZ = self.beta_H_MlZ

        if self.warmup_teacher_iter is not None and self.num_iteration < self.warmup_teacher_iter:
            warming_factor = (1 + self.num_iteration) / self.warmup_teacher_iter
            # warming up the distillation loss => focus on teacher first
            CE_pMlz_qMlza = CE_pMlz_qMlza * warming_factor
            # distillation also decreases determinism and invariance => rescales
            #CE_pMlz_pMlza = CE_pMlz_pMlza / warming_factor

        loss = CE_pMlz_qMlza + beta_pM_unif * fit_pM_Unif + beta_H_MlZ * CE_pMlz_pMlza

        if self.num_iteration % 200 == 0 and self.dist_rank == 0:
            logging.info(f"H[M]: {H_M.mean()}")
            logging.info(f"Distil: {CE_pMlz_qMlza.mean()}")
            logging.info(f"Inv + det: {CE_pMlz_pMlza.mean()}")
            H_Mlz = Categorical(probs=torch.cat(all_p_Mlz, dim=0).detach()).entropy().mean()
            logging.info(f"H[M|Z]: {H_Mlz}")

        return loss.mean()

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "n_Mx": self.n_Mx,
            "temperature_assign": self.temperature_assign,
            "temperature_pred": self.temperature_pred,
            "num_crops": self.num_crops,
            "crops_for_teacher": self.crops_for_teacher,
            "beta_pM_unif": self.beta_pM_unif,
            "beta_H_MlZ": self.beta_H_MlZ,
            "warmup_beta_unif_iter": self.warmup_beta_unif_iter,
            "warmup_teacher_iter": self.warmup_teacher_iter,
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