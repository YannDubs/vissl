# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

import torch
from classy_vision.dataset.transforms import build_transform, register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
import math
import einops
import numpy as np

@register_transform("ImgTensorMask")
class ImgTensorMask(ClassyTransform):
    """
    Apply masking for CNNs to a list of tensor images.
    """

    def __init__(self, prob_mask: List[float], patch_size: int=16, warmup_epochs=None):
        """
        Args:
            prob_mask (List(float)): Probability of masking patches for each image.
            patch_size (int): Size of patches.
            warmup_epochs (int): if not None warms up linearly the masking probability
                for that number of epochs
        """
        self._prob_mask = np.array(prob_mask)
        self.patch_size = patch_size
        self.base_patch = torch.ones(1, patch_size, patch_size).float()
        self.warmup_epochs = warmup_epochs
        self.epoch = 0

    def set_epoch(self, epoch):
        logging.info(f"set epoch to {epoch} in ImgTensorMask")
        self.epoch = epoch

    @property
    def prob_mask(self):
        if self.warmup_epochs is None or self.epoch > self.warmup_epochs:
            return self._prob_mask
        # linear warmup
        return self._prob_mask * self.epoch / self.warmup_epochs

    def __call__(self, image_list: List[torch.Tensor]):
        assert isinstance(image_list, list), "image_list must be a list"
        assert len(image_list) == len(self.prob_mask)

        return [self.mask_single_img(img,p)
                for img, p in zip(image_list, self.prob_mask)]

    def mask_single_img(self, img, p):
        alpha = torch.ones_like(img[0:1])
        new_img = torch.cat([img, alpha], dim=0)
        if p > 0 and not math.isclose(p, 0, abs_tol=1e-5):
            c, h, w = img.shape
            assert h % self.patch_size == 0 and w % self.patch_size == 0
            n_h_patch = h // self.patch_size
            n_w_patch = w // self.patch_size
            n_patches = n_h_patch * n_w_patch
            masked_patches = (torch.rand((n_patches, 1, 1)) > p).float() * self.base_patch
            mask = einops.rearrange(masked_patches, "(nh nw) p1 p2 -> (nh p1) (nw p2)",
                                    nh=n_h_patch, nw=n_w_patch)
            new_img *= mask
        return new_img

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgTensorMask":
        """
        Instantiates ImgPilMultiCropRandomApply from configuration.

        Args:
            config (Dict): arguments for the transform

        Returns:
            ImgPilMultiCropRandomApply instance.
        """
        patch_size = config.get("patch_size", 16)
        prob_mask = config.get("prob_mask", [])
        warmup_epochs = config.get("warmup_epochs", None)
        logging.info(f"ImgTensorMask | Using prob: {prob_mask} | warmup_epochs: {warmup_epochs}")
        return cls(prob_mask=prob_mask, patch_size=patch_size, warmup_epochs=warmup_epochs)
