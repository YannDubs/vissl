# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import build_transform, register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("ImgPilMultiCropRandomApply")
class ImgPilMultiCropRandomApply(ClassyTransform):
    """
    Apply a list of transforms on multi-crop input. The transforms
    are Randomly applied to each crop using the specified probability.
    This is used in BYOL https://arxiv.org/pdf/2006.07733.pdf

    Multi-crops are several crops of a given image. This is most commonly
    used in contrastive learning. For example SimCLR, SwAV approaches use
    multi-crop input.
    """

    def __init__(self, transforms: List[Dict[str, Any]], prob: List[float], warmup_epochs=None):
        """
        Args:
            transforms ( List(tranforms) ): List of transforms that should be applied
                                           to each crop.
            prob (List(float)): Probability of RandomApply for the transforms
                                composition on each crop.
                                example: for 2 crop in BYOL, for solarization:
                                         prob = [0.0, 0.2]
            warmup_epochs (int): if not None warms up linearly the probability of applying
                the transform.
        """

        self.prob = prob
        self.transforms = None
        self._build_transform(transforms)
        self.warmup_epochs = warmup_epochs
        self.epoch = 0

    def _build_transform(self, transforms: List[Dict[str, Any]]):
        out_transforms = []
        for transform_config in transforms:
            out_transforms.append(build_transform(transform_config))
        out_transform = pth_transforms.Compose(out_transforms)

        self.transforms = []
        for idx in range(len(self.prob)):
            self.transforms.append(
                pth_transforms.RandomApply([out_transform], p=self.prob[idx])
            )

    def set_epoch(self, epoch):
        logging.info(f"set epoch to {epoch} in ImgPilMultiCropRandomApply")
        self.epoch = epoch

    def get_current_prob(self, idx):
        if self.warmup_epochs is None or self.epoch > self.warmup_epochs:
            return self.prob[idx]
        return self.prob[idx] * self.epoch / self.warmup_epochs

    def __call__(self, image_list: List[Image.Image]):
        assert isinstance(image_list, list), "image_list must be a list"
        assert len(image_list) == len(self.prob)
        assert len(image_list) == len(self.transforms)

        output = []
        for idx in range(len(image_list)):
            self.transforms[idx].p = self.get_current_prob(idx)
            output.append(self.transforms[idx](image_list[idx]))
        return output

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilMultiCropRandomApply":
        """
        Instantiates ImgPilMultiCropRandomApply from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilMultiCropRandomApply instance.
        """
        transforms = config.get("transforms", [])
        prob = config.get("prob", [])
        warmup_epochs = config.get("warmup_epochs", None)
        logging.info(f"ImgPilMultiCropRandomApply | Using prob: {prob} | warmup_epochs: {warmup_epochs}")
        return cls(transforms=transforms, prob=prob, warmup_epochs=warmup_epochs)
