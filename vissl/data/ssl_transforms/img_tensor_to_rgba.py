# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgTensor2RGBA")
class ImgTensor2RGBA(ClassyTransform):
    """
    Converts tensor to RGBA.
    """
    def __call__(self, image):
        alpha = torch.ones_like(image[0:1])
        new_img = torch.cat([image, alpha], dim=0)
        return new_img

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgTensor2RGBA":
        """
        Instantiates ImgTensor2RGBA from configuration.

        Args:
            config (Dict): arguments for the transform

        Returns:
            ImgTensor2RGBA instance.
        """
        return cls()
