from typing import List, Optional, Union, Tuple
from torch import Tensor

from tarvis.data.utils.image_cropping import compute_mask_containing_crop as compute_mask_containing_image_crop
from tarvis.data.utils.image_cropping import compute_mask_preserving_crop as compute_mask_preserving_image_crop

import numpy as np
import torch


def compute_mask_containing_crop(mask: Union[np.ndarray, Tensor],
                                 target_dims: Tuple[int, int],
                                 min_crop_mask_coverage: float = 0.25,
                                 num_tries: int = 10,
                                 containment_criterion: Optional[str] = "nonzero") -> Union[Tuple[int, int], None]:
    """
    Computes crop parameters for resizing the given video `mask` to `target_dims` such that AT LEAST SOME nonzero
    points in `mask` are contained in the crop. Note that the crop is intended to be the same across the temporal
    dimension
    :param mask: tensor of numpy array of shape [T, H, W] with non-zero values for active pixels
    :param target_dims: Desired size of crop as (height, width) tuple
    :param num_tries: number of attempts made before returning None
    :param min_crop_mask_coverage
    :param containment_criterion
    :return:
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(np.ascontiguousarray(mask))

    # collapse the mask along the time dimension and then just compute a mask containing image crop
    mask = torch.any(mask, 0)  # [H, W]
    return compute_mask_containing_image_crop(
        mask, target_dims, min_crop_mask_coverage=min_crop_mask_coverage, num_tries=num_tries,
        containment_criterion=containment_criterion
    )


def compute_mask_preserving_crop(mask: Union[np.ndarray, Tensor],
                                 target_dims: Tuple[int, int]) -> Union[Tuple[int, int], None]:
    """
    Computes crop parameters for resizing the given video `mask` to `target_dims` such that ALL nonzero points in `mask`
    remain preserved. Note that the crop is intended to be the same across the temporal dimension
    :param mask: tensor of numpy array of shape [T, H, W] with non-zero values for active pixels
    :param target_dims: Desired size of crop as (height, width) tuple
    :return: Tuple of [x, y] values representing the top-left coordinate of the crop or None if no such crop is possible
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(np.ascontiguousarray(mask))

    # collapse the mask along the time dimension and then just compute a mask preserving image crop
    mask = torch.any(mask, 0)  # [H, W]
    return compute_mask_preserving_image_crop(mask, target_dims)
