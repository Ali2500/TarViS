from torch import Tensor
from typing import List, Tuple, Union, Optional, Dict

import math
import torch
import numpy as np


def scale_and_normalize_images(images, means, scales, invert_channels, normalize_to_unit_scale):
    """
    Scales and normalizes images
    :param images: tensor(T, C, H, W)
    :param means: list(float)
    :param scales: list(float)
    :param invert_channels: bool
    :param normalize_to_unit_scale: bool
    :return: tensor(T, C, H, W)
    """
    means = torch.as_tensor(means, dtype=torch.float32)[None, :, None, None]  # [1, 3, 1, 1]
    scales = torch.as_tensor(scales, dtype=torch.float32)[None, :, None, None]  # [1. 3. 1. 1]
    if normalize_to_unit_scale:
        images = images / 255.

    images = (images - means) / scales
    if invert_channels:
        return images.flip(dims=[1])
    else:
        return images


def condense_mask(mask: Union[Tensor, np.ndarray], dtype: Optional[Union[np.dtype, torch.dtype]] = None):
    """
    Condense a one-hot mask into a mask array where pixel value denotes instance ID
    :param mask: Numpy or torch array of shape [N, H, W]
    :param dtype: dtype of output mask
    :return: Numpy or torch array of shape [H, W]
    """
    height, width = mask.shape[1:]
    if torch.is_tensor(mask):
        dtype = torch.long if dtype is None else dtype
        condensed_mask = torch.zeros(height, width, dtype=dtype, device=mask.device)
        where_fn = torch.where
    else:
        dtype = np.int32 if dtype is None else dtype
        condensed_mask = np.zeros((height, width), dtype)
        where_fn = np.where

    for iid, mask_per_channel in enumerate(mask, 1):
        condensed_mask = where_fn(mask_per_channel, iid, condensed_mask)

    return condensed_mask


def expand_mask(mask: Union[Tensor, np.ndarray], instance_ids: Optional[List[int]] = None):
    """
    Expand a condensed mask into a one-hot mask array
    :param mask: Numpy or torch array of shape [H, W] where pixel values denote instace/class ID
    :param instance_ids: Optional list of instance IDs. If not provided, all values in the given mask will be used.
    :return: Numpy or torch array of shape [N, H, W]
    """
    if instance_ids is None:
        if torch.is_tensor(mask):
            instance_ids = mask.unique()
        else:
            instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0].tolist()

    expanded_mask = [mask == iid for iid in instance_ids]

    if torch.is_tensor(mask):
        expanded_mask = torch.stack(expanded_mask, 0).bool()
    else:
        expanded_mask = np.stack(expanded_mask).astype(bool)

    return expanded_mask


def map_array_values(t: Union[Tensor, np.ndarray], mapping: Dict[int, int]) -> Tensor:
    """
    Convert a set of instance masks to a set of semantic class masks
    :param t: Tensor of arbitrary shape
    :param mapping:
    :return: Tensor of identical shape to input with mapped values
    """
    if torch.is_tensor(t):
        mapped_t = torch.zeros_like(t)
        where_fn = torch.where
    else:
        assert isinstance(t, np.ndarray)
        mapped_t = np.zeros_like(t)
        where_fn = np.where

    for val, mapped_val in mapping.items():
        mapped_t = where_fn(t == val, mapped_val, mapped_t)

    return mapped_t


def get_dataset_sample_padding_indices(dataset_length, num_desired_samples):
    whole_reps = int(math.floor(float(num_desired_samples) / float(dataset_length)))
    if whole_reps > 0:
        sample_idxes = list(range(dataset_length)) * whole_reps
    else:
        sample_idxes = []

    n_padded_samples = num_desired_samples % dataset_length
    assert n_padded_samples < dataset_length  # sanity check

    # pad with the same random samples across all parallel processes
    padding_idxes = torch.linspace(0, dataset_length-1, n_padded_samples).round().long()
    sample_idxes.extend(padding_idxes.tolist())

    return sample_idxes
