from typing import List, Dict, Union
from torch import Tensor

import math
import torch
import torch.nn.functional as F


def split_by_query_group(x: Tensor, dim: int, query_group_names: List[str], query_group_counts: List[int]) \
        -> Dict[str, Tensor]:
    """
    Splits the given tensor 'x' along dimension 'dim' based on the query group
    :param x: input tensor
    :param dim: query dimension
    :param query_group_names: names of the query group
    :param query_group_counts:
    :return: dictionary with tensor split based on query group and corresponding name label
    """
    # validity checks
    assert len(query_group_names) == len(query_group_counts)
    total_queries = sum(query_group_counts)
    assert x.size(dim) == total_queries

    return {
        group_name: x_split for group_name, x_split in zip(query_group_names, x.split(query_group_counts, dim))
    }


def compute_padded_dims(height, width, multiple_of):
    padded_width = (int(math.ceil(width / float(multiple_of))) * multiple_of)
    padded_height = (int(math.ceil(height / float(multiple_of))) * multiple_of)
    return padded_height, padded_width


def compute_padding(height, width, pad_multiple_of):
    padded_width = (int(math.ceil(width / pad_multiple_of)) * pad_multiple_of)
    padded_height = (int(math.ceil(height / pad_multiple_of)) * pad_multiple_of)
    return padded_height - height, padded_width - width


@torch.no_grad()
def pad_image_tensor(tensor_list: List[Tensor], pad_value: int, stack: bool, multiple_of: int = 32):
    max_height = max([x.shape[-2] for x in tensor_list])
    max_width = max([x.shape[-1] for x in tensor_list])
    padded_height, padded_width = compute_padded_dims(max_height, max_width, multiple_of)
    padded_images = []

    for x in tensor_list:
        pad_right = padded_width - x.shape[-1]
        pad_bottom = padded_height - x.shape[-2]

        # handle variable number of dimensions in tensor. Allow any tensor with at least 2 dims.
        assert x.ndim >= 2, f"Image tensor must have at least 2 dimensions, but got tensor of shape {x.shape}"
        non_spatial_shape = x.shape[:-2]

        if x.ndim < 4:
            num_padded_dims = 4 - x.ndim
            view_shape = [1 for _ in range(num_padded_dims)] + list(x.shape)
            x = x.view(*view_shape)

        elif x.ndim > 4:
            num_compressed_dims = x.ndim - 4
            x = x.flatten(0, num_compressed_dims)

        assert x.ndim == 4
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode='constant', value=pad_value)
        padded_images.append(x.reshape(list(non_spatial_shape) + list(x.shape[-2:])))

    if stack:
        return torch.stack(padded_images, 0)
    else:
        return padded_images

