from einops import repeat
from torch import Tensor
from typing import List, Optional, Tuple
from tarvis.modelling.point_sampling_utils import point_sample_3d

import math
import itertools
import torch


def sample_features_from_masks(fmap: Tensor, pos_embeddings: Tensor, masks: List[Tensor],
                               max_allowed_num_points: Optional[int] = -1,
                               num_bg_masks: Optional[int] = 0) \
        -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
    """
    Sample features from `fmap` based on a list of provided `masks`.
    :param fmap: tensor of shape [B, C, T, H, W]
    :param masks: list of masks (one per batch sample), each of shape [N, T, H', W'].
    :param pos_embeddings: set of positional embeddings of identical shape to fmap
    :param max_allowed_num_points: optional parameter which controls the maximum number of active points per mask
    :param num_bg_masks: number of bg masks in `masks`. These are allowed to have no active pixels
    :return: list of B tensors, each of shape [M_i, C] (M_i = number of non-zero mask points per object). If
    pos_embeddings is given, another such list for those.
    """
    assert fmap.shape == pos_embeddings.shape, f"Shape mismatch: {fmap.shape}, {pos_embeddings.shape}"
    assert all([m.ndim == 4 for m in masks])
    fmap_with_pos = torch.cat((fmap, pos_embeddings), 1)  # [B, C*2, T, H, W]

    num_channels = fmap.size(1)
    clip_len, height, width = masks[0].shape[-3:]
    coord_normalizer = torch.tensor([width, height, clip_len]).to(fmap_with_pos)[None, :]  # [1, 3]

    sampled_features = []
    sampled_pos_embeddings = []
    max_num_points = -1

    for b in range(fmap_with_pos.size(0)):
        point_coords = []
        num_points = []

        with torch.no_grad():
            for i in range(masks[b].size(0)):
                object_point_coords = masks[b][i].nonzero(as_tuple=False)  # [P, 3]
                if 0 < max_allowed_num_points < object_point_coords.size(0):
                    keep_idxes = torch.randperm(object_point_coords.size(0))[:max_allowed_num_points]
                    object_point_coords = object_point_coords[keep_idxes]

                elif object_point_coords.size(0) == 0:
                    assert i < num_bg_masks, f"Zero mask found at index {i}, num_bg_masks = {num_bg_masks}"

                    # replace with dummy out-of-bounds point. This will be assigned an all-zeros value by `point_sample`
                    object_point_coords = object_point_coords.new_full((1, 3), -1)

                point_coords.append(object_point_coords)
                num_points.append(object_point_coords.size(0))
                # assert num_points[-1] > 0
                max_num_points = max(max_num_points, num_points[-1])

            point_coords = torch.cat(point_coords, 0).float().flip([1]) + 0.5  # (t, y, x) -> (x, y, t)
            point_coords = point_coords / coord_normalizer

            if num_bg_masks == 0:
                assert torch.all(point_coords >= 0.0)
            assert torch.all(point_coords <= 1.0)

            point_coords = point_coords.unsqueeze(0)  # [1, P, 3]

        point_features_pos = point_sample_3d(fmap_with_pos[b].unsqueeze(0), point_coords, mode='nearest',
                                             align_corners=False)  # [1, C*2, P]
        point_features_pos = point_features_pos.squeeze(0).transpose(0, 1)  # [P, C*2]

        point_features, point_pos_embeddings = point_features_pos.split((num_channels, num_channels), 1)  # [P, C], [P, C]

        # split based on points per object
        sampled_features.append(point_features.split(num_points, 0))
        sampled_pos_embeddings.append(point_pos_embeddings.detach().split(num_points, 0))

    return sampled_features, sampled_pos_embeddings


def _pad_features(x: List[Tensor], pad_length: int) -> Tuple[Tensor, Tensor]:
    # x: List[n, [N_i, C]]
    padded_x = []
    padding_mask = []

    for x_i in x:
        assert x_i.size(0) <= pad_length, f"Pad length ({pad_length}) must be greater than or equal to feature vector " \
            f"size ({x_i.shape})"

        n_pad = pad_length - x_i.size(0)
        padded_x.append(torch.cat([
            x_i, x_i.new_zeros(n_pad, x_i.size(1))
        ]))
        padding_mask.append(torch.cat([
            torch.zeros(x_i.size(0), dtype=torch.bool), torch.ones(n_pad, dtype=torch.bool)
        ]))

    padded_x = torch.stack(padded_x)
    padding_mask = torch.stack(padding_mask).to(device=padded_x.device)

    return padded_x, padding_mask


def pad_sampled_features(features: List[Tensor], pos_embeddings: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    max_num_feats = max([x.size(0) for x in features])
    features, padding_mask = _pad_features(features, max_num_feats)
    pos_embeddings, _ = _pad_features(pos_embeddings, max_num_feats)
    return features, pos_embeddings, padding_mask


def pad_batched_sampled_features(features: List[List[Tensor]], pos_embeddings: List[List[Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
    max_num_feats = max(list(itertools.chain(*[[x.size(0) for x in features_i] for features_i in features])))
    padded_features = []
    padded_pos_embeddings = []
    padding_mask = []

    for features_i, pos_embeddings_i in zip(features, pos_embeddings):
        features_i, padding_mask_i = _pad_features(features_i, max_num_feats)
        pos_embeddings_i, _ = _pad_features(pos_embeddings_i, max_num_feats)

        padded_features.append(features_i)
        padded_pos_embeddings.append(pos_embeddings_i)
        padding_mask.append(padding_mask_i)

    padded_features = torch.stack(padded_features)
    padded_pos_embeddings = torch.stack(padded_pos_embeddings).detach()
    padding_mask = torch.stack(padding_mask).detach()

    return padded_features, padded_pos_embeddings, padding_mask


def average_batched_sampled_features(features: List[List[Tensor]]) -> Tensor:
    averaged_feats = []
    for features_i in features:
        averaged_feats.append(torch.stack([x.mean(0) for x in features_i]))
    return torch.stack(averaged_feats)


@torch.no_grad()
def generate_background_masks(object_masks: List[Tensor], grid_dims: Tuple[int, int]) -> List[Tensor]:
    """
    Generate HODOR-style background masks by dividing the video into grid cells.
    :param object_masks: list of masks (one per batch sample), each of shape [N, T, H, W].
    :param grid_dims: size of the grid in terms of (height, width)
    :return: list of masks, each of shape [G, T, H, W]  (G = product(grid_dims))
    """
    fg_masks = torch.stack([torch.any(m, 0, keepdim=True) for m in object_masks]).bool()  # [B, 1, T, H, W]
    batch_sz, _, clip_len, img_h, img_w = fg_masks.shape
    grid_h, grid_w = grid_dims

    coord_y = torch.linspace(-0.499, grid_h-1 + 0.499, img_h, device=fg_masks.device).round().long()
    coord_x = torch.linspace(-0.499, grid_w-1 + 0.499, img_w, device=fg_masks.device).round().long()

    coords = torch.stack(torch.meshgrid([coord_y, coord_x], indexing='ij'))  # [2(y,x), H, W]
    bg_masks = (coords[0] * grid_w) + coords[1]  # [H, W]
    assert bg_masks.unique().tolist() == list(range(0, grid_h * grid_w))  # sanity check

    bg_masks = torch.stack([bg_masks == i for i in range(0, grid_h * grid_w)])  # [G, H, W]

    bg_masks = repeat(bg_masks, "G H W -> B G T H W", B=batch_sz, T=clip_len)  # [B, G, T, H, W]
    bg_masks = torch.where(fg_masks, torch.zeros_like(bg_masks), bg_masks)

    return bg_masks.unbind(0)


@torch.no_grad()
def divide_object_masks(object_masks: List[Tensor], num_divisions: int) -> List[Tensor]:
    """
    Useful for the case where we use multiple queries per object.
    :param object_masks: list of object masks (per batch sample), each of shape [N, T, H, W] (N = number of objects)
    :param num_divisions: number of segments to divide the mask into
    :return:
    """
    divided_object_masks = []
    assert object_masks[0].dtype in (torch.uint8, torch.bool), f"Dtype = {object_masks[0].dtype}"
    zeros_mask = object_masks[0].new_zeros([num_divisions] + list(object_masks[0].shape[1:]))  # [M, T, H, W]

    for masks_b in object_masks:
        mask_coords = masks_b.nonzero(as_tuple=False)  # [P, 4]
        vals, num_points_per_object = mask_coords[:, 0].unique(sorted=True, return_counts=True)
        # try:
        assert vals.tolist() == list(range(masks_b.size(0))), f"{vals.tolist()}, {masks_b.shape}"
        # except AssertionError as err:
        #     import pdb
        #     pdb.set_trace()

        mask_coords = mask_coords.split(num_points_per_object.tolist(), 0)
        divided_masks_b = []

        for n in range(len(mask_coords)):
            if mask_coords[n].size(0) < num_divisions:
                # if any mask contains fewer than `num_division` points, just repeat the same points
                n_repeats = int(math.ceil(float(num_divisions) / mask_coords[n].size(0)))
                mask_coords_n = mask_coords[n].repeat(n_repeats, 1)
            else:
                mask_coords_n = mask_coords[n]

            split_mask_coords = mask_coords_n[:, 1:].tensor_split(num_divisions, dim=0)

            split_mask_coords = torch.cat([
                torch.cat((torch.full_like(coords[:, :1], i), coords), 1)
                for i, coords in enumerate(split_mask_coords)
            ])

            split_mask = zeros_mask.clone()
            split_mask[tuple(split_mask_coords.unbind(1))] = 1
            assert torch.all(torch.any(split_mask.bool().flatten(1), 1))

            divided_masks_b.append(split_mask)

        divided_masks_b = torch.cat(divided_masks_b)  # [N*num_divisions, T, H, W]
        divided_object_masks.append(divided_masks_b)

    return divided_object_masks
