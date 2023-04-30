from typing import List, Dict, Tuple, Optional
from einops import rearrange
from torch import Tensor
from tarvis.modelling.point_sampling_utils import point_sample
from scipy.ndimage import distance_transform_edt

import numpy as np
import torch
import torch.nn as nn


class PETQueryInitializer(nn.Module):
    def __init__(self, dim: int, n_scales: int, hidden_dim: int = -1):
        super().__init__()

        if hidden_dim == -1:
            hidden_dim = dim

        self.layers = nn.Sequential(
            nn.Linear(dim * n_scales, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, fmaps: List[Tensor], point_coords: Tensor, image_padding: Optional[Tuple[int, int]] = None,
                largest_scale: int = 4) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward method
        :param fmaps: list of multi-scale feature maps, each of shape [B, C, H, W]
        :param point_coords: Point-coordinates for each object as a tensor of shape [B, N, 2].
        Coordinates are expected to be (y, x) and normalized to the range [0, 1].
        :param image_padding: image padding as a tuple of (pad_right, pad_bottom)
        :param largest_scale: down-sampling scale for the largest feature map in `fmaps`.
        :return: tensor of shape [B, N, C]
        """
        if image_padding is not None:
            fmap_h, fmap_w = 0, 0
            for f in fmaps:
                fmap_h = max(f.shape[-2], fmap_h)
                fmap_w = max(f.shape[-1], fmap_w)

            padded_image_size = (fmap_h * largest_scale, fmap_w * largest_scale)
            point_coords = self.adapt_point_coords(point_coords, image_padding, padded_image_size)

        assert point_coords.ndim == 3
        point_coords = point_coords.flip([2])  # (y, x) to (x, y)
        assert torch.all(torch.logical_and(point_coords >= 0, point_coords <= 1.0)), \
            f"Point coords: {point_coords.cpu().numpy()}"

        sampled_features = [point_sample(f, point_coords, mode='nearest', align_corners=False) for f in fmaps]

        sampled_features_concat = rearrange(torch.cat(sampled_features, 1), "B C N -> B N C")
        inits = self.layers(sampled_features_concat)
        return inits, sampled_features

    @staticmethod
    def adapt_point_coords(point_coords: Tensor, image_padding: Tuple[int, int], padded_image_size: Tuple[int, int]):
        """
        Adapt point coordinates according to image padding.
        :param point_coords: tensor of shape [B, N, 2] with normalized point coordinates in (y, x) format.
        :param image_padding: image padding as a tuple of (pad_right, pad_bottom)
        :param padded_image_size: dimensions of the padded image as (height, width)
        :return: tensor of shape [B, N, 2]
        """
        height, width = padded_image_size
        pad_right, pad_bottom = image_padding

        unpadded_height = float(height - pad_bottom)
        unpadded_width = float(width - pad_right)
        assert unpadded_width > 0 and unpadded_height > 0

        unpadded_dims = torch.tensor([unpadded_height, unpadded_width]).to(point_coords)[None, None, :]
        pad_size = torch.as_tensor([pad_bottom, pad_right]).to(point_coords)[None, None, :]

        point_coords = (point_coords * unpadded_dims) / (unpadded_dims + pad_size)
        assert torch.all(torch.logical_and(point_coords >= 0.0, point_coords <= 1.0))
        return point_coords

    @classmethod
    def mask_to_point(cls, masks: List[Tensor], coord_offset: float = 0.0) -> List[Tensor]:
        """
        Calculates the center-most point for a set of object masks. Useful during training.
        :param masks: list of masks (one entry per batch sample), each of shape [N, H, W]
        :param coord_offset
        :return: list of point coordinates, each of shape [N, 2] in (y, x) format
        """
        dims = torch.tensor(masks[0].shape[-2:], dtype=torch.float32, device=masks[0].device)
        point_coords = []

        # start_time = current_time()
        for masks_b in masks:
            assert masks_b.ndim == 3, f"Expected masks entry to have 3D arrays, but got array of shape {masks_b.shape}"
            masks_b = masks_b.bool().cpu().numpy()
            point_coords.append(
                (torch.as_tensor([cls._mask_to_point(m) for m in masks_b], dtype=torch.float32, device=masks[0].device) + coord_offset) / dims
            )
        # print(f"Elapsed time: {current_time() - start_time}")
        # cls._viz_center_point(masks, point_coords)

        return point_coords

    @classmethod
    def _mask_to_point(cls, mask: np.ndarray):
        # compute distance field for the given mask
        distances = distance_transform_edt(mask, return_distances=True)
        max_dist = distances == np.max(distances)

        max_dist_pts_y, max_dist_pts_x = np.nonzero(max_dist)

        if len(max_dist_pts_y) > 1:
            # more than one point has max distance from boundary. In this case we choose the one closest to the
            # first moment of mask
            mask_pts_y, mask_pts_x = np.nonzero(mask)
            mean_y = np.mean(mask_pts_y.astype(np.float32))
            mean_x = np.mean(mask_pts_x.astype(np.float32))

            score_y = np.abs(max_dist_pts_y - mean_y)
            score_x = np.abs(max_dist_pts_x - mean_x)
            score_xy = score_x + score_y

            min_idx = np.argmin(score_xy)
            return max_dist_pts_y[min_idx].item(), max_dist_pts_x[min_idx].item()

        else:
            return max_dist_pts_y[0].item(), max_dist_pts_x[0].item()

    @classmethod
    def _viz_center_point(cls, masks, point_coords):
        # masks: List[B, [N, H, W]]
        # point_coords: List[B, [N, 2]]  (y, x)
        import cv2
        cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)

        for masks_b, coords_b in zip(masks, point_coords):
            height, width = masks_b.shape[-2:]
            coords_b[:, 0] *= height
            coords_b[:, 1] *= width

            for mask_i, coord_i in zip(masks_b, coords_b):
                img = mask_i[:, :, None].repeat(1, 1, 3).byte().cpu().numpy() * 255
                pt_y, pt_x = coord_i.round().int().tolist()
                img = cv2.circle(img, (pt_x, pt_y), radius=2, color=(100, 100, 100), thickness=-1)
                cv2.imshow('Mask', img)
                cv2.waitKey(0)

