from einops import rearrange
from torch import Tensor
from typing import Optional, Union, List

from tarvis.data.common import condense_mask
from tarvis.data.utils.image_cropping import compute_mask_containing_crop as compute_mask_containing_image_crop
from tarvis.data.utils.video_cropping import compute_mask_containing_crop as compute_mask_containing_video_crop

import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np


class ImagesResizer:
    def __init__(self, **kwargs):
        mode = kwargs.pop("MODE")
        assert mode in ("min_dim", "pixel_area", "crop", "none")

        if mode in ("pixel_area", "crop"):
            self.pixel_area = kwargs.pop("TARGET_PIXEL_AREA")

        elif mode == "min_dim":
            self.min_dim = kwargs.pop("MIN_DIMS")
            self.max_dim = kwargs.pop("MAX_DIM")

        elif mode != "none":
            raise ValueError("Should not be here.")

        self.mode = mode

    def _resize_masks(self, masks, new_height, new_width):
        dtype = masks.dtype
        if torch.is_tensor(masks):
            masks = F.interpolate(masks.float(), (new_height, new_width), mode='bilinear', align_corners=False)
            masks = (masks > 0.5).astype(dtype)
        else:
            assert isinstance(masks, np.ndarray), f"Unexpected mask type: {type(masks)}"
            B, N = masks.shape[:2]
            masks = np.reshape(masks, (-1, *masks.shape[2:]))
            masks = np.stack([
                (cv2.resize(m.astype(np.float32), (new_width, new_height),
                            interpolation=cv2.INTER_LINEAR) > 0.5).astype(dtype)
                for m in masks
            ])
            masks = np.reshape(masks, (B, N, *masks.shape[1:]))

        return masks

    def __call__(self, images: Union[Tensor, np.ndarray], masks: Optional[Union[Tensor, np.ndarray, List[Union[Tensor, np.ndarray]]]] = None,
                 ref_frame_index: Optional[int] = None):
        """
        Call this method to resize images and masks
        :param images: Torch or Numpy array of shape [T, H, W, C]
        :param masks: Torch or Numpy array of shape [N, T, H, W]. Data type should be uint8 or bool. Can also be a list
        of torch tensors or numpy arrays
        :param ref_frame_index
        :return:
        """
        assert images.shape[3] in (1, 3)

        if self.mode == "none":
            if masks is None:
                return images
            else:
                return images, masks

        new_height, new_width = self.compute_resized_dims(*images.shape[1:3])

        if self.mode == "crop":
            raise RuntimeError("Crop resizing is not properly implemented for semantic masks. Please update the impl")
            assert masks is not None
            crop_result = self.perform_valid_cropping(images, masks, ref_frame_index, new_height, new_width)
            if crop_result is not None:
                # print("Crop successful", ref_frame_index is None)
                return crop_result
            # print("crop failed", ref_frame_index is None)

        if torch.is_tensor(images):
            images = rearrange(images, "B H W C -> B C H W")
            images = F.interpolate(images, (new_height, new_width), mode='bilinear', align_corners=False)
            images = rearrange(images, "B C H W -> B H W C")
        else:
            assert isinstance(images, np.ndarray)
            images = np.stack([cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR) for im in images])

        if masks is None:
            return images

        else:
            if isinstance(masks, (list, tuple)):
                masks = [self._resize_masks(masks[i], new_height, new_width) for i in range(len(masks))]
            else:
                masks = self._resize_masks(masks, new_height, new_width)
            # dtype = masks.dtype
            # if torch.is_tensor(masks):
            #     masks = F.interpolate(masks.float(), (new_height, new_width), mode='bilinear', align_corners=False)
            #     masks = (masks > 0.5).astype(dtype)
            # else:
            #     assert isinstance(masks, np.ndarray)
            #     B, N = masks.shape[:2]
            #     masks = np.reshape(masks, (-1, *masks.shape[2:]))
            #     masks = np.stack([
            #         (cv2.resize(m.astype(np.float32), (new_width, new_height), interpolation=cv2.INTER_LINEAR) > 0.5).astype(dtype)
            #         for m in masks
            #     ])
            #     masks = np.reshape(masks, (B, N, *masks.shape[1:]))

            return images, masks

    def compute_resized_dims(self, height: int, width: int):
        if self.mode in ("pixel_area", "crop"):
            ar = float(height) / float(width)

            new_height = int(math.floor(math.sqrt(ar * self.pixel_area)))
            new_width = self.pixel_area // new_height

        elif self.mode == "min_dim":
            dims = (height, width)
            lower_size = float(min(dims))
            higher_size = float(max(dims))

            if isinstance(self.min_dim, (list, tuple)):
                min_dim = self.min_dim[torch.randint(len(self.min_dim), (1,)).item()]
            else:
                min_dim = self.min_dim

            scale_factor = min_dim / lower_size
            if (higher_size * scale_factor) > self.max_dim:
                scale_factor = self.max_dim / higher_size

            new_height, new_width = round(scale_factor * height), round(scale_factor * width)

        else:
            raise ValueError("Should not be here")

        return new_height, new_width

    def perform_valid_cropping(self, images: Union[Tensor, np.ndarray], masks: Optional[Union[Tensor, np.ndarray]],
                               ref_frame_index: Union[int, None], new_height: int, new_width: int):
        # images: [T, H, W, 3]
        # masks: [N, T, H, W]
        assert images.shape[0] == masks.shape[1]
        assert images.shape[1:3] == masks.shape[2:4]

        return_torch_tensor = False
        if torch.is_tensor(images):
            images = images.numpy()
            masks = masks.numpy()
            return_torch_tensor = True

        # if target size is larger than the input image size then we need to up-sample instead of cropping
        if (new_height * new_width) >= (images.shape[1] * images.shape[2]):
            return None

        if ref_frame_index is None:
            # instance segmentation or semantic segmentation
            crop_params = compute_mask_containing_video_crop(
                mask=np.any(masks, 0), target_dims=(new_height, new_width), min_crop_mask_coverage=0.25,
                containment_criterion="nonzero", num_tries=20
            )

        else:
            # exemplar-based segmentation. Must ensure that all reference masks are included in the crop
            ref_mask_condensed = condense_mask(masks[:, ref_frame_index])
            crop_params = compute_mask_containing_image_crop(
                mask=ref_mask_condensed, target_dims=(new_height, new_width), min_crop_mask_coverage=0.25,
                containment_criterion="unique_values", num_tries=20
            )

        if crop_params is None:
            return None

        # print(f"{images.shape[1] * images.shape[2]} -> {new_height * new_width}")
        crop_x1, crop_y1 = crop_params
        crop_x2, crop_y2 = crop_x1 + new_width, crop_y1 + new_height

        cropped_images = images[:, crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_masks = masks[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

        if return_torch_tensor:
            cropped_images, cropped_masks = [
                torch.from_numpy(np.ascontiguousarray(x))
                for x in (cropped_images, cropped_masks)
            ]

        return cropped_images, cropped_masks
