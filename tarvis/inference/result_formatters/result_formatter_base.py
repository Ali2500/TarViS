from abc import abstractmethod
from torch import Tensor
from typing import Tuple

import os
import torch
import torch.nn.functional as F


def is_oom_error(exc: RuntimeError):
    return repr(exc).startswith("RuntimeError('CUDA out of memory.")


class ResultFormatterBase:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    @abstractmethod
    def add_sequence_result(self, *args, **kwargs):
        pass

    @abstractmethod
    def finalize_output(self):
        pass

    @classmethod
    def make_masks_nonoverlapping(cls, masks: Tensor, scores: Tensor) -> Tensor:
        """
        Makes a collection of masks non-overlapping based on the given scores
        :param masks: tensor of shape [N, T, H, W]
        :param scores: scores which are used to decide which mask gets precedence. Higher score means higher priority.
        Note: Behavior is non-deterministic if multiple masks share the same score.
        :return
        """
        assert masks.ndim == 4
        assert scores.ndim == 1
        assert masks.size(0) == scores.size(0)

        sort_idx = scores.argsort(descending=True)
        masks = masks[sort_idx]

        sentinel = masks.cumsum(0, dtype=torch.int64) == 1  # [N, T, H, W]
        masks = torch.where(sentinel, masks, torch.zeros_like(masks))

        return masks[sort_idx.argsort()]

    @classmethod
    def resize_to_original_dims(cls, x: Tensor, original_image_size: Tuple[int, int], image_padding: Tuple[int, int]):
        """
        Resize the given tensor to original input dimensions
        :param x: 4-D tensor with last 2 dimensions being (H, W)
        :param original_image_size
        :param image_padding
        :return:
        """
        # (1) Model outputs 4x downsampled masks so first reverse that
        x = F.interpolate(x, scale_factor=4.0, mode='bilinear', align_corners=False)

        # (2) Reverse padding
        pad_right, pad_bottom = image_padding
        x = F.pad(x, (0, -pad_right, 0, -pad_bottom))

        # (3) Resize to input resolution
        input_height, input_width = original_image_size

        try:
            x = F.interpolate(x, (input_height, input_width), mode='bilinear', align_corners=False)
        except RuntimeError as err:
            if not is_oom_error(err):
                raise err

            torch.cuda.empty_cache()
            print("OOM error occurred")

            # split tensor along time into chunks and move the result to CPU
            x = x.chunk(4, dim=0)

            x = torch.cat([
                F.interpolate(chunk, (input_height, input_width), mode='bilinear', align_corners=False).cpu()
                for chunk in x
            ], 0)

        return x
