from abc import abstractmethod
from typing import Dict, List, Tuple, Any
from torch import Tensor
from tarvis.inference.misc import is_oom_error

import torch
import torch.nn.functional as F


class ResultAccumulatorBase:
    def __init__(self):
        self.image_padding: Tuple[int, int] = tuple()
        self.original_image_size: Tuple[int, int] = tuple()
        pass

    @abstractmethod
    def add_clip_result(self, model_output: Dict[str, Any], frame_indices: List[int]):
        pass

    @abstractmethod
    def finalize_output(self):
        pass

    def resize_to_original_dims(self, x: Tensor):
        """
        Resize the given tensor to original input dimensions
        :param x: 4-D tensor with last 2 dimensions being (H, W)
        :return:
        """
        # (1) Model outputs 4x downsampled masks so first reverse that
        x = F.interpolate(x, scale_factor=4.0, mode='bilinear', align_corners=False)

        # (2) Reverse padding
        pad_right, pad_bottom = self.image_padding
        x = F.pad(x, (0, -pad_right, 0, -pad_bottom))

        # (3) Resize to input resolution
        input_height, input_width = self.original_image_size

        try:
            x = F.interpolate(x, (input_height, input_width), mode='bilinear', align_corners=False)
        except RuntimeError as err:
            if not is_oom_error(err):
                raise err

            torch.cuda.empty_cache()

            # split tensor along time into chunks and move the result to CPU
            x = x.chunk(4, dim=0)

            x = torch.cat([
                F.interpolate(chunk, (input_height, input_width), mode='bilinear', align_corners=False).cpu()
                for chunk in x
            ], 0)

        return x
