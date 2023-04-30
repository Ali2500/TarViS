from einops import rearrange
from torch import Tensor
from typing import List, Dict, Tuple, Any, Optional

import pycocotools.mask as mt
import numpy as np
import torch

from tarvis.inference.result_accumulators.result_accumulator_base import ResultAccumulatorBase


class SemanticSegmentationResultAccumulator(ResultAccumulatorBase):
    def __init__(self,
                 sequence_length: int,
                 original_image_size: Optional[Tuple[int, int]],
                 image_padding: Optional[Tuple[int, int]],
                 ignore_classes: Optional[List[int]] = None):
        super().__init__()

        self.original_image_size = original_image_size  # [H, W]
        self.image_padding = image_padding  # [right, bottom]

        if ignore_classes is None:
            self.ignore_classes = None
        else:
            self.ignore_classes = torch.as_tensor(ignore_classes, dtype=torch.int64)

        self.previous_clip_logits: Tensor = torch.empty(0)  # [1+num_classes, T, H, W]
        self.previous_clip_frame_indices = []
        self.output_masks: List[Tensor] = [None for _ in range(sequence_length)]

    @torch.no_grad()
    def add_clip_result(self, model_output: Dict[str, Any], frame_indices: Tensor):
        """
        Add the result for a sub-clip of the inference video to the accumulator.
        :param model_output: dictionary with the following entries:
            (1) pred_semantic_seg_logits: tensor of shape [num_classes, T, H, W]
        :param frame_indices: the frame indices for the current result
        :return:
        """
        frame_indices = frame_indices.tolist()
        # get list of frame indices that overlap with the previous clip
        overlap_frames = sorted(list(set(frame_indices).intersection(set(self.previous_clip_frame_indices))))
        current_clip_logits = model_output["pred_semantic_seg_logits"]
        # print(f"Current: {frame_indices}")

        if len(overlap_frames) > 0:
            assert overlap_frames == self.previous_clip_frame_indices[-len(overlap_frames):], \
                f"{overlap_frames}, {self.previous_clip_frame_indices}"
            assert overlap_frames == frame_indices[:len(overlap_frames)], f"{overlap_frames}, {frame_indices}"

            # average overlap logits
            prev_clip_overlap_logits = self.previous_clip_logits[:, -len(overlap_frames):]
            current_clip_overlap_logits = current_clip_logits[:, :len(overlap_frames)]

            averaged_logits = (prev_clip_overlap_logits + current_clip_overlap_logits) / 2.0
            self.previous_clip_logits[:, -len(overlap_frames):] = averaged_logits

            self.finalize_previous_clip_logits()

            self.previous_clip_logits = current_clip_logits[:, len(overlap_frames):]
            self.previous_clip_frame_indices = frame_indices[len(overlap_frames):]

        else:
            self.previous_clip_logits = current_clip_logits
            self.previous_clip_frame_indices = frame_indices

        if frame_indices[-1] == len(self.output_masks) - 1:  # this is the final clip of the video
            self.finalize_previous_clip_logits()

    def finalize_previous_clip_logits(self):
        if self.ignore_classes is not None:
            # set logits for ignore classes to -inf
            # print(f"thing classes: {self.ignore_classes}")
            index = self.ignore_classes.to(device=self.previous_clip_logits.device)
            self.previous_clip_logits.index_fill_(0, index, -1e3)  # filling with -inf does weird things with float16 :S

        semantic_masks = self.resize_to_original_dims(self.previous_clip_logits)
        semantic_masks = semantic_masks.argmax(0).cpu()  # [T, H, W] (non-background prediction for every pixel)
        # print(f"Person pixels: {(semantic_masks == 11).sum()}")
        # print("Finalizing: ", self.previous_clip_frame_indices)

        # import cv2
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)

        for frame_id, mask_t in zip(self.previous_clip_frame_indices, semantic_masks):
            # for cls_id in (0, 2, 8, 9, 10):
            # for cls_id in mask_t.unique() - 1:
            #     print(cls_id)
            #     msk = mask_t == (cls_id + 1)
            #     cv2.imshow("image", msk.byte().numpy() * 255)
            #     cv2.waitKey(0)

            assert self.output_masks[frame_id] is None, f"Finalized masks for frame {frame_id} already exist!"
            self.output_masks[frame_id] = mask_t

    def finalize_output(self):
        return {
            "semantic_seg_masks": self.output_masks
        }
