from torch import Tensor
from typing import List, Dict, Tuple, Any, Optional

import pycocotools.mask as mt
import numpy as np
import torch

from tarvis.inference.result_accumulators.result_accumulator_base import ResultAccumulatorBase


class ExemplarBasedSegmentationResultAccumulator(ResultAccumulatorBase):
    def __init__(self,
                 sequence_length: int,
                 first_ref_mask_frame_index: int,
                 original_image_size: Optional[Tuple[int, int]] = None,
                 image_padding: Optional[Tuple[int, int]] = None):
        super().__init__()

        self.original_image_size = original_image_size  # [H, W]
        self.image_padding = image_padding  # [right, bottom]
        self.first_ref_mask_frame_index = first_ref_mask_frame_index

        self.previous_clip_logits: Tensor = torch.empty(0)  # N, T, H, W
        self.previous_clip_instance_ids = []  # [N]
        self.previous_clip_frame_indices: Tensor = torch.empty(0)  # [N]
        self.lost_instance_ids = []

        self.output_masks: List[Dict[int, Any]] = [dict() for _ in range(sequence_length)]

    def reset_buffers(self):
        self.previous_clip_logits: Tensor = torch.empty(0)  # N, T, H, W
        self.previous_clip_instance_ids = []  # [N]
        self.previous_clip_frame_indices: Tensor = torch.empty(0)  # [N]

    @torch.no_grad()
    def add_clip_result(self, model_output: Dict[str, Any], frame_indices: Tensor):
        """
        Add the result for a sub-clip of the inference video to the accumulator.
        :param model_output: dictionary with the following entries:
            (1) pred_vos_logits: tensor of shape [N, T, H, W] where N = number of instances
            (2) instance_ids: tensor of shape [N] denoting instance IDs
            (2) lost_instance_ids: list of instance IDs which can no longer be tracked
        :param frame_indices: the frame indices for the current result
        :return:
        """
        self.lost_instance_ids.extend(model_output["lost_instance_ids"])
        current_clip_mask_logits = model_output["pred_vos_logits"]
        current_clip_instance_ids: List[int] = model_output["instance_ids"].tolist()

        if model_output is None:  # all instances lost!
            self.previous_clip_logits = torch.empty(0)
            self.previous_clip_frame_indices = torch.empty(0)
            self.previous_clip_instance_ids = []
            return

        # get list of frame indices that overlap with the previous clip
        overlap_frame_indices = sorted(list(set(frame_indices.tolist()).intersection(set(self.previous_clip_frame_indices.tolist()))))
        overlap_frame_indices = torch.tensor(overlap_frame_indices).to(frame_indices)
        num_overlap_frames = overlap_frame_indices.numel()

        if not torch.all(overlap_frame_indices == frame_indices[:num_overlap_frames]) or \
            not torch.all(overlap_frame_indices == self.previous_clip_frame_indices[-num_overlap_frames:]):

            raise RuntimeError(
                f"Frame indices not as expected. Current frame indices: {frame_indices.tolist()}, previous clip "
                f"frame indices: {self.previous_clip_frame_indices.tolist()}"
            )

        if self.previous_clip_logits.numel() > 0:
            # average the logits in the overlapping frames
            overlapping_instance_ids = set(self.previous_clip_instance_ids).intersection(set(current_clip_instance_ids))
            overlapping_instance_ids = list(overlapping_instance_ids)

            prev_indices = [self.previous_clip_instance_ids.index(iid) for iid in overlapping_instance_ids]
            current_indices = [current_clip_instance_ids.index(iid) for iid in overlapping_instance_ids]

            prev_clip_overlap_logits = self.previous_clip_logits[prev_indices][:, -num_overlap_frames:]  # [N_ol, T, H, W]
            current_clip_overlap_logits = current_clip_mask_logits[current_indices][:, :num_overlap_frames]  # [N_ol, T, H, W]

            averaged_logits = (prev_clip_overlap_logits + current_clip_overlap_logits) / 2.0
            self.previous_clip_logits[prev_indices][:, -num_overlap_frames:] = averaged_logits  # not sure if this actually overwrites

            # finalize previous clip
            self.previous_clip_to_rles()

        if frame_indices[-1].item() == len(self.output_masks) - 1:  # current clip includes last frame
            self.previous_clip_logits = current_clip_mask_logits[num_overlap_frames:]
            self.previous_clip_frame_indices = frame_indices[num_overlap_frames:]
            self.previous_clip_logits = current_clip_mask_logits[:, num_overlap_frames:]
            self.previous_clip_to_rles()

        else:
            # update buffers with current clip
            self.previous_clip_logits = current_clip_mask_logits
            self.previous_clip_frame_indices = frame_indices
            self.previous_clip_instance_ids = current_clip_instance_ids

    def previous_clip_to_rles(self):
        if self.previous_clip_logits.numel() == 0:
            return

        prev_clip_logits = self.resize_to_original_dims(self.previous_clip_logits)  # [N, T, H, W]
        prev_clip_logits, prev_clip_masks = prev_clip_logits.max(0)  # [T, H, W]
        prev_clip_masks = torch.where(prev_clip_logits > 0.0, prev_clip_masks + 1, torch.zeros_like(prev_clip_masks))

        for inst_index, inst_id in enumerate(self.previous_clip_instance_ids, 1):
            instance_masks = prev_clip_masks == inst_index  # [T, H, W]
            for t, mask_t in zip(self.previous_clip_frame_indices.tolist(), instance_masks):
                if not torch.any(mask_t):
                    continue

                self.output_masks[t][inst_id] = mt.encode(np.asfortranarray(mask_t.byte().cpu().numpy()))

    def finalize_output(self):
        return {
            "track_mask_rles": self.output_masks
        }
