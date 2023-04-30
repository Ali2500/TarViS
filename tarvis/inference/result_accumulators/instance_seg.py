from collections import defaultdict
from einops import rearrange
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional

import itertools
import pycocotools.mask as mt
import numpy as np
import torch

from tarvis.inference.result_accumulators import ResultAccumulatorBase
from tarvis.inference.result_accumulators.semantic_seg import SemanticSegmentationResultAccumulator


class InstanceSegmentationResultAccumulator(ResultAccumulatorBase):
    def __init__(self,
                 sequence_length: int,
                 max_tracks_per_clip: int,
                 track_score_threshold: float,
                 produce_semseg_output: bool,
                 original_image_size: Optional[Tuple[int, int]] = None,
                 image_padding: Optional[Tuple[int, int]] = None,
                 association_iou_threshold: float = 0.5,
                 tensor_ops_on_cpu: bool = False):
        super().__init__()

        self.original_image_size = original_image_size  # [H, W]
        self.image_padding = image_padding  # [right, bottom]

        self.previous_clip_frame_indices = []
        self.previous_clip_logits = torch.empty(0)  # [N, T, H, W]
        self.previous_clip_track_ids = torch.empty(0)  # [N]
        self.next_track_id = 1

        self.track_score_threshold = track_score_threshold
        self.max_tracks_per_clip = max_tracks_per_clip
        self.association_iou_threshold = association_iou_threshold

        self.track_class_logits = defaultdict(list)
        self.output_masks: List[Dict[int, Any]] = [dict() for _ in range(sequence_length)]
        # self.output_masks: List[Tensor] = [torch.empty(0) for _ in range(sequence_length)]

        self.semantic_seg_result_accumulator = None
        if produce_semseg_output:
            self.semantic_seg_result_accumulator = SemanticSegmentationResultAccumulator(
                sequence_length=sequence_length, original_image_size=original_image_size, image_padding=image_padding
            )

        self.tensor_ops_on_cpu = tensor_ops_on_cpu

    @property
    def semseg_enabled(self):
        return self.semantic_seg_result_accumulator is not None

    @torch.no_grad()
    def add_clip_result(self, model_output: Dict[str, Any], frame_indices: Tensor):
        """
        Add the result for a sub-clip of the inference video to the accumulator.
        :param model_output: dictionary with the following entries:
            (1) pred_instance_seg_logits: tensor of shape [N, T, H, W] where N = number of instances
            (2) pred_instance_class_logits: tensor of shape [N, 1+num_classes]
        :param frame_indices: frame_indices: the frame indices for the current result
        :return:
        """
        # print(frame_indices.tolist())
        if self.semseg_enabled:
            # add the semantic segmentation result to sits accumulator
            if self.semantic_seg_result_accumulator.image_padding is None:
                self.semantic_seg_result_accumulator.image_padding = self.image_padding
                self.semantic_seg_result_accumulator.original_image_size = self.original_image_size

            self.semantic_seg_result_accumulator.add_clip_result(model_output, frame_indices)

        mask_logits: Tensor = model_output["pred_instance_seg_logits"]
        class_logits: Tensor = model_output["pred_instance_class_logits"]

        if self.tensor_ops_on_cpu:
            mask_logits = mask_logits.float().cpu()
            class_logits = class_logits.float().cpu()

        mask_logits, class_logits = self.filter_clip_tracks(mask_logits, class_logits)

        if mask_logits.size(0) == 0:  # no foreground masks predicted
            self.finalize_previous_clip_masks()
            self.previous_clip_frame_indices = []
            self.previous_clip_logits = torch.empty(0)
            self.previous_clip_track_ids = torch.empty(0)

        overlap_frame_indices = set(frame_indices.tolist()).intersection(set(self.previous_clip_frame_indices))
        num_overlap_frames = len(overlap_frame_indices)

        if num_overlap_frames > 0:
            # print("Overlap length: ", num_overlap_frames)
            assert set(frame_indices[:num_overlap_frames].tolist()) == overlap_frame_indices
            assert set(self.previous_clip_frame_indices[-num_overlap_frames:]) == overlap_frame_indices

            # (1) Compute linear assignment between previous clip and current clip logits
            previous_clip_overlap_logits = self.previous_clip_logits[:, -num_overlap_frames:]  # [N', To, H, W]
            current_clip_overlap_logits = mask_logits[:, :num_overlap_frames]

            idxes_prev, idxes_current = self.associate_tracks(previous_clip_overlap_logits, current_clip_overlap_logits)

            if idxes_prev.numel() > 0:
                # (2) Average logits between associated tracks
                averaged_logits = (previous_clip_overlap_logits[idxes_prev] + current_clip_overlap_logits[idxes_current]) / 2.0
                self.previous_clip_logits[idxes_prev][:, -num_overlap_frames:] = averaged_logits

            # (3) create track IDs for current clip masks
            current_clip_track_ids = torch.full((mask_logits.size(0),), -1, dtype=torch.int64)
            current_clip_track_ids[idxes_current] = self.previous_clip_track_ids[idxes_prev]

            num_new_tracks = mask_logits.size(0) - idxes_prev.numel()
            current_clip_track_ids[current_clip_track_ids < 0] = torch.arange(
                self.next_track_id, self.next_track_id + num_new_tracks, dtype=torch.int64
            )
            assert not torch.any(current_clip_track_ids < 0)
            self.next_track_id += num_new_tracks

        else:
            # print("No overlap")
            current_clip_track_ids = torch.arange(self.next_track_id, self.next_track_id + mask_logits.size(0))
            self.next_track_id += mask_logits.size(0)

        # finalize result for previous clip
        self.finalize_previous_clip_masks()

        # add class logits to buffer
        for track_id, logits in zip(current_clip_track_ids.tolist(), class_logits):
            self.track_class_logits[track_id].append(logits)

        # update buffers for next iteration
        self.previous_clip_frame_indices = frame_indices[num_overlap_frames:].tolist()
        self.previous_clip_logits = mask_logits[:, num_overlap_frames:]
        self.previous_clip_track_ids = current_clip_track_ids

        if frame_indices[-1] == len(self.output_masks) - 1:  # last clip
            self.finalize_previous_clip_masks()

        # print("Next track ID: ", self.next_track_id)

    def associate_tracks(self, prev_clip_logits: Tensor, current_clip_logits: Tensor):
        """
        Associate tracks using linear assignment based on their mask IoU
        :param prev_clip_logits: [N, T, H, W]
        :param current_clip_logits: [M, T, H, W]
        :return:
        """
        masks_prev = (prev_clip_logits > 0).flatten(1)[:, None, :]  # [N, 1, T*H*W]
        masks_current = (current_clip_logits > 0).flatten(1)[None, :, :]  # [1, M, T*H*W]

        try:
            intersection = torch.logical_and(masks_prev, masks_current).sum(2, dtype=torch.float32)  # [N, M]
            union = torch.logical_or(masks_prev, masks_current).sum(2, dtype=torch.float32)  # [N, M]
            iou = (intersection + 1.) / (union + 1.)  # [N, M]
        except RuntimeError as err:  # OOM error
            masks_prev = masks_prev.cpu()
            masks_current = masks_current.cpu()
            intersection = torch.logical_and(masks_prev, masks_current).sum(2, dtype=torch.float32)  # [N, M]
            union = torch.logical_or(masks_prev, masks_current).sum(2, dtype=torch.float32)  # [N, M]
            iou = (intersection + 1.) / (union + 1.)  # [N, M]

        idxes_prev, idxes_current = linear_sum_assignment(iou.cpu(), maximize=True)
        idxes_prev = torch.from_numpy(idxes_prev)
        idxes_current = torch.from_numpy(idxes_current)

        association_ious = iou[(idxes_prev, idxes_current)]

        valid_matches = association_ious > self.association_iou_threshold
        idxes_prev = idxes_prev[valid_matches]
        idxes_current = idxes_current[valid_matches]

        return idxes_prev, idxes_current

    def filter_clip_tracks(self, mask_logits: Tensor, class_logits: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Filter the predicted tracks based on score and retain only the top-K (K = self.max_tracks_per_clips)
        :param mask_logits: tensor of shape [N, T, H, W] (float32)
        :param class_logits: tensor of shape [N, 1+num_classes]
        :return:
        """
        scores = class_logits.softmax(1)[:, 1:].max(1).values  # [N]

        if self.track_score_threshold > 0:
            keep_mask = scores >= self.track_score_threshold  # [N]
            mask_logits = mask_logits[keep_mask]  # [N', T, H, W]
            class_logits = class_logits[keep_mask]  # [N', 1+num_classes]
            scores = scores[keep_mask]  # [N']

        if class_logits.size(0) > self.max_tracks_per_clip:
            topk_indices = scores.topk(self.max_tracks_per_clip).indices
            mask_logits = mask_logits[topk_indices]
            class_logits = class_logits[topk_indices]

        return mask_logits, class_logits

    def finalize_previous_clip_masks(self):
        if self.previous_clip_logits.numel() == 0:
            return

        prev_clip_logits = (self.resize_to_original_dims(self.previous_clip_logits).transpose(0, 1) > 0).byte().cpu()  # [T, N, H, W]
        for t, masks_t in zip(self.previous_clip_frame_indices, prev_clip_logits):
            for track_id, mask in zip(self.previous_clip_track_ids.tolist(), masks_t):
                try:
                    assert track_id not in self.output_masks[t], f"Track IDs {self.output_masks[t].keys()} already exist for frame {t}"
                except AssertionError as err:
                    breakpoint()
                self.output_masks[t][track_id] = mt.encode(np.asfortranarray(mask.numpy()))

    def finalize_output(self):
        # average class logits to obtain class scores for every
        track_class_logits = {
            track_id: torch.stack(class_logits).mean(0) for track_id, class_logits in self.track_class_logits.items()
        }

        return {
            "track_masks": self.output_masks,
            "track_class_logits": track_class_logits
        }
