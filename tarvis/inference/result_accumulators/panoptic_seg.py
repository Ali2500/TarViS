from torch import Tensor
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional

import pycocotools.mask as mt
import numpy as np
import torch

from tarvis.inference.result_accumulators import ResultAccumulatorBase
from tarvis.inference.result_accumulators import InstanceSegmentationResultAccumulator
from tarvis.inference.result_accumulators.semantic_seg import SemanticSegmentationResultAccumulator


class PanopticSegmentationResultAccumulator(ResultAccumulatorBase):
    def __init__(self,
                 sequence_length: int,
                 max_tracks_per_clip: int,
                 track_score_threshold: float,
                 original_image_size: Optional[Tuple[int, int]],
                 image_padding: Optional[Tuple[int, int]],
                 thing_class_ids: List[int],
                 association_iou_threshold: float = 0.5,
                 tensor_ops_on_cpu: bool = False):
        super().__init__()

        self.original_image_size = original_image_size  # [H, W]
        self.image_padding = image_padding  # [right, bottom]

        self.instance_seg_accumulator = InstanceSegmentationResultAccumulator(
            sequence_length=sequence_length,
            max_tracks_per_clip=max_tracks_per_clip,
            track_score_threshold=track_score_threshold,
            produce_semseg_output=False,
            original_image_size=original_image_size,
            image_padding=image_padding,
            association_iou_threshold=association_iou_threshold,
            tensor_ops_on_cpu=tensor_ops_on_cpu
        )

        self.semantic_seg_accumulator = SemanticSegmentationResultAccumulator(
            sequence_length=sequence_length,
            original_image_size=original_image_size,
            image_padding=image_padding,
            ignore_classes=thing_class_ids
        )

    def __setattr__(self, key, value):
        if key == "image_padding" and hasattr(self, "instance_seg_accumulator"):
            setattr(self.instance_seg_accumulator, "image_padding", value)
            setattr(self.semantic_seg_accumulator, "image_padding", value)

        super().__setattr__(key, value)

    @torch.no_grad()
    def add_clip_result(self, model_output: Dict[str, Any], frame_indices: Tensor):
        """
        Add the result for a sub-clip of the inference video to the accumulator.
        :param model_output: dictionary with the following entries:
            (1) pred_instance_seg_logits: tensor of shape [N, T, H, W] where N = number of instances
            (2) pred_instance_class_logits: tensor of shape [N, 1+num_classes]
            (3) pred_semantic_seg_logits: tensor of shape [1+num_classes, T, H, W]
        :param frame_indices: frame_indices: the frame indices for the current result
        :return:
        """
        self.instance_seg_accumulator.add_clip_result(model_output, frame_indices)
        self.semantic_seg_accumulator.add_clip_result(model_output, frame_indices)

    def finalize_output(self):
        outputs = self.instance_seg_accumulator.finalize_output()
        outputs.update(self.semantic_seg_accumulator.finalize_output())
        return outputs
