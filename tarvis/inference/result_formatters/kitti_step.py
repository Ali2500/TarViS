from collections import defaultdict
from typing import Dict, Any, List, Union, Tuple
from torch import Tensor
from tarvis.inference.result_formatters import ResultFormatterBase

import cv2
import numpy as np
import os
import os.path as osp
import pycocotools.mask as mt
import torch


class KITTISTEPResultFormatter(ResultFormatterBase):
    def __init__(self, output_dir, track_score_threshold: float = 0.5, max_tracks_per_video: int = 1000):
        super().__init__(output_dir)

        self.track_score_threshold = track_score_threshold
        self.max_tracks_per_video = max_tracks_per_video
        self.tracks = []

        # instance IDs are only possible for thing classes. These have IDs 11 and 13 for KITTI-STEP
        self.thing_classes = [11, 13]  # 'person', 'car'
        self.all_class_ids = list(range(0, 19))  # [0-18]
        self.stuff_class_ids = [i for i in self.all_class_ids if i not in self.thing_classes]

    def add_sequence_result(self, accumulator_output: Dict[str, Any], sequence_info: Dict[str, Any]):
        seq_track_rles = accumulator_output["track_masks"]  # List[Dict[track_id -> mask RLE]]
        seq_class_logits = accumulator_output["track_class_logits"]  # Dict[track_id ->, Tensor[1+num_classes]]
        seq_semantic_masks = accumulator_output["semantic_seg_masks"]  # [T, H, W]  (final image size)

        seq_track_rles, seq_scores, seq_class_labels = self.filter_instances(seq_track_rles, seq_class_logits)

        seq_panoptic_maps = self.generate_panoptic_maps(
            seq_track_rles, seq_scores, seq_class_labels, seq_semantic_masks
        )
        assert len(seq_panoptic_maps) == len(sequence_info["image_paths"]), \
            f"{len(seq_panoptic_maps)}, {len(sequence_info['image_paths'])}"

        # write out panoptic maps as PNG images
        seq_output_dir = osp.join(self.output_dir, sequence_info["dirname"])
        os.makedirs(seq_output_dir, exist_ok=True)

        for panoptic_map, input_img_path in zip(seq_panoptic_maps, sequence_info["image_paths"]):
            filename = osp.splitext(osp.split(input_img_path)[-1])[0]
            cv2.imwrite(osp.join(seq_output_dir, f"{filename}.png"), panoptic_map)

        return {
            "panoptic_masks": seq_panoptic_maps,
            "track_category_ids": seq_class_labels,
            "track_scores": seq_scores
        }

    def finalize_output(self):
        pass

    def filter_instances(self, seq_rle_masks, class_logits):
        if len(class_logits) == 0:
            return seq_rle_masks, dict(), dict()

        # filter by score threshold
        track_ids = list(class_logits.keys())  # [N]
        track_class_logits = torch.stack([class_logits[t_id] for t_id in track_ids], 0)  # [N, 1+num_classes]
        scores, class_ids = track_class_logits.softmax(1).max(1)
        # print("Pre-filtering", class_ids.tolist())
        track_ids = torch.as_tensor(track_ids, dtype=torch.int64)

        # remove background predictions
        keep_mask = class_ids > 0
        track_ids = track_ids[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        # convert class IDs from 1-based index to 0-based index
        class_ids = class_ids - 1

        # remove predictions which have stuff class labels
        remove_mask = torch.zeros_like(class_ids, dtype=torch.bool)
        for disallowed_class_id in self.stuff_class_ids:
            remove_mask = torch.logical_or(remove_mask, class_ids == disallowed_class_id)

        keep_mask = torch.logical_not(remove_mask)
        track_ids = track_ids[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        # apply score threshold
        keep_mask = scores > self.track_score_threshold

        track_ids = track_ids[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        # print("Post-filtering: ", class_ids.tolist())
        # apply max tracks threshold
        if track_ids.numel() > self.max_tracks_per_video:
            scores, topk_indices = scores.topk(self.max_tracks_per_video, largest=True)
            track_ids = track_ids[topk_indices]
            class_ids = class_ids[topk_indices]

        track_ids = track_ids.tolist()

        for rles_t in seq_rle_masks:
            for t_id in list(rles_t.keys()):
                if t_id not in track_ids:
                    rles_t.pop(t_id)

        scores = {t_id: score for t_id, score in zip(track_ids, scores.tolist())}
        class_ids = {t_id: class_id for t_id, class_id in zip(track_ids, class_ids.tolist())}

        return seq_rle_masks, scores, class_ids

    def generate_panoptic_maps(self,
                               track_rle_masks: List[Dict[int, Any]],
                               track_scores: Dict[int, float],
                               class_ids: Dict[int, int],
                               semantic_masks: List[Tensor]) -> List[np.ndarray]:
        """
        TrackEval does not allow masks to overlap
        :param track_rle_masks: list of dicts (one per frame), with track IDs as keys and RLE encoded mask as values
        :param track_scores: Dict with track IDs as keys and normalized score in [0, 1] as value
        :param class_ids: Dict with track IDs as keys and class ID as value
        :param semantic_masks: list of length T, with each element being a tensor of shape [H, W] where pixel values
        denote class ID (0-based index)
        :return:
        """
        assert len(semantic_masks) == len(track_rle_masks), f"{semantic_masks.shape}, {len(track_rle_masks)}"
        panoptic_masks = []

        for t in range(len(track_rle_masks)):
            semantic_mask_t = semantic_masks[t]
            instance_mask_t = torch.zeros_like(semantic_mask_t)

            track_ids = list(track_rle_masks[t].keys())
            if len(track_ids) > 0:

                rle_masks = list(track_rle_masks[t].values())
                masks_t = torch.stack([self.rle_to_mask(rle_mask) for rle_mask in rle_masks])  # [N, H, W]

                scores_t = torch.as_tensor([track_scores[t_id] for t_id in track_ids], dtype=torch.float32)  # [N]
                class_ids_t = torch.as_tensor([class_ids[t_id] for t_id in track_ids], dtype=torch.int64)  # [N]
                track_ids = torch.as_tensor(track_ids, dtype=torch.int64)  # [N]

                scores_t, sort_perm = scores_t.sort()
                track_ids = track_ids[sort_perm]
                class_ids_t = class_ids_t[sort_perm]
                masks_t = masks_t[sort_perm]

                for m, track_id, cls_id in zip(masks_t, track_ids, class_ids_t):
                    instance_mask_t = torch.where(m, track_id, instance_mask_t)
                    assert cls_id.item() in self.thing_classes, f"Invalid instance class ID: {cls_id.item()}"

                    # overwrite semantic mask with class ID for this track
                    semantic_mask_t = torch.where(m, cls_id, semantic_mask_t)

            panoptic_masks.append(self.parse_instance_and_semantic_mask(instance_mask_t, semantic_mask_t))

        return panoptic_masks  # , class_ids

    def parse_instance_and_semantic_mask(self, instance_mask: Tensor, semantic_mask: Tensor) -> np.ndarray:
        # expected format for RGB mask output:
        # - R: semantic ID
        # - G: instance_id // 256
        # - B: instance_id % 256
        panoptic_mask = torch.stack([
            instance_mask % 256,
            torch.div(instance_mask, 256, rounding_mode='floor'),
            semantic_mask
        ], 2)  # [H, W, 3]. We make an array in BGR format because we'll use OpenCV to save it

        return panoptic_mask.byte().cpu().numpy()

    @classmethod
    def mask_to_rle(cls, mask: Union[Tensor, np.ndarray]):
        if torch.is_tensor(mask):
            mask = mask.byte().cpu().numpy()
        assert mask.dtype == np.uint8
        return mt.encode(np.asfortranarray(mask))

    @classmethod
    def rle_to_mask(cls, rle: Dict[str, Any]) -> Tensor:
        return torch.from_numpy(mt.decode(rle).astype(bool))

    @classmethod
    def decode_rle_counts(cls, rle: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "size": rle["size"],
            "counts": rle["counts"].decode("utf-8")
        }
