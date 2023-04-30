from collections import defaultdict
from einops import rearrange
from typing import Dict, Any, List
from torch import Tensor
from tarvis.inference.result_formatters import ResultFormatterBase

import json
import numpy as np
import os.path as osp
import pycocotools.mask as mt
import torch
import torch.nn.functional as F
import zipfile


class YouTubeVISResultFormatter(ResultFormatterBase):
    def __init__(self, output_dir, max_tracks_per_video: int = 10):
        super().__init__(output_dir)
        self.max_tracks_per_video = max_tracks_per_video
        self.tracks = []

    def add_sequence_result(self, accumulator_output: Dict[str, Any], sequence_info: Dict[str, Any]):
        seq_masks = accumulator_output["track_masks"]  # [N, T, H, W] (float logits)
        assert seq_masks.dtype in (torch.float16, torch.float32)

        seq_class_logits = accumulator_output["track_class_logits"]  # [N, 1+C]

        seq_masks, seq_scores, seq_class_labels = self.filter_instances(seq_masks, seq_class_logits)

        # resize and threshold masks
        original_image_size = accumulator_output["original_image_size"]
        image_padding = accumulator_output["image_padding"]
        seq_masks = self.resize_to_original_dims(seq_masks, original_image_size, image_padding)
        seq_masks = (seq_masks > 0.0).cpu()  # threshold at prob = 0.5

        seq_masks = self.postprocess_tracks(seq_masks, seq_scores, seq_class_labels)

        sequence_length = len(sequence_info["image_paths"])
        seq_track_rles = [dict() for _ in range(sequence_length)]

        for track_id, (masks, score, class_id) in enumerate(zip(seq_masks, seq_scores, seq_class_labels), 1):
            assert class_id > 0
            masks = masks.byte().cpu().numpy()

            rles = []
            for t in range(sequence_length):
                if not np.any(masks[t]):
                    continue

                rle_mask = self.mask_to_rle(masks[t])
                seq_track_rles[t][track_id] = rle_mask
                rles.append(self.decode_rle_counts(rle_mask))

            self.tracks.append({
                "video_id": sequence_info["video_id"],
                "score": score.item(),
                "category_id": class_id.item(),
                "segmentations": rles
            })

        seq_class_labels = {track_id: cls_id for track_id, cls_id in enumerate(seq_class_labels.tolist(), 1)}
        seq_scores = {track_id: score for track_id, score in enumerate(seq_scores.tolist(), 1)}

        return {
            "track_mask_rles": seq_track_rles,
            "track_category_ids": seq_class_labels,
            "track_scores": seq_scores
        }

    def finalize_output(self):
        json_output_path = osp.join(self.output_dir, "results.json")
        with open(json_output_path, 'w') as fh:
            json.dump(self.tracks, fh)

        with zipfile.ZipFile(osp.join(self.output_dir, "results.zip"), 'w') as archive:
            archive.write(json_output_path, arcname="results.json")

    def filter_instances(self, masks, class_logits):
        # masks: [N, T, H, W],
        # class_logits: [N, C]
        assert masks.size(0) == class_logits.size(0)
        num_tracks = masks.size(0)
        num_classes = class_logits.size(1) - 1

        scores = F.softmax(class_logits.float(), dim=1)[:, 1:]  # [N, C]
        class_ids = torch.arange(1, num_classes + 1).unsqueeze(0).repeat(num_tracks, 1).flatten(0, 1)  # [N*C]

        track_scores, topk_indices = scores.flatten(0, 1).topk(self.max_tracks_per_video, sorted=False)  # [M]
        track_class_ids = class_ids[topk_indices]  # [M]

        topk_indices = torch.div(topk_indices, num_classes, rounding_mode='floor')
        selected_masks = masks[topk_indices].clone()  # [M, H, W]
        del masks  # <--- this tensor could be huge!

        return selected_masks, track_scores, track_class_ids

    def postprocess_tracks(self,
                           track_masks: Tensor,
                           track_scores: Tensor,
                           class_ids: Tensor):
        """
        YouTube-VIS does not allow masks to overlap if they share the same class label.
        :param track_masks: tensor of shape [N, T, H, W] (bool)
        :param track_scores: tensor of shape [N] (float32)
        :param class_ids: tensor of shape [N] (int64)
        :return:
        """
        assert track_masks.size(0) == track_scores.size(0) == class_ids.size(0), \
            f"track_masks: {track_masks.shape}, track_scores: {track_scores.shape}, class_ids: {class_ids.shape}"

        if len(set(class_ids.tolist())) == len(class_ids.tolist()):  # all tracks have unique class labels
            return track_masks

        # map class IDs to track IDs
        class_label_to_track_ids = defaultdict(list)

        for i, cls_id in enumerate(class_ids.tolist()):
            class_label_to_track_ids[cls_id].append(i)

        for cls_id, track_ids in class_label_to_track_ids.items():
            if len(track_ids) == 1:
                continue

            masks_cls = track_masks[track_ids]
            scores_cls = track_scores[track_ids]

            masks_cls = self.make_masks_nonoverlapping(masks_cls, scores_cls)
            track_masks[track_ids] = masks_cls

        return track_masks

    @classmethod
    def mask_to_rle(cls, mask: np.ndarray):
        assert mask.dtype == np.uint8
        return mt.encode(np.asfortranarray(mask))

    # @classmethod
    # def rle_to_mask(cls, rle: Dict[str, Any]) -> np.ndarray:
    #     return mt.decode(cls.decode_rle_counts(rle)).astype(bool)
    #
    @classmethod
    def decode_rle_counts(cls, rle: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "size": rle["size"],
            "counts": rle["counts"].decode("utf-8")
        }
