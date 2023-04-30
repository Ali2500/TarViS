from collections import defaultdict
from typing import Dict, Any, List, Union
from torch import Tensor
from tarvis.inference.result_formatters import ResultFormatterBase

import json
import numpy as np
import os.path as osp
import pycocotools.mask as mt
import torch
import torch.nn.functional as F
import zipfile


class OVISResultFormatter(ResultFormatterBase):
    def __init__(self, output_dir, max_tracks_per_video: int):
        super().__init__(output_dir)
        self.max_tracks_per_video = max_tracks_per_video
        self.tracks = []

    def add_sequence_result(self, accumulator_output: Dict[str, Any], sequence_info: Dict[str, Any]):
        seq_track_rles = accumulator_output["track_masks"]  # List[Dict[track_id -> mask RLE]]
        seq_class_logits = accumulator_output["track_class_logits"]  # Dict[track_id ->, Tensor[1+num_classes]]

        seq_track_rles, seq_scores, seq_class_labels = self.filter_instances(seq_track_rles, seq_class_logits)
        seq_track_rles = self.postprocess_tracks(seq_track_rles, seq_scores, seq_class_labels)

        for track_id, score in seq_scores.items():
            class_id = seq_class_labels[track_id]
            masks = [None for _ in range(len(seq_track_rles))]
            for t, masks_t in enumerate(seq_track_rles):
                if track_id in masks_t:
                    masks[t] = self.decode_rle_counts(masks_t[track_id])

            self.tracks.append({
                "video_id": sequence_info["video_id"],
                "score": score,
                "category_id": class_id,
                "segmentations": masks
            })

        return {
            "track_mask_rles": seq_track_rles,
            "track_category_ids": seq_class_labels,
            "track_scores": seq_scores
        }

    def finalize_output(self):
        json_output_path = osp.join(self.output_dir, "results.json")
        with open(json_output_path, 'w') as fh:
            json.dump(self.tracks, fh)

        # set filename of zip to the model directory name
        model_dirname = self.output_dir.split("/")[-3]

        with zipfile.ZipFile(osp.join(self.output_dir, f"{model_dirname}.zip"), 'w') as archive:
            archive.write(json_output_path, arcname="results.json")

    def filter_instances(self, seq_rle_masks, class_logits):
        # class_logits: Dict[track_id -> logits]
        track_ids = torch.as_tensor(list(class_logits.keys()), dtype=torch.int64)
        num_tracks = len(track_ids)

        class_logits = torch.stack(list(class_logits.values()))  # [N, 1+C]
        num_classes = class_logits.size(1) - 1

        scores = F.softmax(class_logits.float(), dim=1)[:, 1:]  # [N, C]
        class_ids = torch.arange(1, num_classes + 1).unsqueeze(0).repeat(num_tracks, 1).flatten(0, 1)  # [N*C]

        track_scores, topk_indices = scores.flatten(0, 1).topk(self.max_tracks_per_video, sorted=False)  # [M]
        track_class_ids = class_ids[topk_indices].tolist()  # [M]

        topk_track_indices = torch.div(topk_indices, num_classes, rounding_mode='floor')  # [M]
        track_ids = track_ids[topk_track_indices]  # [M]
        new_track_ids = torch.arange(1, track_ids.numel() + 1, dtype=torch.int64)
        track_id_duplication_mapping = torch.stack((track_ids, new_track_ids))  # [2, M]

        for t in range(len(seq_rle_masks)):
            new_rle_masks_t = dict()
            for track_id, mask in seq_rle_masks[t].items():
                duplicate_track_ids = track_id_duplication_mapping[1, track_id_duplication_mapping[0] == track_id].tolist()
                for new_track_id in duplicate_track_ids:
                    # print(f"{track_id} -> {new_track_id}")
                    new_rle_masks_t[new_track_id] = mask

            seq_rle_masks[t] = new_rle_masks_t

        new_track_ids = track_id_duplication_mapping[1].tolist()

        track_class_ids = {
            track_id: class_id for track_id, class_id in zip(new_track_ids, track_class_ids)
        }

        track_scores = {
            track_id: score for track_id, score in zip(new_track_ids, track_scores.tolist())
        }

        return seq_rle_masks, track_scores, track_class_ids

    def postprocess_tracks(self,
                           track_rle_masks: List[Dict[int, Any]],
                           track_scores: Dict[int, float],
                           class_ids: Dict[int, int]):
        """
        OVIS does not allow masks to overlap if they share the same class label.
        :param track_rle_masks: list of dicts (one per frame), with track IDs as keys and RLE encoded mask as values
        :param track_scores: Dict with track IDs as keys and normalized score in [0, 1] as value
        :param class_ids: Dict with track IDs as keys and class ID as value
        :return:
        """
        for t in range(len(track_rle_masks)):
            track_ids = list(track_rle_masks[t].keys())
            if len(track_ids) == 0:
                continue

            rle_masks = list(track_rle_masks[t].values())
            masks_t = torch.stack([self.rle_to_mask(rle_mask) for rle_mask in rle_masks]).cuda()  # [N, H, W]

            scores_t = torch.as_tensor([track_scores[t_id] for t_id in track_ids], dtype=torch.float32)  # [N]
            class_ids_t = torch.as_tensor([class_ids[t_id] for t_id in track_ids], dtype=torch.int64)  # [N]
            track_ids = torch.as_tensor(track_ids, dtype=torch.int64)  # [N]

            class_id_to_track_ids = defaultdict(list)
            for i, cls_id in enumerate(class_ids_t.tolist()):
                class_id_to_track_ids[cls_id].append(i)

            for cls_id, track_indices in class_id_to_track_ids.items():
                if len(track_indices) == 1:
                    continue

                masks_cls = masks_t[track_indices]  # [N', H, W]
                scores_cls = scores_t[track_indices]  # [N']
                track_ids_cls = track_ids[track_indices]  # [N']

                masks_cls = self.make_masks_nonoverlapping(masks_cls.unsqueeze(1), scores_cls).squeeze(1)  # [N', H, W]

                # encode masks and overwrite existing masks
                for track_id, mask in zip(track_ids_cls.tolist(), masks_cls):
                    assert track_id in track_rle_masks[t]  # sanity check
                    track_rle_masks[t][track_id] = self.mask_to_rle(mask)

        return track_rle_masks

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
