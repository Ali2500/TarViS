from collections import defaultdict
from PIL import Image
from typing import Dict, Any, List
from panopticapi.utils import IdGenerator

from tarvis.inference.result_formatters.kitti_step import KITTISTEPResultFormatter
from tarvis.inference.result_formatters.meta.cityscapes_vps_categories import CATEGORIES

import cv2
import json
import numpy as np
import os
import os.path as osp


class CityscapesVPSResultFormatter(KITTISTEPResultFormatter):
    def __init__(self, output_dir, track_score_threshold: float = 0.5, max_tracks_per_video: int = 1000):
        super().__init__(
            output_dir=output_dir,
            track_score_threshold=track_score_threshold,
            max_tracks_per_video=max_tracks_per_video
        )

        # instance predictions are only possible for thing classes. These have IDs 11-18 in CityscapesVPS.
        # This is different from KITTI-STEP so we overwrite it.
        self.thing_classes = [11, 12, 13, 14, 15, 16, 17, 18]
        self.stuff_class_ids = [i for i in self.all_class_ids if i not in self.thing_classes]

        # load filename mapping because CityscapesVPS has a stupid format with different filenames in 'img_all' and
        # 'img'
        with open(osp.join(osp.dirname(__file__), "meta", "cityscapes_vps_val_filename_mapping.json")) as fh:
            self.filename_mapping = json.load(fh)

        self.json_annotations = []
        self.categories_dict = {cat['id']: cat for cat in CATEGORIES}

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

        cityscapes_vps_panoptic_maps, segm_annotations = self.to_cityscapes_vps_format(seq_panoptic_maps, seq_class_labels)
        assert len(cityscapes_vps_panoptic_maps) in (6, 30)  # sanity check. If using all frames: 30, if only eval frames: 6

        # write out panoptic maps as PNG images
        seq_output_dir_eval = osp.join(self.output_dir, "eval_frames", "pan_pred")
        seq_output_dir_all = osp.join(self.output_dir, "all_frames")

        os.makedirs(seq_output_dir_all, exist_ok=True)
        os.makedirs(seq_output_dir_eval, exist_ok=True)

        for panoptic_map, segm_anns, input_img_path in zip(
            cityscapes_vps_panoptic_maps, segm_annotations, sequence_info["image_paths"]
        ):
            filename = osp.split(input_img_path)[-1]
            panoptic_map = Image.fromarray(panoptic_map)
            panoptic_map.save(osp.join(seq_output_dir_all, filename))

            if filename in self.filename_mapping:
                panoptic_map.save(osp.join(seq_output_dir_eval, self.filename_mapping[filename]))
                self.json_annotations.append(segm_anns)

        return {
            "panoptic_masks": seq_panoptic_maps,
            "track_category_ids": seq_class_labels,
            "track_scores": seq_scores
        }

    def finalize_output(self):
        assert len(self.json_annotations) == len(self.filename_mapping), \
            f"{len(self.json_annotations)} =/= {len(self.filename_mapping)}"

        with open(osp.join(self.output_dir, "eval_frames", "pred.json"), 'w') as fh:
            json.dump({"annotations": self.json_annotations}, fh)

    def to_cityscapes_vps_format(self, panoptic_maps: List[np.ndarray], track_class_ids: Dict[int, int]):
        cvps_panoptic_maps = []

        track_id_to_color = dict()
        id_generator = IdGenerator(self.categories_dict)
        annotations = []

        for map_t in panoptic_maps:
            b, g, r = [map_t[:, :, i] for i in range(3)]
            # - R: semantic ID
            # - G: instance_id // 256
            # - B: instance_id % 256
            cvps_map_t = np.zeros(list(b.shape) + [3], np.uint8)
            segm_info = dict()

            semantic_map = r
            instance_map = (g * 256) + b

            for cls_id in self.stuff_class_ids:
                cls_mask = semantic_map == cls_id
                if not np.any(cls_mask):
                    continue

                segment_id, color = id_generator.get_id_and_color(cls_id)
                segment_id = int(segment_id)
                cvps_map_t = np.where(cls_mask[:, :, None], np.array(color, np.uint8)[None, None, :], cvps_map_t)
                area = np.sum(cls_mask.astype(np.uint32)).item()

                assert segment_id not in segm_info, f"Segment ID {segment_id} already exists"
                segm_info[segment_id] = {
                    "id": segment_id,
                    "category_id": int(cls_id),
                    "iscrowd": 0,
                    "area": area
                }

            for track_id, cls_id in track_class_ids.items():
                assert cls_id in self.thing_classes, f"cls_id = {cls_id}"  # sanity check

                inst_mask = instance_map == track_id
                if not np.any(inst_mask):
                    continue

                if track_id not in track_id_to_color:
                    segment_id, color = id_generator.get_id_and_color(cls_id)
                    track_id_to_color[track_id] = (segment_id, color)
                else:
                    segment_id, color = track_id_to_color[track_id]

                segment_id = int(segment_id)
                cvps_map_t = np.where(inst_mask[:, :, None], np.array(color, np.uint8)[None, None, :], cvps_map_t)
                area = np.sum(inst_mask.astype(np.uint32)).item()

                assert segment_id not in segm_info, f"Segment ID {segment_id} already exists"
                segm_info[segment_id] = {
                    "id": segment_id,
                    "category_id": int(cls_id),
                    "iscrowd": 0,
                    "area": area
                }

            cvps_panoptic_maps.append(cvps_map_t)

            # segment sanity check, area recalculation
            # gt_pan = cvps_map_t.astype(np.uint32)
            # pan_gt = gt_pan[:, :, 0] + (gt_pan[:, :, 1] * 256) + (gt_pan[:, :, 2] * 256 * 256)
            # labels, labels_cnt = np.unique(pan_gt, return_counts=True)
            # gt_labels = list(segm_info.keys())
            # gt_labels_set = set(gt_labels)
            #
            # for label, area in zip(labels, labels_cnt):
            #     if label == 0:
            #         continue
            #
            #     if label not in gt_labels and label > 0:
            #         print('png label not in json labels.')
            #     segm_info[label]["area"] = int(area)
            #     gt_labels_set.remove(label)
            #
            # if len(gt_labels_set) != 0:
            #     raise KeyError('remaining gt_labels json')

            segm_info = {"segments_info": [v for k, v in segm_info.items()]}
            annotations.append(segm_info)

        return cvps_panoptic_maps, annotations
