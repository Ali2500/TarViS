from typing import Dict, Any

from tarvis.utils.visualization import annotate_image_instance, create_color_map

import cv2
import multiprocessing as mp
import numpy as np
import os
import os.path as osp
import pycocotools.mask as mt
import torch


_MP_VARS = {
    "task_type": "",
    "output_dir": "",
    "track_class_ids": None,
    "track_scores": None,
    "class_labels": None,
    "color_map": create_color_map().tolist()
}


class SequenceVisualizer:
    def __init__(self,
                 task_type: str,
                 sequence_results: Dict[str, Any],
                 sequence_info: Dict[str, Any],
                 category_labels: Dict[int, str],
                 num_processes: int = 8
                 ):
        global _MP_VARS

        self.image_paths = sequence_info["image_paths"]
        self.track_rles = sequence_results["track_mask_rles"]
        self.semantic_masks = sequence_results.get("semantic_seg_masks", [None] * len(self.track_rles))
        self.num_processes = num_processes

        _MP_VARS["task_type"] = task_type
        _MP_VARS["track_class_ids"] = sequence_results.get("track_category_ids", None)
        _MP_VARS["track_scores"] = sequence_results.get("track_scores", None)

        if _MP_VARS["track_class_ids"] is not None:
            assert category_labels is not None
            _MP_VARS["class_labels"] = category_labels

        # multiprocessing Pool cannot handle CUDA tensors
        self.to_cpu(_MP_VARS)

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        _MP_VARS["output_dir"] = output_dir

        worker_fn_inputs = [
            self.to_cpu({
                "image_path": image_path_t,
                "track_rles": track_rles_t,
                "semantic_mask": semantic_mask_t
            })
            for image_path_t, track_rles_t, semantic_mask_t in
            zip(self.image_paths, self.track_rles, self.semantic_masks)
        ]

        if self.num_processes > 0:
            with mp.Pool(self.num_processes) as p:
                p.map(process_image_mask_pair, worker_fn_inputs)
        else:
            for inputs in worker_fn_inputs:
                process_image_mask_pair(inputs)

    @classmethod
    def to_cpu(cls, d):
        for k in d.keys():
            if torch.is_tensor(d[k]):
                d[k] = d[k].cpu()

        return d


def process_image_mask_pair(inputs):
    track_rles = inputs["track_rles"]
    image_path = inputs["image_path"]
    semantic_mask = inputs["semantic_mask"]  # [H, W] int64
    cmap = _MP_VARS["color_map"]

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    instance_image = np.copy(image)

    # process tracks in ascending order of score because masks are not guaranteed to be non-overlapping!
    if _MP_VARS["task_type"] == "instance_seg":
        sorted_track_ids = sorted(track_rles.keys(), key=lambda track_id: _MP_VARS["track_scores"][track_id])
    else:
        sorted_track_ids = list(track_rles.keys())

    for track_id in sorted(sorted_track_ids):
        rle = track_rles[track_id]
        mask = mt.decode(rle).astype(np.uint8)

        if _MP_VARS["task_type"] == "instance_seg":
            score = _MP_VARS["track_scores"][track_id]
            class_label = _MP_VARS["class_labels"][_MP_VARS["track_class_ids"][track_id]]
            text_label = f"{class_label};{int(round(score * 100.))}"
            bbox = "mask"

        elif _MP_VARS["task_type"] == "vos":
            text_label, bbox = None, None

        else:
            raise ValueError("Should not be here.")

        instance_image = annotate_image_instance(instance_image, mask, cmap[track_id],
                                                 label=text_label,
                                                 bbox=bbox,
                                                 mask_opacity=0.5,
                                                 text_placement="mask_centroid",
                                                 text_color=cmap[track_id])

    img_fname = osp.split(image_path)[-1]
    cv2.imwrite(osp.join(_MP_VARS["output_dir"], img_fname), instance_image)

    if not semantic_mask:
        return

    semantic_image = np.copy(image)
    class_ids = np.unique(semantic_mask)
    class_ids = class_ids[class_ids > 0]  # remove background

    for cls_id in class_ids.tolist():
        class_label = _MP_VARS["class_labels"][cls_id]
        mask = (semantic_mask == cls_id).astype(np.uint8)
        semantic_image = annotate_image_instance(semantic_image, mask, cmap[cls_id],
                                                 label=class_label,
                                                 bbox='mask',
                                                 mask_opacity=0.5)

    cv2.imwrite(osp.join(_MP_VARS["output_dir"], f"semantic_seg_{img_fname}"), semantic_image)
