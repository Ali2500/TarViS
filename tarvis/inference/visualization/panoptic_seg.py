from typing import List, Dict, Tuple
from tqdm import tqdm

from tarvis.utils.visualization import annotate_image_instance, create_color_map

import cv2
import multiprocessing as mp
import numpy as np
import os
import os.path as osp


_MP_VARS = {
    "track_class_ids": None,
    "track_scores": None,
    "class_labels": None,
    "output_dir": None,
    "thing_class_ids": None,
    "color_map": create_color_map().tolist()
}


class SequenceVisualizer:
    def __init__(self,
                 sequence_results,
                 sequence_info,
                 category_labels,
                 num_processes):

        global _MP_VARS
        _MP_VARS["track_class_ids"] = sequence_results["track_category_ids"]
        _MP_VARS["track_scores"] = sequence_results["track_scores"]
        _MP_VARS["class_labels"] = category_labels
        _MP_VARS["thing_class_ids"] = sequence_info["thing_class_ids"]

        self.image_paths = sequence_info["image_paths"]
        self.panoptic_masks = sequence_results["panoptic_masks"]
        self.num_processes = num_processes

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        _MP_VARS["output_dir"] = output_dir

        worker_fn_inputs = [
            {
                "image_path": image_path_t,
                "panoptic_mask": panoptic_mask_t
            }
            for image_path_t, panoptic_mask_t in zip(self.image_paths, self.panoptic_masks)
        ]

        if self.num_processes > 0:
            with mp.Pool(self.num_processes) as p:
                _ = list(tqdm(
                    p.imap(process_image_mask_pair, worker_fn_inputs, chunksize=10),
                    total=len(worker_fn_inputs), leave=False, desc="Visualization"
                ))
                # p.map(process_image_mask_pair, worker_fn_inputs)
        else:
            for inputs in tqdm(worker_fn_inputs, leave=False, desc="Visualization"):
                process_image_mask_pair(inputs)


def process_image_mask_pair(inputs):
    panoptic_mask: np.ndarray = inputs["panoptic_mask"]
    image_path: List[str] = inputs["image_path"]
    cmap = _MP_VARS["color_map"]

    num_classes = len(_MP_VARS["class_labels"])
    cmap_semantic = cmap[:num_classes]
    cmap_instances = cmap[num_classes:]

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # panoptic_mask should be in BGR format with the following channel spec:
    # - R: semantic ID
    # - G: instance_id // 256
    # - B: instance_id % 256

    panoptic_mask = panoptic_mask.astype(np.int32)
    b, g, r = [panoptic_mask[:, :, i] for i in range(3)]
    instance_mask = (g * 256) + b
    semantic_mask = r

    semantic_ids = np.unique(semantic_mask)
    instance_fg_mask = (instance_mask > 0).astype(np.uint8)
    class_label_color_mapping = dict()

    for cls_id in semantic_ids:
        mask = np.where(instance_fg_mask, 0, (semantic_mask == cls_id).astype(np.uint8))
        if not np.any(mask):
            continue

        if cls_id not in _MP_VARS["thing_class_ids"]:
            class_label_color_mapping[_MP_VARS["class_labels"][cls_id]] = cmap[cls_id]

        image = annotate_image_instance(
            image, mask,
            color=cmap_semantic[cls_id],
            mask_opacity=0.7,
        )

    track_ids = np.unique(instance_mask)
    track_ids = track_ids[track_ids > 0]

    for track_id in track_ids.tolist():
        cls_id = _MP_VARS["track_class_ids"][track_id]
        cls_label = _MP_VARS["class_labels"][cls_id]
        score = _MP_VARS["track_scores"][track_id]
        mask_label = f"{cls_label};{int(round(score * 100.))}"

        mask = (instance_mask == track_id).astype(np.uint8)
        image = annotate_image_instance(
            image, mask,
            color=cmap_instances[track_id % len(cmap_instances)],
            label=mask_label,
            mask_opacity=0.7,
            text_placement='mask_centroid'
        )

    image = draw_semantic_color_legend(image, class_label_color_mapping)

    img_fname = osp.split(image_path)[-1].replace(".png", ".jpg")
    cv2.imwrite(osp.join(_MP_VARS["output_dir"], img_fname), image)


def draw_semantic_color_legend(image: np.ndarray, label_color_mapping: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    width = 150
    box_width = 40
    box_height = 12
    vert_margin = 6

    canvas = np.full((image.shape[0], width, 3), 255, np.uint8)
    y_offset = vert_margin
    x_margin = 5

    for label, color in label_color_mapping.items():
        canvas = cv2.rectangle(canvas, (x_margin, y_offset), (x_margin + box_width, y_offset + box_height), color,
                               thickness=-1)
        canvas = cv2.putText(canvas, label, (x_margin + box_width + x_margin, y_offset + box_height - 3),
                             cv2.FONT_HERSHEY_DUPLEX, fontScale=0.4, color=(0, 0, 0), lineType=cv2.LINE_AA)
        y_offset += box_height + vert_margin

    return np.concatenate((image, canvas), 1)
