from typing import List, Dict, Tuple, Optional, Union, Any
from pycocotools import mask as masktools
from tarvis.data.file_packer import FilePackReader

import cv2
import json
import numpy as np
import os


def parse_generic_video_dataset(base_dir: str, dataset_json: Union[str, Dict[str, Any]]):
    if isinstance(dataset_json, str):
        with open(dataset_json, 'r') as fh:
            dataset = json.load(fh)
    else:
        dataset = dataset_json

    meta_info = dataset["meta"]

    # convert instance and category IDs from str to int
    if "category_labels" in meta_info:
        meta_info["category_labels"] = {int(k): v for k, v in meta_info["category_labels"].items()}

    for seq in dataset["sequences"]:
        seq["categories"] = {int(iid): cat_id for iid, cat_id in seq["categories"].items()}

        seq["segmentations"] = [
            {
                int(iid): seg
                for iid, seg in seg_t.items()
            }
            for seg_t in seq["segmentations"]
        ]

        # sanity check: instance IDs in "segmentations" must match those in "categories"
        seg_iids = set(sum([list(seg_t.keys()) for seg_t in seq["segmentations"]], []))
        assert seg_iids == set(seq["categories"].keys()), "Instance ID mismatch in seq {}: {} vs. {}".format(
            seq["id"], seg_iids, set(seq["categories"].keys())
        )

        if "semantic_segmentations" in seq:
            seq["semantic_segmentations"] = [
                {
                    int(cls_id): seg
                    for cls_id, seg in seg_t.items()
                }
                for seg_t in seq["semantic_segmentations"]
            ]

    seqs = [GenericVideoSequence(seq, base_dir) for seq in dataset["sequences"]]

    return seqs, meta_info


class GenericVideoSequence(object):
    def __init__(self, seq_dict, base_dir):
        assert len(seq_dict["image_paths"]) == len(seq_dict["segmentations"])

        self.base_dir = base_dir
        self.image_paths = seq_dict["image_paths"]
        self.image_dims = (seq_dict["height"], seq_dict["width"])
        self.id = seq_dict["id"]

        self.segmentations = seq_dict["segmentations"]
        self.instance_categories = seq_dict["categories"]
        self.semantic_segmentations = seq_dict.get("semantic_segmentations", None)
        self.instance_areas = None

        self.fpack_reader = None

    @property
    def height(self):
        return self.image_dims[0]

    @property
    def width(self):
        return self.image_dims[1]

    @property
    def instance_ids(self):
        return list(self.instance_categories.keys())

    @property
    def category_labels(self):
        return [self.instance_categories[instance_id] for instance_id in self.instance_ids]

    @property
    def has_semantic_masks(self):
        return self.semantic_segmentations is not None

    def _seg_to_rle(self, encoded_mask: Union[str, List[int]]):
        if isinstance(encoded_mask, list):  # polygons
            encoded_mask = {
                "counts": encoded_mask,
                "size": self.image_dims
            }
            encoded_mask = masktools.frPyObjects(encoded_mask, self.height, self.width)

        else:  # RLE mask
            assert isinstance(encoded_mask, str), f"Unexpected encoded mask type: {type(encoded_mask)}"
            encoded_mask = {
                "counts": encoded_mask.encode("utf-8"),
                "size": self.image_dims
            }

        return encoded_mask

    def _seg_to_mask(self, encoded_mask: Union[str, List[int], None]):
        if encoded_mask is None:
            return np.zeros(self.image_dims, np.uint8)

        rle = self._seg_to_rle(encoded_mask)
        return np.ascontiguousarray(masktools.decode(rle)).astype(np.uint8)

    def __len__(self):
        return len(self.image_paths)

    def load_images(self, frame_idxes=None):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        if self.base_dir.endswith(".fpack") and self.fpack_reader is None:
            self.fpack_reader = FilePackReader(self.base_dir, multiprocess_lock=False)

        images = []
        for t in frame_idxes:
            if self.fpack_reader is None:
                im = cv2.imread(os.path.join(self.base_dir, self.image_paths[t]), cv2.IMREAD_COLOR)
            else:
                im = self.fpack_reader.cv2_imread(self.image_paths[t], cv2.IMREAD_COLOR, True)

            if im is None:
                raise ValueError("No image found at path: {}".format(os.path.join(self.base_dir, self.image_paths[t])))
            images.append(im)

        return images

    def load_masks(self, frame_idxes=None, instance_ids=None, return_zero_masks_for_absent_instances=True):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        if instance_ids is None:
            instance_ids = self.instance_ids
        else:
            assert all([iid in self.instance_ids for iid in instance_ids])

        masks = []
        for t in frame_idxes:
            masks_t = []

            for instance_id in instance_ids:
                if instance_id in self.segmentations[t]:
                    masks_t.append(self._seg_to_mask(self.segmentations[t][instance_id]))

                elif return_zero_masks_for_absent_instances:
                    masks_t.append(np.zeros(self.image_dims, np.uint8))

            masks.append(masks_t)

        return masks

    def load_rle_masks(self, frame_idxes=None):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        masks = []
        for t in frame_idxes:
            masks_t = []

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    rle_mask = self._seg_to_rle(self.segmentations[t][instance_id])
                    masks_t.append(rle_mask)
                else:
                    masks_t.append(masktools.encode(np.asfortranarray(np.zeros(self.image_dims, np.uint8))))

            masks.append(masks_t)

        return masks

    def compute_instance_mask_areas(self):
        if self.instance_areas is not None:
            return self.instance_areas.copy()

        areas = []
        for t in range(len(self)):
            areas_t = []

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    rle_mask = self._seg_to_rle(self.segmentations[t][instance_id])
                    areas_t.append(masktools.area(rle_mask))
                else:
                    areas_t.append(0)

            areas.append(areas_t)

        self.instance_areas = areas.copy()

        return areas

    def load_semantic_masks(self, class_ids: List[int], frame_idxes=None, return_zero_masks_for_absent_instances=True):
        assert self.has_semantic_masks

        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        masks = []
        for t in frame_idxes:
            masks_t = []

            for cls_id in class_ids:
                if cls_id in self.semantic_segmentations[t]:
                    masks_t.append(self._seg_to_mask(self.semantic_segmentations[t][cls_id]))

                elif return_zero_masks_for_absent_instances:
                    masks_t.append(np.zeros(self.image_dims, np.uint8))

            masks.append(masks_t)

        return masks

    def extract_subsequence(self, frame_idxes, new_id=""):
        assert all([t in range(len(self)) for t in frame_idxes])

        subseq_dict = {
            "id": new_id if new_id else self.id,
            "height": self.image_dims[0],
            "width": self.image_dims[1],
            "image_paths": [self.image_paths[t] for t in frame_idxes]
        }

        instance_ids_to_keep = set(sum([list(self.segmentations[t].keys()) for t in frame_idxes], []))

        subseq_dict["segmentations"] = [
            {
                iid: segmentations_t[iid]
                for iid in segmentations_t if iid in instance_ids_to_keep
            }
            for t, segmentations_t in enumerate(self.segmentations) if t in frame_idxes
        ]

        if self.has_semantic_masks:
            subseq_dict["semantic_segmentations"] = [
                semantic_seg_t
                for t, semantic_seg_t in enumerate(self.semantic_segmentations) if t in frame_idxes
            ]

        subseq_dict["categories"] = {iid: self.instance_categories[iid] for iid in instance_ids_to_keep}
        return self.__class__(subseq_dict, self.base_dir)


def visualize_generic_dataset(base_dir, dataset_json):
    from tarvis.utils.visualization import overlay_mask_on_image, create_color_map

    seqs, meta_info = parse_generic_video_dataset(base_dir, dataset_json)
    category_names = meta_info["category_labels"]

    cmap = create_color_map().tolist()
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    for seq in seqs:
        if len(seq) > 100:
            frame_idxes = list(range(100, 150))
        else:
            frame_idxes = None

        images = seq.load_images(frame_idxes)
        masks = seq.load_masks(frame_idxes)
        category_labels = seq.category_labels

        print("[COLOR NAME] -> [CATEGORY NAME]")
        color_key_printed = False

        for image_t, masks_t in zip(images, masks):
            for i, (mask, cat_label) in enumerate(zip(masks_t, category_labels), 1):
                image_t = overlay_mask_on_image(image_t, mask, mask_color=cmap[i])
                # print("{} -> {}".format(rgb_to_name(cmap[i][::-1]), category_names[cat_label]))
                print("{} -> {}".format(cmap[i], category_names[cat_label]))

            color_key_printed = True

            cv2.imshow('Image', image_t)
            if cv2.waitKey(0) == 113:
                exit(0)
