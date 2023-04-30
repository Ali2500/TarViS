from PIL import Image
from typing import Optional, List, Any, Dict, Tuple
from torch.utils.data import Dataset

from tarvis.config import cfg
from tarvis.data.images_resizer import ImagesResizer

import cv2
import numpy as np
import pycocotools.mask as mt
import os.path as osp
import torch


class InferenceDataset(Dataset):
    def __init__(self,
                 task_type: str,
                 image_paths: List[str],
                 clip_length: int,
                 overlap_length: int,
                 image_resize_params: Dict[str, Any],
                 first_frame_mask_paths: Optional[List[str]] = None,
                 first_frame_mask_rles: Optional[List[Dict[int, bytes]]] = None,
                 first_frame_object_points: Optional[List[Dict[int, Tuple[int, int]]]] = None,
                 first_ref_mask_frame_index: int = 0):

        super().__init__()
        assert task_type in ("instance_seg", "panoptic_seg", "vos")

        self.task_type = task_type
        self.image_paths = image_paths
        self.clip_length = clip_length

        assert overlap_length < clip_length
        assert first_ref_mask_frame_index < len(image_paths), \
            f"{first_ref_mask_frame_index} should be less than {len(image_paths)}"

        self.overlap_length = overlap_length
        video_length = len(image_paths)

        self.first_frame_mask_paths: Dict[int, str] = dict()
        self.first_frame_object_points: Dict[int, Dict[int, Tuple[int, int]]] = dict()
        self.first_frame_mask_rles: Dict[int, Dict[int, bytes]] = dict()

        if task_type == "vos":
            assert first_frame_mask_paths is not None or first_frame_object_points is not None \
                   or first_frame_mask_rles is not None

            if first_frame_mask_paths:
                self.parse_first_frame_masks(image_paths, first_frame_mask_paths)

            elif first_frame_mask_rles:
                self.first_frame_mask_rles = {
                    t: mask_rles_t for t, mask_rles_t in enumerate(first_frame_mask_rles)
                    if len(mask_rles_t) > 0
                }

            else:
                self.first_frame_object_points = {
                    t: object_points_t for t, object_points_t in enumerate(first_frame_object_points)
                    if len(object_points_t) > 0
                }

        # create clips
        self.clip_frame_indices = []
        for t in range(first_ref_mask_frame_index, video_length, clip_length - overlap_length):
            start_t = t
            end_t = min(start_t + clip_length, len(self.image_paths))

            indices = list(range(start_t, end_t))
            self.clip_frame_indices.append(indices)
            if end_t == video_length:
                break

        self.resizer = ImagesResizer(**image_resize_params)
        assert self.resizer.mode != "crop"  # resize should be deterministic during inference

    @property
    def is_vos_dataset(self):
        return len(self.first_frame_mask_paths) > 0 or len(self.first_frame_mask_rles) > 0

    @property
    def is_point_vos_dataset(self):
        return len(self.first_frame_object_points) > 0

    @property
    def is_vis_vps_dataset(self):
        return (not self.is_vos_dataset) and (not self.is_point_vos_dataset)

    def __len__(self):
        return len(self.clip_frame_indices)

    def __getitem__(self, index):
        frame_indices = self.clip_frame_indices[index]
        image_paths = [self.image_paths[t] for t in frame_indices]
        images = [cv2.imread(p, cv2.IMREAD_COLOR) for p in image_paths]
        assert all([img is not None for img in images]), f"One or more image files do not exist: {image_paths}"

        orig_height, orig_width = images[0].shape[:2]
        vos_ref_mask_info = dict()

        if self.task_type == "vos":
            vos_ref_mask_info = self.prepare_vos_inputs(frame_indices, orig_height, orig_width)

        mask_list = vos_ref_mask_info.get("masks", None)
        resized_output = self.resizer(images=np.stack(images, 0), masks=mask_list)  # [T, H, W, 3]

        if mask_list is None:
            images = resized_output

        else:
            images, mask_list = resized_output
            mask_list = np.squeeze(mask_list, 1)  # [N, H, W]
            assert len(mask_list) == len(vos_ref_mask_info["instance_ids"])
            vos_ref_mask_info["masks"] = torch.from_numpy(np.ascontiguousarray(mask_list))  # [N, H, W]

        images = torch.from_numpy(images).float().permute(0, 3, 1, 2)  # [T, 3, H, W]

        if cfg.INPUT.RGB:
            images = images.flip([1])

        if not vos_ref_mask_info:
            vos_ref_mask_info = None

        return {
            "images": images,
            "frame_indices": torch.as_tensor(frame_indices, dtype=torch.long),
            "orig_image_size": (orig_height, orig_width),
            "task_type": self.task_type,
            "vos_ref_mask_info": vos_ref_mask_info
        }

    def prepare_vos_inputs(self, frame_indices, img_height, img_width):
        mask_list = []
        object_points = []
        instance_id_list = []
        ref_frame_index_list = []

        for t in frame_indices:
            if self.is_vos_dataset:
                if t not in self.first_frame_mask_paths and t not in self.first_frame_mask_rles:
                    continue

                if t in self.first_frame_mask_paths:
                    mask = np.array(Image.open(self.first_frame_mask_paths[t]))
                    instance_ids = np.unique(mask)
                    instance_ids = instance_ids[instance_ids > 0]
                    assert len(instance_ids) > 0, f"No instances found in mask: {self.first_frame_mask_paths[t]}"

                    for inst_id in instance_ids.tolist():
                        instance_mask = (mask == inst_id).astype(np.uint8)  # [H, W]
                        mask_list.append(instance_mask)
                        instance_id_list.append(inst_id)
                        ref_frame_index_list.append(t)

                else:
                    assert t in self.first_frame_mask_rles
                    instance_ids_t = list(self.first_frame_mask_rles[t].keys())
                    rles_t = [self.first_frame_mask_rles[t][inst_id] for inst_id in instance_ids_t]

                    instance_id_list.extend(instance_ids_t)
                    ref_frame_index_list.extend([t for _ in range(len(instance_ids_t))])
                    mask_list.extend([self.parse_rle_mask(rle, (img_height, img_width)) for rle in rles_t])

            if self.is_point_vos_dataset:
                if t not in self.first_frame_object_points:
                    continue

                for inst_id, inst_point_coords in self.first_frame_object_points[t].items():
                    object_points.append(inst_point_coords)
                    instance_id_list.append(inst_id)
                    ref_frame_index_list.append(t)

        vos_ref_mask_info = dict()
        if mask_list or object_points:
            vos_ref_mask_info = {
                "instance_ids": torch.tensor(instance_id_list, dtype=torch.int64),
                "frame_indices": torch.tensor(ref_frame_index_list, dtype=torch.int64),
            }

        if mask_list:
            vos_ref_mask_info["masks"] = np.stack(mask_list, 0)[:, None]  # [N, 1, H, W]

        elif object_points:
            vos_ref_mask_info["point_coords"] = torch.tensor(object_points, dtype=torch.float32)

        return vos_ref_mask_info

    def parse_first_frame_masks(self, image_paths: List[str], mask_paths: List[str]):
        # for VOS task (DAVIS, YouTube-VOS)
        filename_to_frame_index_mapping = {
            osp.split(p)[-1].replace(".jpg", ""): t for t, p in enumerate(image_paths)
        }

        for p in mask_paths:
            filename = osp.split(p)[-1].replace(".png", "")
            self.first_frame_mask_paths[filename_to_frame_index_mapping[filename]] = p

    def parse_rle_mask(self, rle, img_dims):
        return mt.decode({
            "counts": rle.encode("utf-8"),
            "size": img_dims
        }).astype(np.uint8)
