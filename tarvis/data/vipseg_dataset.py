from torch import Tensor
from einops import rearrange
from typing import Dict, Any, List
from tqdm import tqdm
from tarvis.data.file_packer import FilePackReader
from tarvis.data.file_packer import utils as fpack_utils

from tarvis.config import cfg
from tarvis.data.training_dataset_base import TrainingDatasetBase
from tarvis.data.images_resizer import ImagesResizer
from tarvis.data.utils.video_cropping import compute_mask_containing_crop

import cv2
import imgaug.augmenters as iaa
import numpy as np
import os.path as osp
import json
import random
import torch


class VipSegDataset(TrainingDatasetBase):
    def __init__(self,
                 images_dir: str,
                 panoptic_masks_dir: str,
                 json_info_file: str,
                 clip_length: int,
                 num_samples: int,
                 frame_sampling_multiplicative_factor: float,
                 resize_params: Dict[str, Any]):

        super().__init__(name="VIPSEG", task_type="panoptic_seg")

        assert frame_sampling_multiplicative_factor >= 1

        self.images_dir = images_dir
        self.panoptic_masks_dir = panoptic_masks_dir

        self.image_fpack_reader = None
        self.panoptic_masks_fpack_reader = None
        assert osp.exists(images_dir), f"Images directory not found at {images_dir}"

        with open(json_info_file, 'r') as fh:
            self.video_info = json.load(fh)

        self.clip_length = clip_length
        self.resizer = ImagesResizer(**resize_params)
        self.num_samples = num_samples

        self.sampling_frame_range = int(round(frame_sampling_multiplicative_factor * clip_length))
        self.sampling_frame_shuffle = False

        self.feasible_samples = []
        self.max_retries = 5

        # remove videos with zero instances
        self.videos = [vid for vid in self.video_info["sequences"] if len(vid["instance_ids"]) > 0]

        self.color_augmenter = iaa.Sequential([
            iaa.AddToHueAndSaturation(value_hue=(-10, 10), value_saturation=(-10, 10)),
            iaa.LinearContrast(alpha=(0.95, 1.05)),
            iaa.AddToBrightness(add=(-20, 20))
        ])

        self.crop_factor = 0.75, 0.5

        self.sample_image_dims = [[1, 1] for _ in range(num_samples)]

        self.ignore_class_id = 0
        self.class_labels = {cat['id']: cat['name'] for cat in self.video_info["categories"]}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        vid_index = index % len(self.videos)
        video = self.videos[vid_index]

        # sample one frame which is guaranteed to have at least one instance
        frames_with_instances = [t for t in range(len(video["filenames"])) if len(video["frame_instance_occupancy"][t]) > 0]
        ref_frame = random.choice(frames_with_instances)

        start_idx = max(0, ref_frame - self.sampling_frame_range)
        end_idx = min(len(video["filenames"]), ref_frame + self.sampling_frame_range + 1)

        frame_indices = random.sample(
            list(range(start_idx, ref_frame)) + list(range(ref_frame + 1, end_idx)), self.clip_length - 1,
        )

        frame_indices = sorted(frame_indices + [ref_frame])
        if self.sampling_frame_shuffle:
            random.shuffle(frame_indices)

        n_retries = 0
        while n_retries < self.max_retries:
            ret = self.parse_sample(video, frame_indices)

            if ret is None:
                print("Failed to parse sample with instances")
                if len(self.feasible_samples) == 0:
                    raise RuntimeError("Sample generation failed and there are no backup feasible samples saved")
                video_index, frame_indices = random.choice(self.feasible_samples)
                video = self.videos[video_index]
                n_retries += 1
            else:
                self.feasible_samples.append((vid_index, frame_indices))
                if len(self.feasible_samples) > 5000:
                    self.feasible_samples.pop()

                return ret

        raise RuntimeError(f"No feasible sample could be found even after {self.max_retries} retries")

    def parse_sample(self, video: Dict[str, Any], frame_indices: List[int]):
        # clip = video.extract_subsequence(frame_indices)
        images, panoptic_masks = self.load_images_and_masks(video, frame_indices)

        # Check if the sampled clip
        # if len(clip.instance_ids) == 0:
        #     return None

        # images = clip.load_images()
        images = self.apply_color_augmentation(images)
        images = np.stack(images)  # [T, H, W, 3]
        panoptic_masks = np.stack(panoptic_masks)  # [T, H, W]

        instance_masks = []
        semantic_masks = [np.zeros_like(panoptic_masks, bool) for _ in range(124)]
        instance_categories = []
        ignore_mask = np.ones_like(panoptic_masks, np.uint8)

        for inst_id in video["instance_ids"]:
            inst_mask = panoptic_masks == inst_id
            if not np.any(inst_mask):
                continue

            instance_masks.append(inst_mask)
            class_id = (inst_id // 100) - 1
            instance_categories.append(class_id + 1)   # this ID is in 1-based index
            semantic_masks[class_id] = np.where(inst_mask, True, semantic_masks[class_id])
            ignore_mask = np.where(inst_mask, False, ignore_mask)

        instance_masks = np.stack(instance_masks)  # [N, T, H, W]

        for cls_id in video["stuff_classes"]:
            cls_mask = panoptic_masks == (cls_id + 1)
            if not np.any(cls_mask):
                continue

            semantic_masks[cls_id] = cls_mask
            ignore_mask = np.where(cls_mask, 0, ignore_mask)

        semantic_masks = np.stack(semantic_masks + [ignore_mask])  # [num_classes+1, T, H, W]

        # apply random cropping, reversal and horizontal flip
        images, instance_masks, semantic_masks = self.apply_random_horizontal_flip(images, instance_masks, semantic_masks)
        images, instance_masks, semantic_masks = self.apply_random_crop(images, instance_masks, semantic_masks)

        categories = torch.as_tensor(instance_categories, dtype=torch.int64)

        meta_info = {
            "orig_dims": images[0].shape[:2],
            "seq_name": video["name"],
            "frame_indices": frame_indices
        }

        images, (instance_masks, semantic_masks) = self.resizer(images=images, masks=[instance_masks, semantic_masks])

        # remove masks that are all-zeros after resizing
        valid_instance_ids = [i for i in range(instance_masks.shape[0]) if np.any(instance_masks[i])]
        if len(valid_instance_ids) == 0:
            return None

        instance_masks = instance_masks[valid_instance_ids]
        categories = categories[valid_instance_ids]

        # convert to torch array
        images = torch.from_numpy(np.ascontiguousarray(images)).float()
        instance_masks = torch.from_numpy(instance_masks).bool()  # [N, T, H, W]
        semantic_masks = torch.from_numpy(semantic_masks).bool()  # [num_classes, T, H, W]

        meta_info["semantic_category_labels"] = self.class_labels
        semantic_masks, ignore_masks = semantic_masks[:-1], semantic_masks[-1]

        # condense semantic masks
        semantic_masks = self.condense_semantic_masks(semantic_masks)

        category_labels = [self.class_labels[cat_id - 1] for cat_id in categories.tolist()]

        images = rearrange(images.float(), "T H W C -> T C H W")
        if cfg.INPUT.RGB:
            images = images.flip([1])

        meta_info["category_labels"] = category_labels

        return {
            "images": images,
            "instance_masks": instance_masks,
            "semantic_masks": semantic_masks,
            "ignore_masks": ignore_masks,
            "class_ids": categories,
            "dataset": self.name,
            "task_type": self.task_type,
            "meta": meta_info
        }

    def load_images_and_masks(self, video: Dict[str, Any], frame_indices: List[int]):
        if self.images_dir.endswith(".fpack") and self.image_fpack_reader is None:
            self.image_fpack_reader = FilePackReader(self.images_dir, base_path=osp.dirname(self.images_dir))
            self.images_dir = osp.dirname(self.images_dir)

        if self.panoptic_masks_dir.endswith(".fpack") and self.panoptic_masks_fpack_reader is None:
            self.panoptic_masks_fpack_reader = FilePackReader(self.panoptic_masks_dir, base_path=osp.dirname(self.panoptic_masks_dir))
            self.panoptic_masks_dir = osp.dirname(self.panoptic_masks_dir)

        images, panoptic_masks = [], []
        for t in frame_indices:
            image_path_t = osp.join(self.images_dir, video['name'], video["filenames"][t] + ".jpg")
            images.append(fpack_utils.cv2_imread(image_path_t, self.image_fpack_reader, cv2.IMREAD_COLOR))

            mask_path_t = osp.join(self.panoptic_masks_dir, video['name'], video["filenames"][t] + ".png")
            panoptic_masks.append(fpack_utils.cv2_imread(mask_path_t, self.panoptic_masks_fpack_reader, cv2.IMREAD_UNCHANGED))            

        return images, panoptic_masks

    def apply_random_crop(self, images: np.ndarray, masks: np.ndarray, semantic_masks: np.ndarray):
        # images: [T, H, W, 3], masks: [N, T, H, W], semantic_masks: [N, T, H, W]
        assert images.ndim == 4 and masks.ndim == 4

        crop_height, crop_width = [int(round(float(x) * cf)) for x, cf in zip(masks.shape[-2:], self.crop_factor)]
        crop_params = compute_mask_containing_crop(np.any(masks, 0), (crop_height, crop_width))
        if crop_params is None:
            return images, masks, semantic_masks

        crop_x1, crop_y1 = crop_params
        crop_x2, crop_y2 = crop_x1 + crop_width, crop_y1 + crop_height

        images = images[:, crop_y1:crop_y2, crop_x1:crop_x2]
        masks = masks[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
        semantic_masks = semantic_masks[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

        return images, masks, semantic_masks

    def apply_random_horizontal_flip(self, images: np.ndarray, masks: np.ndarray, semantic_masks: np.ndarray):
        # images: [T, H, W, 3], masks: [N, T, H, W]
        assert images.ndim == 4 and masks.ndim == 4

        if torch.rand(1) < 0.5:
            images = np.flip(images, 2)
            masks = np.flip(masks, 3)
            semantic_masks = np.flip(semantic_masks, 3)

        return images, masks, semantic_masks

    def apply_color_augmentation(self, images: List[np.ndarray]):
        # apply the same augmentation to all frames
        det_augmenter = self.color_augmenter.to_deterministic()
        return [det_augmenter(image=img) for img in images]

    def condense_semantic_masks(self, masks: Tensor):
        # masks: [num_classes, T, H, W]
        condensed_mask = torch.zeros(masks.shape[1:], dtype=torch.int64)
        for cls_id, cls_mask in enumerate(masks):
            condensed_mask = torch.where(cls_mask, cls_id, condensed_mask)
        return condensed_mask
