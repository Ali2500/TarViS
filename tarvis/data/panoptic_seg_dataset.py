from torch import Tensor
from einops import rearrange
from typing import Dict, Any, List, Callable, Union

from tarvis.data.video_dataset_parser import GenericVideoSequence, parse_generic_video_dataset
from tarvis.config import cfg
from tarvis.data.training_dataset_base import TrainingDatasetBase
from tarvis.data.images_resizer import ImagesResizer
from tarvis.data.utils.video_cropping import compute_mask_containing_crop

import imgaug.augmenters as iaa
import numpy as np
import os.path as osp
import json
import random
import torch


class PanopticSegDataset(TrainingDatasetBase):
    def __init__(self,
                 name: str,
                 images_dir: str,
                 annotations_json_path: str,
                 annotations_mapper_fn: Union[Callable, None],
                 clip_length: int,
                 num_samples: int,
                 frame_sampling_multiplicative_factor: float,
                 resize_params: Dict[str, Any],
                 ignore_class_id: int,
                 instance_class_id_mapping: Dict[int, int]):

        super().__init__(name=name, task_type="panoptic_seg")

        assert frame_sampling_multiplicative_factor >= 1

        # `images_dir` could be an fpack file
        if not osp.exists(images_dir):
            images_dir = f"{images_dir}.fpack"
        assert osp.exists(images_dir), f"Images directory not found at {images_dir}"

        if annotations_mapper_fn is None:
            with open(annotations_json_path, 'r') as fh:
                annotations_content = json.load(fh)
        else:
            annotations_content = annotations_mapper_fn(annotations_json_path)

        videos, meta_info = parse_generic_video_dataset(images_dir, annotations_content)
        self.meta = meta_info

        self.clip_length = clip_length
        self.resizer = ImagesResizer(**resize_params)
        self.num_samples = num_samples

        self.sample_consecutive_frames = "cityscapes" in name.lower()
        self.sampling_frame_range = int(round(frame_sampling_multiplicative_factor * clip_length))
        self.sampling_frame_shuffle = False

        self.feasible_samples = []
        self.max_retries = 5

        # remove videos with zero instances
        self.videos = [vid for vid in videos if len(vid.instance_ids) > 0]

        self.color_augmenter = iaa.Sequential([
            iaa.AddToHueAndSaturation(value_hue=(-10, 10), value_saturation=(-10, 10)),
            iaa.LinearContrast(alpha=(0.95, 1.05)),
            iaa.AddToBrightness(add=(-20, 20))
        ])

        if "kitti" in name.lower():  # todo: this is hacky. make it part of the config.
            self.crop_factor = 0.75, 0.5
        else:
            assert "cityscapes" in name.lower()
            self.crop_factor = 0.5, 0.5

        self.sample_image_dims = [[1, 1] for _ in range(num_samples)]

        self.instance_class_id_mapping = instance_class_id_mapping
        self.ignore_class_id = ignore_class_id
        self.class_labels = meta_info['category_labels']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        vid_index = index % len(self.videos)
        video = self.videos[vid_index]

        if self.sample_consecutive_frames:
            start_frame = random.randint(0, len(video) - self.clip_length)
            frame_indices = list(range(0, len(video)))[start_frame:start_frame + self.clip_length]
        else:
            # sample one frame which is guaranteed to have at least one instance
            frames_with_instances = [t for t in range(len(video)) if len(video.segmentations[t]) > 0]
            ref_frame = random.choice(frames_with_instances)

            start_idx = max(0, ref_frame - self.sampling_frame_range)
            end_idx = min(len(video), ref_frame + self.sampling_frame_range + 1)

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
                if len(self.feasible_samples) > 1000:
                    self.feasible_samples.pop()

                return ret

        raise RuntimeError(f"No feasible sample could be found even after {self.max_retries} retries")

    def parse_sample(self, video: GenericVideoSequence, frame_indices: List[int]):
        clip = video.extract_subsequence(frame_indices)

        # Check if the sampled clip
        if len(clip.instance_ids) == 0:
            return None

        images = clip.load_images()
        images = self.apply_color_augmentation(images)
        images = np.stack(images)  # [T, H, W, 3]
        instance_masks = clip.load_masks()  # List[List[np.ndarray]]

        instance_masks = np.stack([
            np.stack(masks_t) for masks_t in instance_masks
        ])

        instance_masks = np.transpose(instance_masks, (1, 0, 2, 3))  # [N, T, H, W]
        # print(f"Has semantic masks: {clip.has_semantic_masks}")
        assert clip.has_semantic_masks

        all_class_ids = list(sorted(self.instance_class_id_mapping.keys())) + [self.ignore_class_id]
        semantic_masks = clip.load_semantic_masks(all_class_ids)  # [T, num_classes, H, W]
        semantic_masks = np.stack(semantic_masks).transpose(1, 0, 2, 3)  # [num_classes, T, H, W]

        # apply random cropping, reversal and horizontal flip
        images, instance_masks, semantic_masks = self.apply_random_horizontal_flip(images, instance_masks, semantic_masks)
        images, instance_masks, semantic_masks = self.apply_random_crop(images, instance_masks, semantic_masks)

        categories = torch.as_tensor(clip.category_labels, dtype=torch.int64)

        meta_info = {
            "orig_dims": images[0].shape[:2],
            "seq_name": video.id,
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

        category_labels = [self.meta["category_labels"][cat_id] for cat_id in categories.tolist()]

        images = rearrange(images.float(), "T H W C -> T C H W")
        if cfg.INPUT.RGB:
            images = images.flip([1])

        meta_info["category_labels"] = category_labels

        # apply class ID mapping to instance IDs
        categories = torch.as_tensor(
            [self.instance_class_id_mapping[cls_id] for cls_id in categories.tolist()], dtype=torch.int64
        )

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
