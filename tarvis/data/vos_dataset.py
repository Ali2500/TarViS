from einops import rearrange
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
from tarvis.data.video_dataset_parser import GenericVideoSequence, parse_generic_video_dataset

from tarvis.config import cfg
from tarvis.data.training_dataset_base import TrainingDatasetBase
from tarvis.data.images_resizer import ImagesResizer
from tarvis.data.utils.image_cropping import compute_mask_preserving_crop

import cv2
import imgaug.augmenters as iaa
import math
import json
import numpy as np
import os.path as osp
import random
import torch
import torch.nn.functional as F


class VOSDataset(TrainingDatasetBase):
    def __init__(self, name: str, images_dir: str, annotations_json_path: str, clip_length: int, num_samples: int,
                 frame_sampling_multiplicative_factor: float, resize_params: Dict[str, Any], max_num_instances: int,
                 annotations_mapper_fn: Optional[Callable] = None):
        super().__init__(name=name, task_type="vos")

        assert max_num_instances >= 1
        assert frame_sampling_multiplicative_factor >= 1

        # `images_dir` could be an fpack file
        if not osp.exists(images_dir):
            images_dir = f"{images_dir}.fpack"
            assert osp.exists(images_dir), f"Directory not found: {images_dir}"

        if annotations_mapper_fn is None:
            with open(annotations_json_path, 'r') as fh:
                annotations_content = json.load(fh)
        else:
            annotations_content = annotations_mapper_fn(annotations_json_path)

        videos, meta_info = parse_generic_video_dataset(images_dir, annotations_content)
        self.meta = meta_info

        self.clip_length = clip_length
        self.resizer = ImagesResizer(**resize_params)

        self.samples, self.sample_image_dims, self.sample_instance_counts = self.create_training_samples(
            videos, num_samples, frame_sampling_multiplicative_factor, max_num_instances
        )

        self.videos: Dict[str, GenericVideoSequence] = {vid.id: vid for vid in videos}

        self.color_augmenter = iaa.Sequential([
            iaa.AddToHueAndSaturation(value_hue=(-12, 12), value_saturation=(-12, 12)),
            iaa.LinearContrast(alpha=(0.95, 1.05)),
            iaa.AddToBrightness(add=(-25, 25))
        ])

        self.crop_factor = 0.75

        # store fallback candidates
        self.fallback_candidates = defaultdict(set)
        for i, num_instances in enumerate(self.sample_instance_counts):
            self.fallback_candidates[num_instances].add(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        n_tries = 0
        while True:
            sample = self.parse_sample(index)
            if sample is not None:
                return sample

            num_instances = self.sample_instance_counts[index]
            self.fallback_candidates[num_instances].discard(index)
            index = random.choice(list(self.fallback_candidates[num_instances]))
            n_tries += 1
            if n_tries % 3 == 0:
                print(f"Num failed tries = {n_tries} for dataset {self.name}, num_instances {num_instances}")

    def parse_sample(self, index):
        sample = self.samples[index]
        video = self.videos[sample['vid_id']]

        frame_indices = [sample["ref_frame"]] + sample["other_frames"]
        clip = video.extract_subsequence(frame_indices)

        images = clip.load_images()
        images = self.apply_color_augmentation(images)
        images = np.stack(images)  # [T, H, W, 3]
        masks = clip.load_masks(instance_ids=sample["ref_inst_ids"])  # List[List[np.ndarray]]
        masks = np.stack([
            np.stack(masks_t) for masks_t in masks
        ])

        masks = np.transpose(masks, (1, 0, 2, 3))  # [N, T, H, W]
        orig_masks = np.copy(masks)

        meta_info = {
            "orig_dims": images[0].shape[:2],
            "seq_name": video.id,
            "frame_indices": frame_indices
        }

        images, masks = self.resizer(images=images, masks=masks, ref_frame_index=0)

        # apply random cropping and horizontal flip
        images, masks = self.apply_random_horizontal_flip(images, masks)
        images, masks = self.apply_random_crop(images, masks, ref_frame_index=0)

        # convert to torch array
        images = torch.from_numpy(np.copy(images))
        masks = torch.from_numpy(np.ascontiguousarray(masks)).bool()  # [N, T, H, W]

        if not torch.all(torch.any(masks[:, 0].flatten(1), 1)):
            # orig_masks = np.reshape(orig_masks[:, 0], (orig_masks.shape[0], -1))
            # print(np.sum(orig_masks, 1))
            # print(meta_info["orig_dims"])
            # print(f"{torch.any(masks[:, 0].flatten(1), 1).tolist()}")
            return None

        images = rearrange(images.float(), "T H W C -> T C H W")
        if cfg.INPUT.RGB:
            images = images.flip([1])

        return {
            "images": images,
            "instance_masks": masks,
            "semantic_masks": None,
            "ref_frame_index": 0,
            "dataset": self.name,
            "task_type": self.task_type,
            "meta": meta_info
        }

    def apply_random_crop(self, images: np.ndarray, masks: np.ndarray, ref_frame_index: int):
        # images: [T, H, W, 3], masks: [N, T, H, W]
        assert images.ndim == 4 and masks.ndim == 4

        crop_height, crop_width = [int(round(float(x) * self.crop_factor)) for x in masks.shape[-2:]]
        crop_params = compute_mask_preserving_crop(np.any(masks[:, ref_frame_index], 0), (crop_height, crop_width))
        if crop_params is None:
            return images, masks

        crop_x1, crop_y1 = crop_params
        crop_x2, crop_y2 = crop_x1 + crop_width, crop_y1 + crop_height
        return images[:, crop_y1:crop_y2, crop_x1:crop_x2], masks[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

    def apply_random_horizontal_flip(self, images: np.ndarray, masks: np.ndarray):
        # images: [T, H, W, 3], masks: [N, T, H, W]
        assert images.ndim == 4 and masks.ndim == 4

        if torch.rand(1) < 0.5:
            images = np.flip(images, 2)
            masks = np.flip(masks, 3)

        return images, masks

    def apply_color_augmentation(self, images: List[np.ndarray]):
        # apply the same augmentation to all frames
        det_augmenter = self.color_augmenter.to_deterministic()
        return [det_augmenter(image=img) for img in images]

    def create_training_samples(self, videos: List[GenericVideoSequence],
                                num_total_samples: int,
                                frame_sampling_multiplicative_factor: float,
                                max_num_instances: int):

        # fix seed so that same set of samples is generated across all processes
        rnd_state_backup = random.getstate()
        random.seed(2202)

        samples_by_num_instance = defaultdict(list)
        max_temporal_span = int(round(frame_sampling_multiplicative_factor * self.clip_length))

        for vid in videos:
            last_t = len(vid) - self.clip_length

            for t in range(last_t):
                valid_instance_ids = [iid for iid in vid.instance_ids if iid in vid.segmentations[t]]
                if not valid_instance_ids:
                    continue

                bin_id = min(max_num_instances, len(valid_instance_ids))
                samples_by_num_instance[bin_id].append((vid.id, t, valid_instance_ids))

        train_samples = []
        train_sample_dims = []
        train_sample_ni = []

        videos = {vid.id: vid for vid in videos}
        num_instances_per_count = int(math.ceil(num_total_samples / float(max_num_instances)))
        available_sample_pool = []

        for ni in range(max_num_instances, 0, -1):
            available_sample_pool = samples_by_num_instance[ni] + available_sample_pool

            for ii in range(num_instances_per_count):
                ii = ii % len(available_sample_pool)
                vid_id, ref_frame_idx, instance_ids = available_sample_pool[ii]

                assert len(instance_ids) >= ni

                if len(instance_ids) > ni:
                    instance_ids = random.sample(instance_ids, ni)

                # sample other frames
                vid = videos[vid_id]
                other_frames_list = list(
                    range(ref_frame_idx + 1, min(len(vid), ref_frame_idx + max_temporal_span))
                )

                assert len(other_frames_list) >= self.clip_length - 1, \
                    f"Something went wrong here: {ref_frame_idx}, {len(vid)}, {other_frames_list}"

                other_frame_idxes = sorted(random.sample(other_frames_list, self.clip_length - 1))

                train_samples.append({
                    "vid_id": vid_id,
                    "ref_frame": ref_frame_idx,
                    "other_frames": other_frame_idxes,
                    "ref_inst_ids": instance_ids
                })

                train_sample_dims.append((vid.height, vid.width))
                train_sample_ni.append(ni)

        # restore initial random state
        random.setstate(rnd_state_backup)

        train_samples = train_samples[:num_total_samples]
        train_sample_dims = train_sample_dims[:num_total_samples]
        train_sample_ni = train_sample_ni[:num_total_samples]

        return train_samples, train_sample_dims, train_sample_ni
