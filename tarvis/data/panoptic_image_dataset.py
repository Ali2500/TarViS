from collections import defaultdict
import detectron2.data.transforms as T
from torch.utils.data import Dataset
from fvcore.transforms import PadTransform
from typing import Dict, Any, List, Tuple

from tarvis.config import cfg
from tarvis.data.training_dataset_base import TrainingDatasetBase
from tarvis.data.images_resizer import ImagesResizer

import cv2
import json
from PIL import Image
import numpy as np
import imgaug
import imgaug.augmenters as iaa
import os.path as osp
import random
import torch


class NoValidInstancesException(RuntimeError):
    pass


def parse_json_annotations(filepath: str, is_cityscapes: bool):
    with open(filepath, 'r') as fh:
        content = json.load(fh)

    images_struct = dict()

    for img in content["images"]:
        filename = img["file_name"]
        if is_cityscapes:
            filename = filename.replace("gtFine_", "")
            cityname = filename.split("_")[0]
            filename = osp.join(cityname, filename)

        images_struct[img['id']] = {
            "file_name": filename,
            "height": img['height'],
            "width": img['width'],
            "segments": []
        }

    for ann in content["annotations"]:
        images_struct[ann['image_id']]["segments"] = ann["segments_info"]
        images_struct[ann['image_id']]["file_name_png"] = ann["file_name"]

    return images_struct


class PanopticImageDataset(TrainingDatasetBase):
    def __init__(self, images_base_dir: str, panoptic_maps_dir: str, parsed_image_samples: Dict[str, Dict[str, Any]],
                 clip_length: int, num_samples: int, categories_info: List[Dict[str, Any]], output_dims: Tuple[int],
                 dataset_name: str):
        super().__init__(
            name=dataset_name,
            task_type="panoptic_seg"
        )

        assert osp.exists(images_base_dir), f"Images directory not found: {images_base_dir}"
        assert osp.exists(panoptic_maps_dir), f"Panoptic maps directory not found: {panoptic_maps_dir}"

        self.images_base_dir = images_base_dir
        self.png_segments_dir = panoptic_maps_dir
        self.image_samples = parsed_image_samples
        self.image_ids = list(parsed_image_samples.keys())

        self.clip_length = clip_length
        self.num_samples = num_samples
        self.output_dims = output_dims

        self.color_augmenter = iaa.Sequential([
            iaa.AddToHueAndSaturation(value_hue=(-12, 12), value_saturation=(-12, 12)),
            iaa.LinearContrast(alpha=(0.95, 1.05)),
            iaa.AddToBrightness(add=(-25, 25))
        ])

        self.fallback_candidates = set(self.image_ids)
        self.sample_image_dims = [[1, 1] for _ in range(self.num_samples)]

        self.categories_info = categories_info
        self.stuff_category_ids = self.get_stuff_category_ids(categories_info)

        self.category_id_mapping = {
            category['id']: i for i, category in enumerate(categories_info)
        }
        self.category_names = {
            category['id']: category['name'] for category in categories_info
        }

        self.augmentations = [
            T.ResizeScale(min_scale=0.4, max_scale=2.0, target_height=output_dims[0], target_width=output_dims[1]),
            T.FixedSizeCrop(crop_size=output_dims)
        ]

        self.augmentation_affine = iaa.Affine(scale=(0.9, 1.1), rotate=(-20, 20), shear=(-10, 10))
        self.augmentation_fixed_crop = iaa.CropToFixedSize(self.output_dims[0], self.output_dims[1])
        self.augmentation_min_dims = [self.output_dims[0], self.output_dims[0] + 32, self.output_dims[0] + 64]
        self.fixed_crop_across_frames = False

        self.filter_zero_instance_images()

    def filter_zero_instance_images(self):
        stuff_classes = self.get_stuff_category_ids(self.categories_info)
        ignore_classes = self.get_ignore_category_ids()

        image_ids = self.image_ids.copy()
        n_removed = 0

        for img_idx in image_ids:
            struct = self.image_samples[img_idx]
            instance_found = False
            for ann_segment in struct["segments"]:
                cat_id = ann_segment["category_id"]
                if cat_id not in stuff_classes and cat_id not in ignore_classes:
                    instance_found = True
                    break

            if not instance_found:
                n_removed += 1
                self.image_ids.remove(img_idx)   

        self.image_samples = {
            img_id: self.image_samples[img_id] 
            for img_id in self.image_ids
        }
        self.fallback_candidates = set(self.image_ids)

    def get_stuff_category_ids(self, categories_info: List[Dict[str, Any]]) -> List[int]:
        pass

    def get_ignore_category_ids(self) -> List[int]:
        return []

    def preprocess_image_panmap(self, image: np.ndarray, panmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply pre-processing steps to the image and panoptic map
        :param image: [H, W, 3]
        :param panmap: [H, W, 3]
        :return:
        """
        return image, panmap

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        img_idx = self.image_ids[index % len(self.image_ids)]

        n_tries = 0
        while True:
            try:
                sample = self.parse_image(img_idx)
                self.fallback_candidates.add(img_idx)
                return sample

            except NoValidInstancesException as _:
                self.fallback_candidates.discard(img_idx)
                img_idx = random.choice(list(self.fallback_candidates))
                n_tries += 1
                if n_tries % 3 == 0:
                    print(f"Num failed tries: {n_tries}")

    def parse_image(self, img_idx: str):
        struct = self.image_samples[img_idx]

        image_path = osp.join(self.images_base_dir, struct["file_name"])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # [H, W, 3]
        if image is None:
            raise FileNotFoundError(f"{image_path} not found")

        if cfg.INPUT.RGB:
            image = image[:, :, ::-1]

        meta_info = {
            "orig_dims": image.shape[:2],
            "seq_name": img_idx,
            "frame_indices": [0 for _ in range(self.clip_length)]
        }

        assert image is not None
        png_mask = cv2.imread(osp.join(self.png_segments_dir, struct["file_name_png"]), cv2.IMREAD_COLOR)  # [H, W, 3] (BGR)
        image, png_mask = self.preprocess_image_panmap(image, png_mask)

        png_mask = png_mask.astype(np.int64)
        png_mask = (256 ** 2 * png_mask[:, :, 0]) + (256 * png_mask[:, :, 1]) + png_mask[:, :, 2]

        # apply color augmentations to image
        image = self.color_augmenter(image=image)

        # apply random horizontal flip
        if random.random() < 0.5:
            image = np.flip(image, 1)
            png_mask = np.flip(png_mask, 1)

        instance_masks = []
        instance_categories = []

        semantic_masks = defaultdict(list)
        ignore_category_ids = set(self.get_ignore_category_ids())

        for ann_segment in struct["segments"]:
            if ann_segment["category_id"] in ignore_category_ids:
                continue

            m = (png_mask == ann_segment["id"]).astype(np.uint8)
            semantic_masks[ann_segment["category_id"]].append(m)

            if ann_segment["category_id"] not in self.stuff_category_ids:
                instance_masks.append(m)
                instance_categories.append(ann_segment["category_id"])

        semantic_categories = list(semantic_masks.keys())
        semantic_masks = np.stack([
            np.any(np.stack(semantic_masks[cat_id], 0), 0)
            for cat_id in semantic_categories
        ])  # [C', H, W]

        if len(instance_masks) == 0:
            print(f"Image has no instances!")
            raise NoValidInstancesException()

        instance_masks = np.stack(instance_masks)  # [N, H, W]
        instance_categories = np.array(instance_categories, np.int64)  # [N]

        seq_images, seq_instance_masks, instance_categories, seq_semantic_masks, seq_ignore_masks = \
            self.augment_sample(image, instance_masks, instance_categories, semantic_masks)

        # assign final, mapped category IDs to semantic masks. -100 is the ignore class label used by F.cross_entropy
        mapped_seq_semantic_masks = np.full_like(seq_semantic_masks, -100, dtype=np.int64)
        semantic_category_labels = dict()

        for i, cat_id in enumerate(semantic_categories):
            mapped_cat_id = self.category_id_mapping[cat_id]
            semantic_category_labels[mapped_cat_id] = self.category_names[cat_id]
            mapped_seq_semantic_masks = np.where(seq_semantic_masks == i, mapped_cat_id, mapped_seq_semantic_masks)

        meta_info["category_labels"] = [self.category_names[cat_id] for cat_id in instance_categories.tolist()]
        meta_info["semantic_category_labels"] = semantic_category_labels

        # convert everything to torch tensors
        seq_images = torch.from_numpy(seq_images).permute(0, 3, 1, 2).to(torch.float32)  # [T, 3, H, W]
        seq_instance_masks = torch.from_numpy(seq_instance_masks).to(torch.bool)  # [N, T, H, W]
        mapped_seq_semantic_masks = torch.from_numpy(mapped_seq_semantic_masks).to(torch.int64)  # [T, H, W]
        seq_ignore_masks = torch.from_numpy(seq_ignore_masks).to(torch.bool)  # [T, H, W]

        # map category IDs for instances to the range 1,...
        instance_categories = torch.tensor(
            [self.category_id_mapping[cat_id] + 1 for cat_id in instance_categories.tolist()], dtype=torch.int64
        )  # [N]

        # print(f"Classes: {mapped_seq_semantic_masks.unique().tolist()}")
        return {
            "images": seq_images,
            "instance_masks": seq_instance_masks,
            "semantic_masks": mapped_seq_semantic_masks,
            "ignore_masks": seq_ignore_masks,
            "class_ids": instance_categories,
            "meta": meta_info,
            "dataset": self.name,
            "task_type": self.task_type
        }

    def augment_sample(self, image: np.ndarray, instance_masks: np.ndarray, instance_categories: np.ndarray,
                       semantic_masks: np.ndarray) -> Any:
        """
        Augment the image/mask tuple and turn it into a sequence
        :param image: [H, W, 3] (BGR)
        :param instance_masks: instance masks of shape [N, H, W]
        :param instance_categories: list of size [N]
        :param semantic_masks: semantic masks of shape [C, H, W]
        :return:
        """
        # return self.augment_sample_by_cropping(image, instance_masks, instance_categories, semantic_masks)
        return self.augment_sample_by_affine(image, instance_masks, instance_categories, semantic_masks)

    def augment_sample_by_affine(self, image: np.ndarray, instance_masks: np.ndarray, instance_categories: np.ndarray,
                                 semantic_masks: np.ndarray):
        n_inst, n_sem = instance_masks.shape[0], semantic_masks.shape[0]
        condensed_instance_mask = np.zeros_like(instance_masks[0], np.int32)
        condensed_semantic_mask = np.zeros_like(semantic_masks[0], np.int32)

        for i, m in enumerate(instance_masks, 1):
            condensed_instance_mask = np.where(m, i, condensed_instance_mask)

        for i, m in enumerate(semantic_masks, 1):
            condensed_semantic_mask = np.where(m, i, condensed_semantic_mask)

        condensed_masks = np.stack([condensed_instance_mask, condensed_semantic_mask], 2)  # [H, W, 2]
        masks = imgaug.SegmentationMapsOnImage(condensed_masks, image.shape)

        seq_images, seq_semantic_masks, seq_instance_masks, seq_ignore_masks = [], [], [], []
        if self.fixed_crop_across_frames:
            crop_augmenter = self.augmentation_fixed_crop.to_deterministic()
        else:
            crop_augmenter = self.augmentation_fixed_crop

        for _ in range(self.clip_length):
            aug_image, aug_masks = self.augmentation_affine(image=image, segmentation_maps=masks)
            aug_masks = aug_masks.get_arr()

            # resize lower dim
            aug_image, aug_masks = self.random_resize(aug_image, aug_masks)

            # fixed size crop
            aug_masks = imgaug.SegmentationMapsOnImage(aug_masks, aug_image.shape)

            aug_image, aug_masks = crop_augmenter(image=aug_image, segmentation_maps=aug_masks)
            aug_masks = aug_masks.get_arr()  # [H, W, 2] (int32)

            inst_masks_t = np.stack([aug_masks[:, :, 0] == i for i in range(1, n_inst + 1)])
            sem_masks_t = aug_masks[:, :, 1]  # 1-indexed class IDs

            seq_images.append(aug_image)
            seq_semantic_masks.append(sem_masks_t - 1)  # make class index 0-based. don't care that the ignore regions are now at -1
            seq_ignore_masks.append(sem_masks_t == 0)
            seq_instance_masks.append(inst_masks_t)

        seq_semantic_masks = np.stack(seq_semantic_masks, 0)  # [T, H, W]
        seq_instance_masks = np.stack(seq_instance_masks, 1)  # [N, T, H, W]
        seq_images = np.stack(seq_images)  # [T, H, W, 3]
        seq_ignore_masks = np.stack(seq_ignore_masks)  # [T, H, W]

        # remove null instances
        inst_keep_flag = np.any(np.reshape(seq_instance_masks, (seq_instance_masks.shape[0], -1)), 1)  # [N]
        if not np.any(inst_keep_flag):
            raise NoValidInstancesException()

        seq_instance_masks = seq_instance_masks[inst_keep_flag]
        instance_categories = instance_categories[inst_keep_flag]

        return seq_images, seq_instance_masks, instance_categories, seq_semantic_masks, seq_ignore_masks

    def random_resize(self, image: np.ndarray, masks: np.ndarray):
        """Resize while preserving aspect ratio

        Args:
            image (np.ndarray): [H, W, 3]
            masks (np.ndarray): [H, W, 2]
        """
        height, width = image.shape[:2]
        dims = [height, width]
        lower_size = float(min(dims))
        higher_size = float(max(dims))

        if isinstance(self.augmentation_min_dims, (list, tuple)):
            min_dim = self.augmentation_min_dims[torch.randint(len(self.augmentation_min_dims), (1,)).item()]
        else:
            min_dim = self.min_daugmentation_min_dimsim

        scale_factor = min_dim / lower_size
        if (higher_size * scale_factor) > 1333:
            scale_factor = 1333 / higher_size

        new_height, new_width = round(scale_factor * height), round(scale_factor * width)

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR_EXACT)
        masks = cv2.resize(masks, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT) 
        return image, masks  # np.stack(masks, 2)

        # masks = [
        #     cv2.resize(masks[:, :, i], (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT) 
        #     for i in range(masks.shape[2])
        # ]
        # return image, np.stack(masks, 2)

    def augment_sample_by_cropping(self, image: np.ndarray, instance_masks: np.ndarray, instance_categories: np.ndarray,
                                   semantic_masks: np.ndarray):
        assert semantic_masks.shape[0] < 255

        # combine semantic masks into a single condensed mask.
        condensed_semantic_mask = np.zeros(image.shape[:2], np.uint8)
        for i, mask in enumerate(semantic_masks, 1):
            condensed_semantic_mask = np.where(mask, i, condensed_semantic_mask)

        seq_images, seq_semantic_masks, seq_instance_masks, seq_ignore_masks = [], [], [], []

        for t in range(self.clip_length):
            aug_image, transforms = T.apply_augmentations(self.augmentations, np.copy(image))
            if isinstance(transforms[-1], PadTransform):
                transforms[-1].seg_pad_value = 0
            seq_images.append(aug_image)

            aug_semantic = transforms.apply_segmentation(np.copy(condensed_semantic_mask))
            # pixel values of zero indicate padded regions
            seq_ignore_masks.append(aug_semantic == 0)

            # offset the semantic classes to be 0-based
            seq_semantic_masks.append(aug_semantic - 1)  # don't care that the padded regions are now at -1
            seq_instance_masks.append(np.stack([transforms.apply_segmentation(m) for m in instance_masks]))

        seq_semantic_masks = np.stack(seq_semantic_masks, 0)  # [T, H, W]
        seq_instance_masks = np.stack(seq_instance_masks, 1)  # [N, T, H, W]
        seq_images = np.stack(seq_images)  # [T, H, W, 3]
        seq_ignore_masks = np.stack(seq_ignore_masks)  # [T, H, W]

        # remove null instances
        inst_keep_flag = np.any(np.reshape(seq_instance_masks, (seq_instance_masks.shape[0], -1)), 1)  # [N]
        if not np.any(inst_keep_flag):
            raise NoValidInstancesException()

        seq_instance_masks = seq_instance_masks[inst_keep_flag]
        instance_categories = instance_categories[inst_keep_flag]

        return seq_images, seq_instance_masks, instance_categories, seq_semantic_masks, seq_ignore_masks


class COCOPanopticDataset(PanopticImageDataset):
    def __init__(self, images_base_dir: str, panoptic_maps_dir: str, json_annotations_path: str, clip_length: int,
                 num_samples: int, output_dims: Tuple[int]):

        with open(osp.join(osp.dirname(__file__), "metainfo", "coco_panoptic_categories.json"), 'r') as fh:
            categories_info = json.load(fh)

        super().__init__(
            images_base_dir=images_base_dir,
            panoptic_maps_dir=panoptic_maps_dir,
            parsed_image_samples=parse_json_annotations(json_annotations_path, is_cityscapes=False),
            clip_length=clip_length,
            num_samples=num_samples,
            categories_info=categories_info,
            output_dims=output_dims,
            dataset_name="COCO"
        )

    def get_stuff_category_ids(self, categories_info: List[Dict[str, Any]]) -> List[int]:
        return [x['id'] for x in categories_info if x['id'] > 90]


class ADE20kPanopticDataset(PanopticImageDataset):
    def __init__(self, images_base_dir: str, panoptic_maps_dir: str, json_annotations_path: str, clip_length: int,
                 num_samples: int, output_dims: Tuple[int]):

        with open(osp.join(osp.dirname(__file__), "metainfo", "ade20k_panoptic_categories.json"), 'r') as fh:
            categories_info = json.load(fh)

        super().__init__(
            images_base_dir=images_base_dir,
            panoptic_maps_dir=panoptic_maps_dir,
            parsed_image_samples=parse_json_annotations(json_annotations_path, is_cityscapes=False),
            clip_length=clip_length,
            num_samples=num_samples,
            categories_info=categories_info,
            output_dims=output_dims,
            dataset_name="ADE20K"
        )

    def get_stuff_category_ids(self, categories_info: List[Dict[str, Any]]) -> List[int]:
        return [x['id'] for x in categories_info if not x['isthing']]


class CityscapesPanopticDataset(PanopticImageDataset):
    def __init__(self, images_base_dir: str, panoptic_maps_dir: str, json_annotations_path: str, clip_length: int,
                 num_samples: int, output_dims: Tuple[int]):

        with open(osp.join(osp.dirname(__file__), "metainfo", "cityscapes_panoptic_categories.json"), 'r') as fh:
            categories_info = json.load(fh)

        categories_info = [x for x in categories_info if x['id'] >= 0]

        super().__init__(
            images_base_dir=images_base_dir,
            panoptic_maps_dir=panoptic_maps_dir,
            parsed_image_samples=parse_json_annotations(json_annotations_path, is_cityscapes=True),
            clip_length=clip_length,
            num_samples=num_samples,
            categories_info=categories_info,
            output_dims=output_dims,
            dataset_name="CITYSCAPES"
        )

        self.augmentations = [
            T.RandomCrop(crop_type="relative", crop_size=(0.7, 0.7)),
            T.ResizeScale(min_scale=0.5, max_scale=1.6, target_height=output_dims[0], target_width=output_dims[1]),
            T.FixedSizeCrop(crop_size=output_dims)
        ]

        # self.augmentations = [
        #     T.ResizeShortestEdge(
        #         [min(output_dims)], int(round(min(output_dims) * 1.2)), "choice"
        #     ),
        #     T.RandomCrop("absolute", output_dims)
        # ]

        # self.augmentation_affine = iaa.Affine(scale=(0.9, 1.1), rotate=(-10, 10), shear=(-8, 8))
        self.fixed_crop_across_frames = True

    def get_stuff_category_ids(self, categories_info: List[Dict[str, Any]]) -> List[int]:
        return [x['id'] for x in categories_info if not x['hasInstances']]

    def get_ignore_category_ids(self) -> List[int]:
        return [-1, 0]


class MapillaryPanopticDataset(PanopticImageDataset):
    def __init__(self, images_base_dir: str, panoptic_maps_dir: str, json_annotations_path: str, clip_length: int,
                 num_samples: int, output_dims: Tuple[int]):

        with open(osp.join(osp.dirname(__file__), "metainfo", "mapillary_panoptic_categories.json"), 'r') as fh:
            categories_info = json.load(fh)

        super().__init__(
            images_base_dir=images_base_dir,
            panoptic_maps_dir=panoptic_maps_dir,
            parsed_image_samples=parse_json_annotations(json_annotations_path, is_cityscapes=False),
            clip_length=clip_length,
            num_samples=num_samples,
            categories_info=categories_info,
            output_dims=output_dims,
            dataset_name="MAPILLARY"
        )

        self.augmentations = [
            T.RandomCrop(crop_type="relative", crop_size=(0.7, 0.7)),
            T.ResizeScale(min_scale=0.5, max_scale=1.6, target_height=output_dims[0], target_width=output_dims[1]),
            T.FixedSizeCrop(crop_size=output_dims)
        ]
        # self.augmentation_affine = iaa.Affine(scale=(0.9, 1.1), rotate=(-7, 7), shear=(-8, 8))
        self.fixed_crop_across_frames = True

        self.preprocess_transform = T.ResizeShortestEdge([800], 1333, sample_style="choice")

    def get_stuff_category_ids(self, categories_info: List[Dict[str, Any]]) -> List[int]:
        return [x['id'] for x in categories_info if not x['isthing']]

    def preprocess_image_panmap(self, image: np.ndarray, panmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply pre-processing steps to the image and panoptic map
        :param image: [H, W, 3]
        :param panmap: [H, W]
        :return:
        """
        # Mapillary images are huge and slow down the data loading. Better to resize them before futher processing
        aug_input = T.AugInput(image=image, sem_seg=panmap)
        self.preprocess_transform(aug_input)  # applied in-place
        # print(aug_input.image.shape, aug_input.sem_seg.shape)
        return aug_input.image, aug_input.sem_seg
