from typing import Dict, Any, Union, Callable, List, Optional
from functools import partial
from tarvis.config import cfg
from tarvis.utils.paths import Paths

# dataset classes
from tarvis.data.training_dataset_base import ConcatDataset
from tarvis.data.vos_dataset import VOSDataset
from tarvis.data.instance_seg_video_dataset import InstanceSegDataset
from tarvis.data.panoptic_seg_dataset import PanopticSegDataset
from tarvis.data.vipseg_dataset import VipSegDataset
from tarvis.data.panoptic_image_dataset import (
    COCOPanopticDataset, CityscapesPanopticDataset, MapillaryPanopticDataset,
    ADE20kPanopticDataset
)
from tarvis.data.format_mappers import (
    youtube_vis_mapper, ovis_mapper, burst_mapper
)

import numpy as np


def _build_instance_seg_dataset(name: str, images_dir: str, annotations_json_path: str,
                                annotations_mapper_fn: Union[Callable, None], num_samples: int,
                                frame_sampling_factor: float, resize_params: Dict[str, Any]):
    return InstanceSegDataset(
        name=name,
        images_dir=images_dir,
        annotations_json_path=annotations_json_path,
        annotations_mapper_fn=annotations_mapper_fn,
        num_samples=num_samples,
        clip_length=cfg.TRAINING.CLIP_LENGTH,
        frame_sampling_multiplicative_factor=frame_sampling_factor,
        resize_params=resize_params
    )


def _build_vos_dataset(name: str, images_dir: str, annotations_json_path: str,
                       num_samples: int, frame_sampling_factor: float,
                       resize_params: Dict[str, Any], annotations_mapper_fn: Optional[Callable] = None):
    return VOSDataset(
        name=name,
        images_dir=images_dir,
        annotations_json_path=annotations_json_path,
        annotations_mapper_fn=annotations_mapper_fn,
        clip_length=cfg.TRAINING.CLIP_LENGTH,
        num_samples=num_samples,
        frame_sampling_multiplicative_factor=frame_sampling_factor,
        resize_params=resize_params,
        max_num_instances=4
    )


def build_image_panoptic_dataset(name, output_dims, num_samples: int):
    dataset_class_dict = {
        "coco": COCOPanopticDataset,
        "mapillary": MapillaryPanopticDataset,
        "cityscapes": CityscapesPanopticDataset,
        "ade20k": ADE20kPanopticDataset
    }

    images_dir = Paths.panoptic_train_images(name)
    panmaps_dir, segments_info = Paths.panoptic_train_anns(name)

    return dataset_class_dict[name](
        images_base_dir=images_dir,
        panoptic_maps_dir=panmaps_dir,
        json_annotations_path=segments_info,
        clip_length=cfg.TRAINING.CLIP_LENGTH,
        num_samples=num_samples,
        output_dims=output_dims
    )


def build_youtube_vis_dataset(num_samples):
    return _build_instance_seg_dataset(
        name="YOUTUBE_VIS",
        images_dir=Paths.youtube_vis_train_images(),
        annotations_json_path=Paths.youtube_vis_train_anns(),
        annotations_mapper_fn=youtube_vis_mapper,
        num_samples=num_samples,
        frame_sampling_factor=cfg.DATASETS.YOUTUBE_VIS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
        resize_params=cfg.DATASETS.YOUTUBE_VIS.TRAINING.IMAGE_RESIZE.as_dict()
    )


def build_ovis_dataset(num_samples):
    return _build_instance_seg_dataset(
        name="OVIS",
        images_dir=Paths.ovis_train_images(),
        annotations_json_path=Paths.ovis_train_anns(),
        annotations_mapper_fn=ovis_mapper,
        num_samples=num_samples,
        frame_sampling_factor=cfg.DATASETS.OVIS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
        resize_params=cfg.DATASETS.OVIS.TRAINING.IMAGE_RESIZE.as_dict()
    )


def build_kitti_step_dataset(num_samples):
    split = cfg.DATASETS.KITTI_STEP.TRAINING.SPLIT
    assert split == "train", f"Invalid split: {split}"

    instance_class_id_mapping = {i: i + 1 for i in range(0, 19)}

    return PanopticSegDataset(
        name="KITTI_STEP",
        images_dir=Paths.kitti_step_train_images(),
        annotations_json_path=Paths.kitti_step_train_anns(),
        annotations_mapper_fn=None,
        clip_length=cfg.TRAINING.CLIP_LENGTH,
        num_samples=num_samples,
        frame_sampling_multiplicative_factor=cfg.DATASETS.KITTI_STEP.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
        resize_params=cfg.DATASETS.KITTI_STEP.TRAINING.IMAGE_RESIZE.as_dict(),
        ignore_class_id=255,
        instance_class_id_mapping=instance_class_id_mapping
    )


def build_cityscapes_vps_dataset(num_samples):
    split = cfg.DATASETS.CITYSCAPES_VPS.TRAINING.SPLIT
    assert split == "train", f"Invalid split: {split}"

    instance_class_id_mapping = {i: i + 1 for i in range(0, 19)}

    return PanopticSegDataset(
        name="CITYSCAPES_VPS",
        images_dir=Paths.cityscapes_vps_train_images(),
        annotations_json_path=Paths.cityscapes_vps_train_anns(),
        annotations_mapper_fn=None,
        clip_length=cfg.TRAINING.CLIP_LENGTH,
        num_samples=num_samples,
        frame_sampling_multiplicative_factor=cfg.DATASETS.CITYSCAPES_VPS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
        resize_params=cfg.DATASETS.CITYSCAPES_VPS.TRAINING.IMAGE_RESIZE.as_dict(),
        ignore_class_id=255,
        instance_class_id_mapping=instance_class_id_mapping
    )


def build_davis_dataset(num_samples):
    return _build_vos_dataset(
        name="DAVIS",
        images_dir=Paths.davis_train_images(),
        annotations_json_path=Paths.davis_train_anns(),
        num_samples=num_samples,
        frame_sampling_factor=cfg.DATASETS.DAVIS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
        resize_params=cfg.DATASETS.DAVIS.TRAINING.IMAGE_RESIZE.as_dict()
    )


def build_youtube_vos_dataset(num_samples):
    return _build_vos_dataset(
        name="YOUTUBE_VOS",
        images_dir=Paths.youtube_vos_train_images(),
        annotations_json_path=Paths.youtube_vos_train_anns(),
        num_samples=num_samples,
        frame_sampling_factor=cfg.DATASETS.YOUTUBE_VOS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
        resize_params=cfg.DATASETS.YOUTUBE_VOS.TRAINING.IMAGE_RESIZE.as_dict()
    )


def build_burst_dataset(num_samples):
    return _build_vos_dataset(
        name="BURST",
        images_dir=Paths.burst_train_images(),
        annotations_json_path=Paths.burst_train_anns(),
        annotations_mapper_fn=burst_mapper,
        num_samples=num_samples,
        frame_sampling_factor=cfg.DATASETS.BURST.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
        resize_params=cfg.DATASETS.BURST.TRAINING.IMAGE_RESIZE.as_dict()
    )


def build_vipseg_dataset(num_samples):
    return VipSegDataset(
        images_dir=Paths.vipseg_train_images(),
        panoptic_masks_dir=Paths.vipseg_train_panoptic_masks(),
        json_info_file=Paths.vipseg_train_video_info(),
        clip_length=cfg.TRAINING.CLIP_LENGTH,
        num_samples=num_samples,
        frame_sampling_multiplicative_factor=cfg.DATASETS.VIPSEG.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
        resize_params=cfg.DATASETS.VIPSEG.TRAINING.IMAGE_RESIZE.as_dict()
    )


def build_concat_dataset(dataset_list: List[str], dataset_weights: List[float], num_samples: int):
    assert abs(sum(dataset_weights) - 1.0) < 1e-4, f"Sum of weights is {sum(dataset_weights)} but should be 1.0"

    dataset_builder_fn = {
        # video datasets
        "YOUTUBE_VIS": build_youtube_vis_dataset,
        "OVIS": build_ovis_dataset,
        "KITTI_STEP": build_kitti_step_dataset,
        "CITYSCAPES_VPS": build_cityscapes_vps_dataset,
        "VIPSEG": build_vipseg_dataset,
        "DAVIS": build_davis_dataset,
        "BURST": build_burst_dataset,
        # image datasets
        "COCO": partial(build_image_panoptic_dataset, "coco", cfg.TRAINING.PRETRAIN_IMAGE_SIZE),
        "MAPILLARY": partial(build_image_panoptic_dataset, "mapillary", cfg.TRAINING.PRETRAIN_IMAGE_SIZE),
        "CITYSCAPES": partial(build_image_panoptic_dataset, "cityscapes", cfg.TRAINING.PRETRAIN_IMAGE_SIZE),
        "ADE20K": partial(build_image_panoptic_dataset, "ade20k", cfg.TRAINING.PRETRAIN_IMAGE_SIZE),
    }

    if len(dataset_list) > 1:
        dataset_num_samples = np.round(np.array(dataset_weights, np.float32) * num_samples).astype(int)
        dataset_num_samples[-1] = num_samples - dataset_num_samples[:-1].sum()
        dataset_num_samples = dataset_num_samples.tolist()
    else:
        dataset_num_samples = [num_samples]

    datasets = []
    for ds_name, ds_num_samples in zip(dataset_list, dataset_num_samples):
        datasets.append(dataset_builder_fn[ds_name](ds_num_samples))

    if len(datasets) > 1:
        return ConcatDataset(datasets)
    else:
        return datasets[0]
