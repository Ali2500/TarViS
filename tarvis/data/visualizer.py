import cv2
import numpy as np
import os.path as osp
import random
import torch

from argparse import ArgumentParser
from einops import rearrange
from torch.utils.data import DataLoader

from tarvis.config import cfg
from tarvis.utils.paths import Paths
from tarvis.data.panoptic_image_dataset import (
    COCOPanopticDataset, 
    CityscapesPanopticDataset,
    MapillaryPanopticDataset,
    ADE20kPanopticDataset
)
from tarvis.data.instance_seg_video_dataset import InstanceSegDataset
from tarvis.data.panoptic_seg_dataset import PanopticSegDataset
from tarvis.data.vipseg_dataset import VipSegDataset
from tarvis.data.vos_dataset import VOSDataset
from tarvis.data.collate import collate_fn_train
from tarvis.data.format_mappers import (
    youtube_vis_mapper, 
    ovis_mapper, 
    burst_mapper, 
)
from tarvis.modelling.utils import pad_image_tensor
from tarvis.utils.visualization import create_color_map, annotate_image_instance


def visualize_dataloader_samples(dataloader: DataLoader, dataset_type: str):
    for t in range(cfg.TRAINING.CLIP_LENGTH):
        cv2.namedWindow(f'Image {t+1}', cv2.WINDOW_NORMAL)

    semantic_window_exists = False
    cmap = create_color_map().tolist()

    for batch in dataloader:
        seq_images = pad_image_tensor(batch["images"], 128, True).squeeze(0)  # [T, 3, H, W]
        if cfg.INPUT.RGB:
            seq_images = seq_images.flip([1])

        seq_images = rearrange(seq_images, "T C H W -> T H W C").byte().numpy()
        seq_instance_masks = pad_image_tensor(batch["instance_masks"], 0, False)[0]  # [N, T, H, W]
        print(f"Seq name: {batch['meta'][0]['seq_name']}")
        print(f"Frame indices: {batch['meta'][0]['frame_indices']}")
        print(f"Dims: {seq_images.shape[1:3]}")
        if "ref_frame_index" in batch:
            print(f"Ref frame index: {batch['ref_frame_index'][0] + 1}")

        if batch["semantic_masks"] is None:
            seq_semantic_masks = torch.empty(0)
            seq_ignore_masks = torch.empty(0)
            semantic_labels = None
        else:
            if not semantic_window_exists:
                for t in range(cfg.TRAINING.CLIP_LENGTH):
                    cv2.namedWindow(f'Semantic {t+1}', cv2.WINDOW_NORMAL)
                    cv2.namedWindow(f'Ignore {t+1}', cv2.WINDOW_NORMAL)
                semantic_window_exists = True

            seq_semantic_masks = pad_image_tensor(batch["semantic_masks"], 1000, False)[0]  # 1+num_classes, T, H, W]
            seq_ignore_masks = pad_image_tensor(batch["ignore_masks"], 1, False)[0]  # [T, H, W]
            seq_semantic_masks = torch.where(seq_ignore_masks, torch.full_like(seq_semantic_masks, 1000), seq_semantic_masks)
            semantic_labels = batch['meta'][0]['semantic_category_labels']

        category_labels = None
        if dataset_type in ("panoptic_seg", "instance_seg"):
            category_labels = batch['meta'][0]['category_labels']

        for t in range(cfg.TRAINING.CLIP_LENGTH):
            image = np.copy(seq_images[t])
            semantic_image = np.copy(seq_images[t])
            ignore_image = np.copy(seq_images[t])

            for iid in range(seq_instance_masks.size(0)):
                mask_iid = seq_instance_masks[iid, t].numpy()
                if not np.any(mask_iid):
                    continue

                if category_labels is None:
                    label, bbox = None, None
                else:
                    label = category_labels[iid]
                    bbox = "mask"

                image = annotate_image_instance(image, mask_iid, cmap[iid + 1], label=label, bbox=bbox, text_placement="mask_centroid")

            for cls_id in seq_semantic_masks.unique().tolist():
                if cls_id == 1000:
                    continue
                elif cls_id == -100:
                    print("-100 found but shouldn't exist!")
                    continue
                mask_cls = (seq_semantic_masks == cls_id)[t].numpy()
                label = semantic_labels[cls_id]
                semantic_image = annotate_image_instance(semantic_image, mask_cls, cmap[cls_id + 1], label=label, text_placement="mask_centroid")

            if seq_ignore_masks.numel() > 0:
                ignore_image = annotate_image_instance(ignore_image, seq_ignore_masks[t], (0, 0, 255))

            cv2.imshow(f"Image {t+1}", image)
            if seq_semantic_masks.numel() > 0:
                cv2.imshow(f"Semantic {t+1}", semantic_image)

            if seq_ignore_masks.numel() > 0:
                cv2.imshow(f"Ignore {t+1}", ignore_image)

        if cv2.waitKey(0) == 113:
            return


def main(args):
    torch.manual_seed(22021994)
    random.seed(22021994)
    np.random.seed(22021994)

    if osp.isabs(args.cfg):
        config_path = args.cfg
    else:
        config_path = osp.join(Paths.configs_dir(), args.cfg)

    cfg.merge_from_file(config_path)

    if args.dataset.lower() in ("coco", "mapillary", "ade20k", "cityscapes"):
        dataset_fn = {
            "coco": COCOPanopticDataset,
            "mapillary": MapillaryPanopticDataset,
            "cityscapes": CityscapesPanopticDataset,
            "ade20k": ADE20kPanopticDataset
        }

        images_dir = Paths.panoptic_train_images(args.dataset.lower())
        panmaps_dir, segments_info = Paths.panoptic_train_anns(args.dataset.lower())

        dataset = dataset_fn[args.dataset.lower()](
            images_base_dir=images_dir,
            panoptic_maps_dir=panmaps_dir,
            json_annotations_path=segments_info,
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            num_samples=10000,
            output_dims=[512, 512]
        )

    elif args.dataset == "ytvis":
        dataset = InstanceSegDataset(
            "YOUTUBE_VIS",
            images_dir=Paths.youtube_vis_train_images(),
            annotations_json_path=Paths.youtube_vis_train_anns(),
            annotations_mapper_fn=youtube_vis_mapper,
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.YOUTUBE_VIS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.YOUTUBE_VIS.TRAINING.IMAGE_RESIZE.as_dict()
        )

    elif args.dataset == "ovis":
        dataset = InstanceSegDataset(
            "OVIS",
            images_dir=Paths.ovis_train_images(),
            annotations_json_path=Paths.ovis_train_anns(),
            annotations_mapper_fn=ovis_mapper,
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.OVIS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.OVIS.TRAINING.IMAGE_RESIZE.as_dict()
        )

    elif args.dataset == "davis":
        dataset = VOSDataset(
            "DAVIS",
            images_dir=Paths.davis_train_images(),
            annotations_json_path=Paths.davis_train_anns(),
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.DAVIS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.DAVIS.TRAINING.IMAGE_RESIZE.as_dict(),
            max_num_instances=4
        )

    elif args.dataset == "ytvos":
        dataset = VOSDataset(
            "DAVIS",
            images_dir=Paths.youtube_vos_train_images(),
            annotations_json_path=Paths.youtube_vos_train_anns(),
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.YOUTUBE_VOS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.YOUTUBE_VOS.TRAINING.IMAGE_RESIZE.as_dict(),
            max_num_instances=4
        )

    elif args.dataset == "burst":
        dataset = VOSDataset(
            "BURST",
            images_dir=Paths.burst_train_images(),
            annotations_json_path=Paths.burst_train_anns(),
            annotations_mapper_fn=burst_mapper,
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.BURST.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.BURST.TRAINING.IMAGE_RESIZE.as_dict(),
            max_num_instances=4
        )

    elif args.dataset == "bl30k":
        dataset = VOSDataset(
            "BL30K",
            images_dir=Paths.bl30k_train_images(),
            annotations_json_path=Paths.bl30k_train_anns(),
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.BL30K.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.BL30K.TRAINING.IMAGE_RESIZE.as_dict(),
            max_num_instances=4
        )

    elif args.dataset == "kittimots":
        dataset = InstanceSegDataset(
            name="KITTI_MOTS",
            images_dir=Paths.kitti_mots_train_images(),
            annotations_json_path=Paths.kitti_mots_train_anns(),
            annotations_mapper_fn=kitti_mots_mapper,
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.KITTI_MOTS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.KITTI_MOTS.TRAINING.IMAGE_RESIZE.as_dict()
        )

    elif args.dataset == "motschallenge":
        dataset = InstanceSegDataset(
            name="MOTS_CHALLENGE",
            images_dir=Paths.mots_challenge_train_images(),
            annotations_json_path=Paths.mots_challenge_train_anns(),
            annotations_mapper_fn=mots_challenge_mapper,
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.MOTS_CHALLENGE.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.MOTS_CHALLENGE.TRAINING.IMAGE_RESIZE.as_dict()
        )

    elif args.dataset == "kittistep":
        class_id_mapping = {i: i+1 for i in range(0, 19)}
        dataset = PanopticSegDataset(
            name="KITTI_STEP",
            images_dir=Paths.kitti_step_train_images(),
            annotations_json_path=Paths.kitti_step_trainval_anns(),  # train or trainval
            annotations_mapper_fn=None,
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.KITTI_STEP.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.KITTI_STEP.TRAINING.IMAGE_RESIZE.as_dict(),
            ignore_class_id=255,
            instance_class_id_mapping=class_id_mapping
        )

    elif args.dataset == "cityscapes_vps":
        class_id_mapping = {i: i + 1 for i in range(0, 19)}
        dataset = PanopticSegDataset(
            name="CITYSCAPES_VPS",
            images_dir=Paths.cityscapes_vps_train_images(),
            annotations_json_path=Paths.cityscapes_vps_train_anns(),  # train or trainval
            annotations_mapper_fn=None,
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.CITYSCAPES_VPS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.CITYSCAPES_VPS.TRAINING.IMAGE_RESIZE.as_dict(),
            ignore_class_id=255,
            instance_class_id_mapping=class_id_mapping
        )

    elif args.dataset == "motschallenge_step":
        class_id_mapping = {i: i + 1 for i in range(0, 6)}
        dataset = PanopticSegDataset(
            name="MOTS_CHALLENGE_STEP",
            images_dir=Paths.mots_challenge_step_train_images(),
            annotations_json_path=Paths.mots_challenge_step_train_anns(),  # train or trainval
            annotations_mapper_fn=None,
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            frame_sampling_multiplicative_factor=cfg.DATASETS.MOTS_CHALLENGE_STEP.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            num_samples=10000,
            resize_params=cfg.DATASETS.MOTS_CHALLENGE_STEP.TRAINING.IMAGE_RESIZE.as_dict(),
            ignore_class_id=255,
            instance_class_id_mapping=class_id_mapping
        )

    elif args.dataset == "vipseg":
        dataset = VipSegDataset(
            images_dir=Paths.vipseg_train_images(),
            panoptic_masks_dir=Paths.vipseg_train_panoptic_masks(),
            json_info_file=Paths.vipseg_train_video_info(),
            clip_length=cfg.TRAINING.CLIP_LENGTH,
            num_samples=10000,
            frame_sampling_multiplicative_factor=cfg.DATASETS.VIPSEG.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR,
            resize_params=cfg.DATASETS.VIPSEG.TRAINING.IMAGE_RESIZE.as_dict()
        )

    else:
        raise ValueError("Should not be here")

    dataloader = DataLoader(dataset, 1, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn=collate_fn_train)
    visualize_dataloader_samples(dataloader, dataset.task_type)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--cfg", required=True)

    parser.add_argument("--num_workers", "-nw", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true')

    main(parser.parse_args())
