from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import Dict, Any, Union
from tarvis.utils.timer import Timer

from tarvis.utils.paths import Paths
from tarvis.inference.tarvis_inference_model import TarvisInferenceModel, AllObjectsLostException
from tarvis.config import cfg
from tarvis.data.inference_dataset import InferenceDataset as SequenceClipGenerator
from tarvis.data.collate import collate_fn_inference
from typing import List, Tuple, Optional
from tarvis.inference.visualization import save_vizualization
from tarvis.inference.main import process_sequence
from tarvis.demo.color_generator import ColorGenerator
from tarvis.utils.visualization import annotate_image_instance, overlay_mask_on_image
from tarvis.inference.dataset_parser import (
    CityscapesVPSParser,
    VIPSegDatasetParser,
    YoutubeVISParser,
    OVISParser
)

import numpy as np
import pycocotools.mask as mt
import os
import os.path as osp
import json
import torch


class TarvisDemoInferer:
    def __init__(self, model_path: Optional[str] = None):
        self.model: TarvisInferenceModel = None
        if model_path:
            self.init_model(model_path)

    def init_model(self, model_path: str):
        if not osp.isabs(model_path):
            model_path = osp.join(Paths.saved_models_dir(), model_path)

        expected_cfg_path = osp.join(osp.dirname(model_path), "config.yaml")
        assert osp.exists(expected_cfg_path), f"Config file not found at expected path: {expected_cfg_path}"
        cfg.merge_from_file(expected_cfg_path)

        print("Creating model...")
        model = TarvisInferenceModel("YOUTUBE_VIS").cuda()  # dataset name is a dummy value
        print("Restoring weights")
        model.restore_weights(model_path)
        self.model = model.eval()

    @torch.no_grad()
    def run(self, 
            images: List[np.ndarray],  # RGB format
            task_type: str,
            first_frame_masks: List[np.ndarray],
            first_frame_points: List[Tuple[int, int]]):
        
        dataset_name, task_type = self.task_type_to_dataset(task_type)
        self.model.change_target_dataset(dataset_name)
        cfg_dataset = cfg.DATASETS.get(dataset_name).INFERENCE.as_dict()
        cfg_dataset['MAX_TRACKS_PER_CLIP'] = 1000
        cfg_dataset['TRACK_SCORE_THRESHOLD'] = 0.5
        image_resize_params = cfg_dataset["IMAGE_RESIZE"]

        def normalize_point_coords(pt: Tuple[int, int]):
            y, x = pt
            h, w = images[0].shape[:2]
            return float(y + 0.5) / h, float(x + 0.5) / w

        images_bgr = [im[:, :, ::-1] for im in images]  # RGB to BGR
        if first_frame_masks:
            first_frame_masks = [{iid: mask for iid, mask in enumerate(first_frame_masks, 1)}]
        if first_frame_points:
            first_frame_points = [{iid: normalize_point_coords(point) for iid, point in enumerate(first_frame_points, 1)}]

        seq_clip_generator = SequenceClipGenerator(
            task_type=task_type,
            image_paths=[],
            clip_length=cfg_dataset['CLIP_LENGTH'],
            overlap_length=cfg_dataset['FRAME_OVERLAP'],
            image_resize_params=image_resize_params,
            first_frame_mask_paths=None,
            first_frame_mask_rles=None,
            first_frame_masks=first_frame_masks,
            first_frame_object_points=first_frame_points,
            images=images_bgr
        )

        data_loader = DataLoader(
            seq_clip_generator, shuffle=False, batch_size=1, num_workers=0,
            collate_fn=collate_fn_inference
        )
        seq_info = self.get_sequence_info(dataset_name, task_type, images)

        # try:
        sequence_results = process_sequence(
            model=self.model,
            mixed_precision=True,
            data_loader=data_loader,
            sequence_info=seq_info,
            inference_params=cfg_dataset
        )

        
        if task_type == "vos":
            object_masks = sequence_results["track_mask_rles"]
        elif task_type in ("instance_seg", "panoptic_seg"):
            object_masks = sequence_results["track_masks"]

        for masks_t in object_masks:
            for obj_id in masks_t:
                masks_t[obj_id] = mt.decode(masks_t[obj_id]).astype(np.uint8)

        if task_type in ("instance_seg", "panoptic_seg"):
            class_labels = self.get_category_labels(dataset_name)
            object_class_ids = {iid: x.argmax(0).item() for iid, x in sequence_results["track_class_logits"].items()}
            # for panoptic datasets, the class IDs start at 0, so we need to subtract 1 from the given class IDs since the latter
            # start from 1.
            cls_offset = 1 if task_type == "panoptic_seg" else 0  
            object_class_labels = {iid: class_labels[cls_id - cls_offset] for iid, cls_id in object_class_ids.items()}
        else:
            object_class_ids, object_class_labels = {}, {}

        semantic_masks = sequence_results.get("semantic_seg_masks", None)
        semantic_colormap = self.get_semantic_colormap(dataset_name)

        return self.visualize_masks_on_images(
            images=images,
            object_masks=object_masks,
            object_class_ids=object_class_ids,
            object_class_labels=object_class_labels,
            semantic_masks=semantic_masks,
            semantic_color_map=semantic_colormap
        )

    def visualize_masks_on_images(
            self,
            images: List[np.ndarray], 
            object_masks: List[Dict[int, np.ndarray]], 
            object_class_ids: Dict[int, int],
            object_class_labels: Dict[int, str],
            semantic_masks: Union[List[torch.Tensor], None], 
            semantic_color_map: Union[Dict[int, Tuple[int, int, int]], None]
        ):
        viz_images = []
        assert len(object_masks) == len(images)
        color_generator = ColorGenerator(semantic_color_map, delta=30)

        for t, img in enumerate(images):
            if semantic_masks is not None:
                fg_mask = np.zeros_like(img[:, :, 0])
                for _, mask in object_masks[t].items():
                    fg_mask = np.logical_or(mask, fg_mask)

                # overlay semantic masks
                semantic_mask_t = semantic_masks[t]
                class_ids = semantic_mask_t.unique().tolist()
                for cls_id in class_ids:
                    cls_mask = (semantic_mask_t == cls_id).cpu().numpy()
                    cls_mask = np.where(fg_mask, 0, cls_mask)
                    img = overlay_mask_on_image(img, cls_mask, mask_opacity=0.6, mask_color=tuple(semantic_color_map[cls_id]))

            # overlay object masks
            for obj_id, mask in object_masks[t].items():
                obj_cls_id = object_class_ids.get(obj_id, None)
                text_label = object_class_labels.get(obj_id, None)
                color = color_generator.get_color(obj_id, obj_cls_id)
                bbox = "mask" if text_label is not None else None
                img = annotate_image_instance(
                    image=img, mask=mask, color=color, label=text_label, mask_border=3, bbox=bbox
                )

            viz_images.append(img)
        return viz_images

    @staticmethod
    def task_type_to_dataset(task_type: str):
        if "VOS" in task_type:
            return "DAVIS", "vos"
        elif "PET" in task_type:
            return "BURST", "vos"
        elif "YouTube-VIS" in task_type:
            return "YOUTUBE_VIS", "instance_seg"
        elif "OVIS" in task_type:
            return "OVIS", "instance_seg"
        elif "KITTI" in task_type:
            return "KITTI_STEP", "panoptic_seg"
        elif "CITYSCAPES" in task_type:
            return "CITYSCAPES_VPS", "panoptic_seg"
        elif "VIPSeg" in task_type:
            return "VIPSEG", "panoptic_seg"
        else:
            raise RuntimeError(f"Invalid task type: {task_type}")
        
    @staticmethod
    def get_sequence_info(dataset_name: str, task_type: str, images: List[np.ndarray]):
        image_dims = images[0].shape[:2]
        info = {
            "image_dims": image_dims,
            "image_paths": [0 for _ in range(len(images))],  # length has to be equal to seq len, elements dont matter
            "first_ref_mask_frame_index": 0
        }

        if task_type == "panoptic_seg":
            if dataset_name == "KITTI_STEP":
                info["thing_class_ids"] = [11, 13]
            elif dataset_name == "CITYSCAPES_VPS":
                info["thing_class_ids"] = [11, 12, 13, 14, 15, 16, 17, 18]
            elif dataset_name == "VIPSEG":
                info["thing_class_ids"] = VIPSegDatasetParser._thing_classes
            else:
                raise ValueError(f"Unexpected dataset: {dataset_name}")
            
        return info
    
    @staticmethod
    def get_category_labels(dataset: str):
        if dataset == "YOUTUBE_VIS":
            return YoutubeVISParser._category_labels_2021
        elif dataset == "OVIS":
            return OVISParser._category_labels
        elif dataset in ("KITTI_STEP", "CITYSCAPES_VPS"):
            return CityscapesVPSParser.c_ategory_labels
        elif dataset in "VIPSEG":
            return VIPSegDatasetParser._category_labels
        else:
            raise ValueError(f"Should not be here")
        

    @staticmethod
    def get_semantic_colormap(dataset: str):
        if dataset == "VIPSEG":
            json_path = osp.join(osp.dirname(__file__), "vipseg_colormap.json")
        elif dataset in ("KITTI_STEP", "CITYSCAPES_VPS"):
            json_path = osp.join(osp.dirname(__file__), "cityscapes_colormap.json")
        else:
            json_path = None

        if json_path:
            with open(json_path, 'r') as fh:
                colormap = {int(cls_id): color for cls_id, color in json.load(fh).items()}
            return colormap
        else:
            return None
