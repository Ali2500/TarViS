from collections import defaultdict
from einops import rearrange, repeat
from typing import Dict, List, Optional, Any, Tuple
from torch import Tensor

import torch
import torch.nn.functional as F

from tarvis.config import cfg
from tarvis.modelling.tarvis_model_base import TarvisModelBase
from tarvis.modelling.utils import split_by_query_group


class AllObjectsLostException(RuntimeError):
    def __init__(self, *args):
        super().__init__(*args)


class TarvisInferenceModel(TarvisModelBase):
    def __init__(self, dataset_name: str):
        cfg_dataset = cfg.DATASETS
        train_datasets = cfg.TRAINING.DATASET_LIST

        semantic_query_group_sizes = {
            name: cfg_dataset.get(name).NUM_CLASSES for name in train_datasets
            if cfg_dataset.get(name).TYPE in ("panoptic_seg", "instance_seg")
        }
        task_type = cfg_dataset.get(dataset_name).TYPE

        super().__init__(
            semantic_query_group_sizes=semantic_query_group_sizes,
            num_obj_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
            num_bg_queries=cfg.MODEL.NUM_BACKGROUND_QUERIES
        )

        self.dataset_name = dataset_name

        if task_type == "instance_seg":
            assert dataset_name in train_datasets, f"Model was only trained on {train_datasets} which does not include" \
                f" {dataset_name}"

        self.query_group_names = self.get_query_group_names(task_type)
        self.task_type = task_type

        # For VOS/PET, we have to store the logits produced by the previous forward pass so
        # we can initialize the queries for the current forward pass
        self.previous_clip_mask_logits: Tensor = torch.empty(0)  # [N, T, H, W]
        self.previous_clip_frame_indices: Tensor = torch.empty(0)  # [T]
        self.previous_clip_instance_ids: Tensor = torch.empty(0)  # [N]

    def reset_vos_buffers(self):
        self.previous_clip_mask_logits = torch.empty(0)  # key: instance ID, value: logits [T, H, W]
        self.previous_clip_frame_indices = torch.empty(0)
        self.previous_clip_instance_ids = torch.empty(0)

    def restore_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path)['model']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        assert len(missing) == 0, f"Missing keys: {missing}"

    def extract_feature_maps(self, images: Tensor) -> List[Tensor]:
        # extract feature maps from backbone
        assert images.ndim == 5, f"Expected 5-D tensor, but got tensor of shape {images.shape}"
        batch_sz = images.size(0)
        assert batch_sz == 1, f"Batch size must be 1 during inference"

        if self.backbone.normalize_image:
            images = images / 255.

        images = (images - self.pixel_mean[None, None, :, None, None]) / self.pixel_std[None, None, :, None, None]
        fmaps = self.backbone(images)["output_features"]  # list of multi-scale features: 32x, 16x, 8x, 4x
        assert len(fmaps) == 4

        return [f.squeeze(0) for f in fmaps]  # remove batch dimension

    def prepare_vos_inputs(self, fmaps: List[Tensor], clip_frame_indices: Tensor, vos_ref_mask_info: Dict[str, Any],
                           padding_size: Tuple[int, int]):
        if self.task_type != "vos":
            return defaultdict(lambda: None)

        assert clip_frame_indices is not None
        if vos_ref_mask_info is None:
            vos_ref_mask_info = dict()

        vos_inputs = self.initialize_vos_queries(
            fmaps=fmaps,
            frame_indices=clip_frame_indices,
            vos_ref_mask_info=vos_ref_mask_info,
            image_padding_size=padding_size
        )

        assert sum(vos_inputs["object_query_counts"]) == vos_inputs["object_queries"].size(0)
        vos_inputs["object_queries"] = vos_inputs["object_queries"].unsqueeze(0)  # [1, N_o, C]
        if vos_inputs["bg_queries"] is not None:
            vos_inputs["bg_queries"] = vos_inputs["bg_queries"].unsqueeze(0)  # [1, N_b, C]

        return vos_inputs

    def forward(self, images: Tensor,
                clip_frame_indices: Optional[Tensor] = None,
                vos_ref_mask_info: Optional[Dict[str, Any]] = None):
        if self.task_type == "vos" and self.previous_clip_instance_ids.numel() == 0 and vos_ref_mask_info is None:
            # all previous objects are lost, and no news objects are present in the given clip
            raise AllObjectsLostException()

        images, vos_ref_mask_info, padding_size = self.pad_inputs(images, vos_ref_mask_info)
        fmaps = self.extract_feature_maps(images)
        vos_inputs = self.prepare_vos_inputs(fmaps, clip_frame_indices, vos_ref_mask_info, padding_size)

        fmaps = [repeat(f, "T C H W -> B C T H W", B=1) for f in fmaps]

        # get the required query set for the current sample
        query_set = self.get_query_set(
            self.query_group_names, 1, self.dataset_name,
            vos_query_init=vos_inputs["object_queries"],
            vos_bg_query_init=vos_inputs["bg_queries"]
        )

        decoder_output = self.decoder(
            fmaps=fmaps,
            queries=query_set["inits"],
            query_embed=query_set["embeds"],
            query_group_names=self.query_group_names,
            query_group_sizes=query_set["counts"]
        )

        # split the output mask logits according to query group
        all_mask_logits = split_by_query_group(
            decoder_output["mask_logits"], 2, self.query_group_names, query_set["counts"]
        )  # [L, B, Q, T, H, W]

        if self.task_type in ("instance_seg", "panoptic_seg"):
            pred_class_logits = decoder_output["class_logits"]  # [L, B, Q, 1+num_classes]
            semantic_seg_mask_logits = all_mask_logits["semantic"][-1].squeeze(0)  # [num_classes, T, H, W]

            if self.task_type == "instance_seg":  # include background mask logits
                bg_mask_logits = all_mask_logits["background"][-1].squeeze(0)  # [N_b, T, H, W]
                bg_mask_logits = bg_mask_logits.max(0, keepdim=True)[0]  # [1, T, H, W]
                semantic_seg_mask_logits = torch.cat((bg_mask_logits, semantic_seg_mask_logits), 0)  # [1+num_classes, T, H, W]

            output_dict = {
                "pred_instance_seg_logits": all_mask_logits["instance"][-1].squeeze(0),  # [Q, T, H, W]
                "pred_semantic_seg_logits": semantic_seg_mask_logits,
                "pred_instance_class_logits": pred_class_logits[-1].squeeze(0)  # [Q, 1+num_classes]
            }

        elif self.task_type == "vos":
            vos_mask_logits = all_mask_logits["vos"][-1].squeeze(0)  # [num_objects, T, H, W]
            vos_mask_logits = self.condense_vos_masks(
                vos_mask_logits, query_dim=0, object_query_counts=vos_inputs["object_query_counts"]
            )

            # if the first-frame mask/point for an object is in the middle of the clip, zero out the logits for all
            # preceding frames for that object.
            for i, inst_id in enumerate(vos_inputs["instance_ids"].tolist()):
                first_frame_id = vos_inputs["object_gt_frame_indices"].get(inst_id, 0)
                if first_frame_id > 0:
                    # print(f"Zeroing ID {inst_id} until {first_frame_id}")
                    vos_mask_logits[i, :first_frame_id] = -1e3

            bg_mask_logits = all_mask_logits["background_vos"][-1].squeeze(0)  # [N_b, T, H, W]

            output_dict = {
                "pred_vos_logits": vos_mask_logits,
                "pred_background_logits": bg_mask_logits,
                "instance_ids": vos_inputs["instance_ids"],
                "lost_instance_ids": vos_inputs["lost_instance_ids"]
            }

            self.previous_clip_mask_logits = vos_mask_logits
            self.previous_clip_instance_ids = vos_inputs["instance_ids"]
            self.previous_clip_frame_indices = clip_frame_indices

        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

        return output_dict

    def initialize_vos_queries(self, fmaps: List[Tensor],
                               frame_indices: Tensor,
                               vos_ref_mask_info: Dict[str, Any],
                               image_padding_size: Tuple[int, int]):
        """
        Initialize queries for the VOS task.
        :param fmaps: list of tensors, each of shape [T, C, H', W']
        :param frame_indices: tensor of shape [T]
        :param vos_ref_mask_info:
        :param image_padding_size:
        :return:
        """
        lost_instance_ids = []
        fmap_4x = fmaps[-1]
        assert frame_indices.size(0) == fmap_4x.size(0), f"Shape mismatch: {frame_indices.shape}, {fmap_4x.shape}"
        img_height, img_width = fmap_4x.shape[-2] * 4, fmap_4x.shape[-1] * 4

        gt_points_available = "point_coords" in vos_ref_mask_info
        gt_masks_available = "masks" in vos_ref_mask_info
        assert not (gt_points_available and gt_masks_available)  # either one but not both
        gt_instance_id_set = set()

        point_vos_coords = None
        fmaps_point_vos = None
        point_vos_instance_ids = torch.zeros(0, dtype=torch.int64)

        mask_vos_ref_frame_local_indices = None
        mask_vos_instance_ids = torch.zeros(0, dtype=torch.int64)

        object_gt_frame_indices = dict()

        t_min = int(1e6)
        t_max = -1
        active_frames = []

        if gt_points_available or gt_masks_available:
            active_frames.extend(vos_ref_mask_info["frame_indices"].tolist())
            t_min = min(vos_ref_mask_info["frame_indices"].min().item(), t_min)
            t_max = max(vos_ref_mask_info["frame_indices"].max().item(), t_max)
            gt_instance_id_set = set(vos_ref_mask_info["instance_ids"].tolist())

            object_gt_frame_indices = {
                inst_id: t.item() for inst_id, t in zip(vos_ref_mask_info["instance_ids"].tolist(),
                                                        vos_ref_mask_info["frame_indices"] - frame_indices[0])
            }

        if gt_points_available:
            point_vos_coords = vos_ref_mask_info["point_coords"].to(fmaps[0].device)  # [N, 2] (y, x coords)

            point_vos_ref_frame_local_indices = vos_ref_mask_info["frame_indices"] - frame_indices[0]
            point_vos_instance_ids = vos_ref_mask_info["instance_ids"]

            # the reference point could lie in different frames for different objects.
            fmaps_point_vos = [f[point_vos_ref_frame_local_indices] for f in fmaps]  # each element: [N, C, H, W]

        if gt_masks_available:
            assert list(vos_ref_mask_info["masks"].shape[-2:]) == [img_height, img_width]
            # mask_vos_ref_frame_local_indices = vos_ref_mask_info["frame_indices"] - frame_indices[0]

        overlapping_frame_indices = sorted(list(set(self.previous_clip_frame_indices.tolist()).intersection(set(frame_indices.tolist()))))
        active_frames.extend(overlapping_frame_indices)
        num_overlapping_frames = len(overlapping_frame_indices)

        if num_overlapping_frames > 0:
            t_min = min(overlapping_frame_indices + [t_min])
            t_max = max(overlapping_frame_indices + [t_max])

        vos_init_frames = torch.arange(t_min, t_max + 1, dtype=torch.int64)

        mask_vos_ref_masks = defaultdict(
            lambda: torch.zeros(len(vos_init_frames), img_height, img_width, dtype=torch.bool, device=fmap_4x.device)
        )

        if num_overlapping_frames > 0:
            prev_clip_masks = self.previous_clip_mask_logits[:, -num_overlapping_frames:]  # [N, T', H, W]
            prev_clip_masks = F.interpolate(prev_clip_masks, scale_factor=4.0, mode='bilinear', align_corners=False)
            prev_clip_masks = prev_clip_masks > 0.0

            # overlapping frames should be at the start of the current clip and at the end of the previous clip
            assert overlapping_frame_indices == frame_indices[:num_overlapping_frames].tolist()
            assert overlapping_frame_indices == self.previous_clip_frame_indices[-num_overlapping_frames:].tolist()

            overlapping_frame_indices = torch.as_tensor(overlapping_frame_indices, dtype=torch.int64)
            local_frame_indices = overlapping_frame_indices - vos_init_frames[0]

            for inst_id, inst_mask in zip(self.previous_clip_instance_ids.tolist(), prev_clip_masks):
                if not torch.any(inst_mask):
                    if inst_id not in gt_instance_id_set:
                        lost_instance_ids.append(inst_id)
                    continue

                mask_vos_ref_masks[inst_id][local_frame_indices] = inst_mask

        if gt_masks_available:
            gt_masks = vos_ref_mask_info["masks"].to(fmap_4x.device)  # [M, H, W]
            mask_vos_ref_frame_local_indices = vos_ref_mask_info["frame_indices"] - t_min
            assert torch.all(mask_vos_ref_frame_local_indices >= 0)

            for t_init, inst_id, inst_mask in zip(mask_vos_ref_frame_local_indices.tolist(),
                                                  vos_ref_mask_info["instance_ids"].tolist(),
                                                  gt_masks):
                # If a ref mask is part of the set of overlapping frames, we will have the GT mask for this object,
                # but also predicted masks. In such cases, it makes sense to overwrite the predicted mask with the
                # ground-truth mask for that particular frame. The predicted masks in the remaining frames can stay
                # (not sure if this is the best policy though)
                mask_vos_ref_masks[inst_id][t_init] = inst_mask

        if len(mask_vos_ref_masks) > 0:
            mask_vos_instance_ids = list(mask_vos_ref_masks.keys())
            mask_vos_ref_masks = torch.stack([mask_vos_ref_masks[inst_id] for inst_id in mask_vos_instance_ids])
            mask_vos_instance_ids = torch.tensor(mask_vos_instance_ids, dtype=torch.int64)
        else:
            mask_vos_ref_masks = torch.zeros(0, len(vos_init_frames), img_height, img_width, dtype=torch.bool, device=fmap_4x.device)

        fmap_mask_vos = rearrange(fmap_4x[vos_init_frames - frame_indices[0]], "T C H W -> C T H W")

        active_frames = set(active_frames)
        vos_init_frames = vos_init_frames.tolist()
        assert active_frames.issubset(set(vos_init_frames)), f"{active_frames}, {vos_init_frames}"
        frame_idxes_to_keep = [i for i, t_init in enumerate(vos_init_frames) if t_init in active_frames]

        mask_vos_ref_masks = mask_vos_ref_masks[:, frame_idxes_to_keep]
        fmap_mask_vos = fmap_mask_vos[:, frame_idxes_to_keep]

        all_instance_ids = torch.cat((mask_vos_instance_ids, point_vos_instance_ids))

        if all_instance_ids.numel() == 0:
            raise AllObjectsLostException()  # we've lost all the objects we were supposed to track! :(

        vos_queries_dict = self.vos_query_extractor(
            fmap_mask_vos=fmap_mask_vos,
            ref_masks=mask_vos_ref_masks,
            fmaps_point_vos=fmaps_point_vos,
            point_coords=point_vos_coords,
            image_padding=image_padding_size,
            mode="infer"
        )

        vos_queries_dict["instance_ids"] = all_instance_ids
        vos_queries_dict["lost_instance_ids"] = lost_instance_ids
        vos_queries_dict["object_gt_frame_indices"] = object_gt_frame_indices

        return vos_queries_dict

    def pad_inputs(self, images: Tensor, vos_ref_mask_info: Optional[Dict[str, Any]] = None):
        images, padding = self.pad_tensors(images, 128, stack=None, return_padding=True)

        if vos_ref_mask_info is not None:
            if "masks" in vos_ref_mask_info:
                vos_ref_mask_info["masks"] = self.pad_tensors(vos_ref_mask_info["masks"], pad_value=0, stack=None)

        return images, vos_ref_mask_info, padding
