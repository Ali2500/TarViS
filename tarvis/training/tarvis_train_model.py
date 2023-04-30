from collections import defaultdict
from einops import rearrange
from typing import Dict, List, Any
from torch import Tensor

from tarvis.config import cfg
from tarvis.modelling.tarvis_model_base import TarvisModelBase
from tarvis.modelling.utils import split_by_query_group
from tarvis.training.loss_computation.instance_seg import InstanceSegLoss
from tarvis.training.loss_computation.panoptic_seg import PanopticSegLoss
from tarvis.training.loss_computation.semantic_seg import SemanticSegLoss
from tarvis.training.loss_computation.vos import VOSLoss

import random
import torch
import torch.nn.functional as F


class TarvisTrainModel(TarvisModelBase):
    def __init__(self):
        dataset_class_group_sizes = dict()
        cfg_datasets = cfg.DATASETS.as_dict()

        for ds in cfg.TRAINING.DATASET_LIST:
            if cfg_datasets[ds]["TYPE"] in ("instance_seg", "panoptic_seg"):
                dataset_class_group_sizes[ds] = cfg_datasets[ds]["NUM_CLASSES"]

        super().__init__(
            semantic_query_group_sizes=dataset_class_group_sizes,
            num_obj_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
            num_bg_queries=cfg.MODEL.NUM_BACKGROUND_QUERIES
        )

        num_points_for_mask_loss = cfg.TRAINING.NUM_SAMPLED_POINTS_FOR_MASK_LOSS
        self.instance_seg_criterion = InstanceSegLoss(
            bg_coef=0.1, num_points=num_points_for_mask_loss, oversample_ratio=3,
            importance_sample_ratio=0.75
        )

        self.semantic_seg_criterion = SemanticSegLoss(
            w_dice=0.0, w_ce=cfg.TRAINING.LOSSES.INSTANCE_SEMANTIC_WEIGHT_CROSS_ENTROPY,
            num_points=num_points_for_mask_loss,
            ignore_null_class_logits=cfg.TRAINING.LOSSES.SEMANTIC_IGNORE_NULL_CLASSES,
            logit_regularization_loss=cfg.TRAINING.LOSSES.SEMANTIC_REGULARIZATION_LOSS
        )

        self.vos_criterion = VOSLoss(
            num_points=num_points_for_mask_loss, apply_bg_loss=True, w_dice=5.0, w_ce=5.0, 
            loss_type="sparse"
        )

        self.panoptic_seg_criterion = PanopticSegLoss(
            num_points_instance=cfg.TRAINING.LOSSES.PANOPTIC_INSTANCE_NUM_POINTS,
            num_points_semantic=cfg.TRAINING.LOSSES.PANOPTIC_SEMANTIC_NUM_POINTS,
            semantic_loss_type=cfg.TRAINING.LOSSES.PANOPTIC_SEMANTIC,
            semantic_w_ce=cfg.TRAINING.LOSSES.PANOPTIC_SEMANTIC_WEIGHT_CROSS_ENTROPY,
            semantic_w_dice=cfg.TRAINING.LOSSES.PANOPTIC_SEMANTIC_WEIGHT_DICE,
            oversample_ratio=3,
            importance_sample_ratio=0.75,
            ignore_null_class_logits=cfg.TRAINING.LOSSES.SEMANTIC_IGNORE_NULL_CLASSES,
            semantic_logit_regularization_loss=cfg.TRAINING.LOSSES.SEMANTIC_REGULARIZATION_LOSS
        )

    @property
    def pretraining(self):
        return cfg.TRAINING.PRETRAINING

    @property
    def ref_mask_in_attention_for_vos(self):
        return cfg.TRAINING.REF_MASK_IN_ATTENTION_FOR_VOS

    def preprocess_inputs(self, inputs: Dict[str, Any], target_task: str):
        inputs["images"] = self.pad_tensors(inputs["images"], pad_value=128, stack=True)
        inputs["instance_masks"] = self.pad_tensors(inputs["instance_masks"], pad_value=0, stack=False)

        if inputs["semantic_masks"] is not None:  # panoptic seg sample
            inputs["semantic_masks"] = self.pad_tensors(inputs["semantic_masks"], pad_value=0, stack=True)  # [B, T, H, W]
            inputs["ignore_masks"] = self.pad_tensors(inputs["ignore_masks"], pad_value=1, stack=True)  # [B, T, H, W]

        if self.pretraining:
            inputs["task_type"] = target_task
            if target_task == "vos":
                inputs = self.make_usable_as_vos_sample(inputs)

        return inputs

    def extract_feature_maps(self, inputs: Dict[str, Any]) -> List[Tensor]:
        images = inputs["images"]  # [B, T, 3, H, W] (RGB)
        if self.backbone.normalize_image:
            images = images / 255.

        images = (images - self.pixel_mean[None, None, :, None, None]) / self.pixel_std[None, None, :, None, None]

        backbone_output_dict = self.backbone(images)
        fmaps = backbone_output_dict["output_features"]  # list of multi-scale features: 32x, 16x, 8x, 4x
        assert len(fmaps) == 4

        return [rearrange(f, "B T C H W -> B C T H W") for f in fmaps]

    def prepare_vos_inputs(self, inputs: Dict[str, Any], fmaps: List[Tensor]):
        # The forward pass logic is simplified if all batch samples have the same number of objects.
        num_instances = [masks.size(0) for masks in inputs["instance_masks"]]
        if len(set(num_instances)) > 1:
            min_num_instances = min(num_instances)
            inputs["instance_masks"] = [
                masks[torch.randperm(n)[:min_num_instances]]
                for masks, n in zip(inputs["instance_masks"], num_instances)
            ]

        assert len(set(inputs["ref_frame_index"])) == 1, \
            f"All batch samples must share the same reference frame index"

        ref_frame_index = inputs["ref_frame_index"][0]

        gt_instance_masks = inputs["instance_masks"]  # List[B, tensor[N, T, H, W]]
        ref_masks = [m[:, ref_frame_index] for m in gt_instance_masks]  # List[tensor[N, H, W]]

        ref_fmaps = [f[:, :, ref_frame_index].unsqueeze(2) for f in fmaps]  # [B, C, T(1), H, W]
        ref_masks = [m.unsqueeze(1) for m in ref_masks]  # List[tensor[N, T(1), H, W]]

        output_dict = self.vos_query_extractor(fmaps=ref_fmaps, ref_masks=ref_masks, mode="train")
        return output_dict

    def forward(self, inputs, target_task: str):
        inputs = self.preprocess_inputs(inputs, target_task)
        # print(inputs["dataset"])
        # self.viz_masks(inputs)

        assert len(set(inputs["dataset"])) == 1, f"Found images from multiple datasets: {inputs['dataset']}"
        dataset_name = inputs["dataset"][0]

        batch_sz, num_frames = inputs["images"].shape[:2]
        fmaps = self.extract_feature_maps(inputs)

        query_group_names = self.get_query_group_names(inputs["task_type"])

        if inputs["task_type"] == "vos":
            vos_inputs = self.prepare_vos_inputs(inputs, fmaps)
            gt_classes = None
        else:
            vos_inputs = defaultdict(lambda: None)
            gt_classes = inputs["class_ids"]  # List[B, tensor[N]]

        query_set = self.get_query_set(
            query_group_names=query_group_names,
            batch_sz=batch_sz,
            dataset_name=dataset_name,
            vos_query_init=vos_inputs["object_queries"],
            vos_bg_query_init=vos_inputs["bg_queries"]
        )
        gt_instance_masks = inputs["instance_masks"]  # List[B, tensor[N, T, H, W]]

        # Forward pass through decoder
        decoder_output = self.decoder(fmaps=fmaps,
                                      queries=query_set["inits"],
                                      query_embed=query_set["embeds"],
                                      query_group_names=query_group_names,
                                      query_group_sizes=query_set["counts"])

        # split the output mask logits according to query group
        all_mask_logits = split_by_query_group(
            decoder_output["mask_logits"], 2, query_group_names, query_set["counts"]
        )  # [L, B, Q, T, H, W]

        output_dict = dict()

        if inputs["task_type"] == "vos":
            output_dict = self.compute_vos_outputs(
                all_mask_logits=all_mask_logits, decoder_output=decoder_output, query_group_names=query_group_names,
                gt_instance_masks=gt_instance_masks, object_query_counts=vos_inputs["object_query_counts"],
                output_dict=output_dict
            )

        elif inputs["task_type"] == "instance_seg":
            output_dict = self.compute_instance_seg_outputs(
                all_mask_logits=all_mask_logits, decoder_output=decoder_output, gt_instance_masks=gt_instance_masks,
                gt_classes=gt_classes, output_dict=output_dict
            )

        elif inputs["task_type"] == "panoptic_seg":
            output_dict = self.compute_panoptic_seg_outputs(
                all_mask_logits=all_mask_logits, decoder_output=decoder_output, dataset_name=dataset_name,
                gt_instance_masks=gt_instance_masks, gt_semantic_masks=inputs["semantic_masks"],
                gt_classes=gt_classes, ignore_masks=inputs["ignore_masks"], output_dict=output_dict
            )

        else:
            raise ValueError(f"Invalid task type: {inputs['task_type']}")

        return output_dict

    def compute_vos_outputs(self, all_mask_logits: Dict[str, Tensor], decoder_output: Dict[str, Tensor],
                            query_group_names: List[str], gt_instance_masks: List[Tensor],
                            object_query_counts: List[int], output_dict: Dict[str, Any]):
        # sanity checks
        assert "class_logits" not in decoder_output
        assert "instance" not in query_group_names

        bg_mask_logits = all_mask_logits["background_vos"]  # [L, B, Qb, T, H, W]
        vos_mask_logits = all_mask_logits["vos"]  # [L, B, Qv, T, H, W]

        vos_mask_logits = self.condense_vos_masks(vos_mask_logits, query_dim=2, object_query_counts=object_query_counts)

        bg_mask_logits = bg_mask_logits.max(2, keepdim=True)[0]  # L, B, 1, T, H, W]
        combined_mask_logits = torch.cat((bg_mask_logits, vos_mask_logits), 2)

        output_dict["pred_vos_task_logits"] = combined_mask_logits
        output_dict["losses"] = dict()

        output_dict["losses"]["vos"] = self.vos_criterion(
            pred_mask_logits=combined_mask_logits,
            gt_masks=gt_instance_masks
        )

        return output_dict

    def compute_instance_seg_outputs(self, all_mask_logits: Dict[str, Tensor], decoder_output: Dict[str, Tensor],
                                     gt_instance_masks: List[Tensor], gt_classes: List[Tensor],
                                     output_dict: Dict[str, Any]):

        pred_class_logits = decoder_output["class_logits"]  # [L, B, Q, 1+num_classes]

        output_dict.update({
            "pred_instance_seg_logits": all_mask_logits["instance"],
            "pred_instance_class_logits": pred_class_logits,
            "losses": dict()
        })

        output_dict["losses"]["instance_seg"] = self.instance_seg_criterion(
            pred_mask_logits=all_mask_logits["instance"],
            pred_class_logits=pred_class_logits,
            gt_masks=gt_instance_masks,
            gt_classes=gt_classes
        )

        bg_mask_logits = all_mask_logits["background"]  # [L, B, Qb, T, H, W]
        semantic_mask_logits = all_mask_logits["semantic"]  # [L, B, Qs, T, H, W]

        bg_mask_logits = bg_mask_logits.max(2, keepdim=True)[0]  # L, B, 1, T, H, W]
        combined_mask_logits = torch.cat((bg_mask_logits, semantic_mask_logits), 2)

        output_dict["losses"]["semantic_seg"] = self.semantic_seg_criterion(
            pred_mask_logits=combined_mask_logits,
            gt_instance_masks=gt_instance_masks,
            gt_classes=gt_classes
        )

        return output_dict

    def compute_panoptic_seg_outputs(self, all_mask_logits: Dict[str, Tensor], decoder_output: Dict[str, Tensor],
                                     dataset_name, gt_instance_masks: List[Tensor], gt_semantic_masks: Tensor,
                                     gt_classes: List[Tensor], ignore_masks: Tensor, output_dict: Dict[str, Any]):
        pred_class_logits = decoder_output["class_logits"]  # [L, B, Q, 1+num_classes]

        output_dict.update({
            "pred_instance_seg_logits": all_mask_logits["instance"],
            "pred_semantic_seg_logits": all_mask_logits["semantic"],
            "pred_instance_class_logits": pred_class_logits,
            "losses": dict()
        })

        # if dataset_name == "KITTI_STEP" and "CITYSCAPES_VPS" in cfg.TRAINING.DATASET_LIST and self.shared_semseg_queries_for_cityscapes:
        #     instance_cls_ignore_mask = self.get_kitti_step_instance_cls_ignore_mask(
        #         all_mask_logits["instance"], gt_semantic_masks
        #     )
        # else:
        #     instance_cls_ignore_mask = None

        output_dict["losses"]["panoptic_seg"] = self.panoptic_seg_criterion(
            pred_instance_mask_logits=all_mask_logits["instance"],
            gt_instance_masks=gt_instance_masks,
            pred_instance_class_logits=pred_class_logits,
            gt_instance_classes=gt_classes,
            pred_semantic_mask_logits=all_mask_logits["semantic"],
            gt_semantic_masks=gt_semantic_masks,
            ignore_masks=ignore_masks,
            instance_cls_ignore_mask=None
        )

        return output_dict

    # @torch.no_grad()
    # def convert_d2_inputs(self, d2_inputs):
    #     instances_masks = []
    #     class_ids = []
    #     for b in range(len(d2_inputs)):
    #         instance_masks_b = [instances_t.gt_masks for instances_t in d2_inputs[b]["instances"]]  # List[T, [N, H, W]]
    #         instances_masks.append(torch.stack(instance_masks_b, 1))  # [N, T, H, W]
    #         class_ids.append(d2_inputs[b]["instances"][0].gt_classes + 1)

    #     return {
    #         "images": torch.stack([sample["images"] for sample in d2_inputs], 0),  # [B, T, 3, H, W] (RGB)
    #         "instance_masks": instances_masks,
    #         "class_ids": class_ids,
    #         "dataset": [sample["dataset"] for sample in d2_inputs],
    #         "semantic_masks": None,
    #         "ignore_masks": None,
    #         "task_type": "instance_seg"
    #     }

    @torch.no_grad()
    def make_usable_as_vos_sample(self, inputs):
        instance_masks = inputs["instance_masks"]  # List[B, [N, T, H, W]]
        instance_class_ids = inputs["class_ids"]  # List[B, [N]]

        # We need to choose a reference frame such that there is at least one instance in that frame
        # for all batch samples.
        feasible_ref_frames = torch.stack([torch.any(m.flatten(2), 2).any(0)  for m in instance_masks])  # [B, T]

        if torch.any(torch.all(feasible_ref_frames, 0)):
            feasible_ref_frames = torch.all(feasible_ref_frames, 0)  # [T]
            feasible_ref_frames = feasible_ref_frames.nonzero(as_tuple=False).squeeze(1).tolist()
            ref_frame_index = random.choice(feasible_ref_frames)
        else:
            # We can rearrange the frame order such that there is at least
            # one valid ref frame. This is fine because the clip comes from an augmented sequence so
            # there's no smooth video motion anyway
            first_valid_frame = [x.nonzero(as_tuple=False).squeeze(1).tolist()[0] for x in feasible_ref_frames.unbind(0)]  # [B]
            reorder_perm = [
                torch.tensor([t_valid] + [t for t in range(feasible_ref_frames.size(1)) if t != t_valid], dtype=torch.int64)
                for t_valid in first_valid_frame
            ]

            def reorder_tensor(x):
                return torch.stack([
                    x[b, perm_b] for b, perm_b in enumerate(reorder_perm)
                ])

            # rearrange 'images', 'instance_masks', 'semantic_masks', 'ignore_masks'
            inputs["images"] = reorder_tensor(inputs["images"])
            inputs["semantic_masks"] = reorder_tensor(inputs["semantic_masks"])
            inputs["ignore_masks"] = reorder_tensor(inputs["ignore_masks"])
            inputs["instance_masks"] = [
                instance_masks[b][:, perm_b] for b, perm_b in enumerate(reorder_perm)
            ]
            ref_frame_index = 0

        # calculate which instances are usable for each batch sample
        valid_instance_mask = [
            torch.any(rearrange(masks_per_sample, "N T H W -> N T (H W)")[:, ref_frame_index], 1).bool()
            for masks_per_sample in instance_masks
        ]

        # to simplify forward pass, we keep the same number of objects across all batch samples
        num_objects = [m.sum(dtype=torch.int64).item() for m in valid_instance_mask]
        min_num_objects = min(num_objects)
        assert min_num_objects > 0, f"Valid objects: {num_objects}"

        if min_num_objects * self.num_vos_queries_per_object > 32:  # avoid OOM error when too many instances
            min_num_objects = 32 // self.num_vos_queries_per_object

        pruned_instance_masks = []
        pruned_class_ids = []

        for masks_per_sample, class_ids_per_sample, keep_mask, num_objects_per_sample in zip(
                instance_masks, instance_class_ids, valid_instance_mask, num_objects
        ):
            keep_idxes = torch.randperm(num_objects_per_sample)[:min_num_objects]
            pruned_instance_masks.append(masks_per_sample[keep_mask][keep_idxes])
            pruned_class_ids.append(class_ids_per_sample[keep_mask][keep_idxes])

        inputs["instance_masks"] = pruned_instance_masks
        inputs["class_ids"] = pruned_class_ids
        inputs["ref_frame_index"] = [ref_frame_index for _ in range(len(pruned_instance_masks))]
        inputs.pop("semantic_masks")

        return inputs

    # @torch.no_grad()
    # def get_kitti_step_instance_cls_ignore_mask(self, pred_instance_masks: Tensor, gt_semantic_masks: Tensor,
    #                                             ioa_threshold: float = 0.5):
    #     # This function exists to solve a strange problem. We train on CityscapesVPS and KITTI-STEP which both share the
    #     # same class IDs, so we use the same set of semantic queries for both datasets. However, KITTI-STEP is weird in
    #     # that it only contains instance annotations for the 'car' and 'person' classes, and other thing classes
    #     # such as 'rider' ,'bicycle' and 'truck' are considered as stuff classes. In CityscapesVPS however, we
    #     # do have instance annotations for these classes. Thus, naively training with the given ground-truth will
    #     # result in instances of these classes being pulled towards background for KITTI-STEP samples, and towards
    #     # the correct class for CityscapesVPS. To resolve this inconsistency, we need to create an ignore mask for
    #     # KITTI-STEP samples which contains all pixels belonging to these false stuff classes (12, 14, 15,
    #     # 16, 17, 18). For KITTI-STEP samples, the instance classification loss for any instances with a large
    #     # intersection-over-area with this ignore mask will be ignored.

    #     # input shapes:
    #     # - pred_instance_masks: [L, B, Qi, T, H, W] (Qi = number of instance queries)
    #     # - gt_semantic_masks: [B, T, H, W]
    #     false_stuff_class_ids = [12, 14, 15, 16, 17, 18]
    #     # T, H, W = pred_instance_masks.shape[-3:]

    #     # resize semantic masks to predicted mask size
    #     gt_semantic_masks = F.interpolate(gt_semantic_masks.float(), scale_factor=0.25, mode='nearest-exact')
    #     assert gt_semantic_masks.shape[-3:] == pred_instance_masks.shape[-3:], \
    #         f"Shape mismatch: {gt_semantic_masks.shape}, {pred_instance_masks.shape}"

    #     pred_instance_masks = rearrange(pred_instance_masks > 0.0, "L B Q T H W -> L B Q (T H W)")
    #     ignore_semantic_mask = torch.any(torch.stack([gt_semantic_masks == cls_id for cls_id in false_stuff_class_ids]), 0)  # [B, T, H, W]
    #     ignore_semantic_mask = rearrange(ignore_semantic_mask.bool(), "B T H W -> 1 B 1 (T H W)")

    #     intersection = torch.logical_and(pred_instance_masks, ignore_semantic_mask).sum(3, dtype=torch.float32)  # [L, B, Q]
    #     area = pred_instance_masks.sum(3, dtype=torch.float32)  # [L, B, Q]

    #     ioa = intersection / area
    #     ioa = torch.where(torch.isinf(ioa), torch.zeros_like(ioa), ioa)
    #     mask = ioa > ioa_threshold
    #     # print(mask.sum().item(), mask.numel())
    #     return mask
