from collections import defaultdict
from torch import Tensor
from einops import rearrange, repeat
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tarvis.modelling.point_sampling_utils import get_uncertain_point_coords_with_randomness, point_sample
from tarvis.training.loss_computation.common import multiclass_dice_loss


# def calculate_uncertainty(logits):
#     """
#     We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
#         foreground class in `classes`.
#     Args:
#         logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
#             class-agnostic, where R is the total number of predicted masks in all images and C is
#             the number of foreground classes. The values are logits.
#     Returns:
#         scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
#             the most uncertain locations having the highest uncertainty score.
#     """
#     assert logits.shape[1] == 1
#     gt_class_logits = logits.clone()
#     return -(torch.abs(gt_class_logits))


# def dice_loss(
#         inputs: torch.Tensor,
#         targets: torch.Tensor,
# ):
#     """
#     Compute the DICE loss, similar to generalized IOU for masks
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     """
#     inputs = inputs.sigmoid()
#     inputs = inputs.flatten(1)
#     numerator = 2 * (inputs * targets).sum(-1)
#     denominator = inputs.sum(-1) + targets.sum(-1)
#     loss = 1 - (numerator + 1) / (denominator + 1)
#     return loss.mean()


# dice_loss_jit = torch.jit.script(
#     dice_loss
# )  # type: torch.jit.ScriptModule


# @GlobalRegistry.register("SemanticSegLosses", "per_class")
# class SemanticSegmentationLoss(nn.Module):
#     def __init__(self, w_dice: float, w_ce: float, num_points: int):
#         super().__init__()

#         assert w_dice > 0 and w_ce > 0
#         self.w_dice = w_dice
#         self.w_ce = w_ce

#         self.num_points = num_points
#         self.oversample_ratio = 3
#         self.importance_sample_ratio = 0.75

#     @staticmethod
#     @torch.no_grad()
#     def instance_masks_to_semantic_masks(instance_masks: List[Tensor], class_ids: List[Tensor]):
#         """
#         Convert instance masks to semantic class masks
#         :param instance_masks: list of masks of shape [N, T, H, W]
#         :param class_ids: list of class IDs, each of shape [N]
#         :return: semantic masks of shape [B, 1+present_classes, T, H, W]
#         """
#         batch_sz = len(instance_masks)
#         clip_len, height, width = instance_masks[0].shape[-3:]
#         assert len(instance_masks) == len(class_ids)

#         present_class_ids = torch.cat(class_ids).unique()  # [M]
#         semantic_masks = torch.zeros(batch_sz, 1 + present_class_ids.numel(), clip_len, height, width, dtype=torch.bool,
#                                      device=instance_masks[0].device)

#         for b in range(batch_sz):
#             assert instance_masks[b].ndim == 4

#             for idx, cls_id in enumerate(present_class_ids, 1):
#                 instance_masks_cls = instance_masks[b][class_ids[b] == cls_id]  # [M', T, H, W]
#                 if instance_masks_cls.numel() > 0:
#                     semantic_masks[b, idx] = torch.any(instance_masks_cls, 0)

#         # background mask
#         semantic_masks[:, 0] = torch.logical_not(torch.any(semantic_masks[:, 1:], 1))

#         present_class_ids = torch.cat([present_class_ids.new_zeros((1,)), present_class_ids])
#         return semantic_masks, present_class_ids

#     def forward(self, pred_mask_logits: Tensor, gt_instance_masks: List[Tensor], gt_classes: List[Tensor]):
#         """
#         Forward method
#         :param pred_mask_logits: tensor of shape [L, B, 1+num_classes, T, H, W]
#         :param gt_instance_masks: List of tensors, each of shape [num_instances, T, H, W]
#         :param gt_classes: list of tensors, each of shape [num_instances] with value denoting class IDs
#         `gt_instance_masks` and `gt_classes` will be ignored.
#         :return: loss dict
#         """
#         num_layers = pred_mask_logits.shape[0]

#         # if gt_semantic_masks is None:
#         gt_semantic_masks, present_class_ids = self.instance_masks_to_semantic_masks(gt_instance_masks, gt_classes)
#         # else:
#         #     gt_semantic_masks, present_class_ids = self.filter_absent_classes(gt_semantic_masks)

#         pred_mask_logits = pred_mask_logits[:, :, present_class_ids]

#         # merge batch, time and class dims
#         gt_semantic_masks = rearrange(gt_semantic_masks, "B C T H W -> (B C T) 1 H W").to(pred_mask_logits.dtype)
#         pred_mask_logits = rearrange(pred_mask_logits, "L B C T H W -> L (B C T) 1 H W")

#         outputs = defaultdict(list)

#         for l in range(num_layers):
#             with torch.no_grad():
#                 point_coords = get_uncertain_point_coords_with_randomness(
#                     pred_mask_logits[l], calculate_uncertainty, self.num_points, self.oversample_ratio,
#                     self.importance_sample_ratio
#                 ).squeeze(1)  # [B*C*T, P]

#                 point_labels = point_sample(
#                     gt_semantic_masks, point_coords, align_corners=False
#                 ).squeeze(1)  # [B*C*T, P]

#             point_logits = point_sample(pred_mask_logits[l], point_coords, align_corners=False).squeeze(1)  # [B, P]

#             outputs["loss_ce"].append(F.binary_cross_entropy_with_logits(point_logits, point_labels))
#             outputs["loss_dice"].append(dice_loss_jit(point_logits, point_labels))
#             outputs["loss_total"].append((self.w_ce * outputs["loss_ce"][-1] + (self.w_dice * outputs["loss_dice"][-1])))

#         return {
#             k: torch.stack(v) for k, v in outputs.items()
#         }


def uncertainty_fn_multiclass(logits: Tensor):
    # logits: [N, C, ...]
    top2 = logits.topk(k=2, dim=1).values
    diffs = top2[:, 1] - top2[:, 0]
    return diffs.unsqueeze(1)


class SemanticSegLoss(nn.Module):
    def __init__(self, w_dice: float, w_ce: float, num_points: int, ignore_null_class_logits: bool,
                 logit_regularization_loss: bool):
        super().__init__()

        self.w_dice = w_dice
        self.w_ce = w_ce
        self.w_reg = 1e-3

        self.num_points = num_points
        self.oversample_ratio = 3
        self.importance_sample_ratio = 0.75

        self.ignore_null_class_logits = ignore_null_class_logits
        self.logit_regularization_loss = logit_regularization_loss

    @staticmethod
    @torch.no_grad()
    def instance_masks_to_semantic_masks(instance_masks: List[Tensor], class_ids: List[Tensor]) -> Tensor:
        """
        Convert instance masks to semantic class masks
        :param instance_masks: list of masks of shape [N, T, H, W]
        :param class_ids: list of class IDs, each of shape [N]
        :return: semantic masks of shape [B, T, H, W]
        """
        batch_sz = len(instance_masks)
        clip_len, height, width = instance_masks[0].shape[-3:]
        assert len(instance_masks) == len(class_ids)

        semantic_masks = torch.zeros(len(instance_masks), clip_len, height, width, dtype=torch.int64,
                                     device=instance_masks[0].device)

        for b in range(batch_sz):
            assert instance_masks[b].ndim == 4

            # instances can be overlapping. Re-order them in descending order of mask area so that smaller objects
            # take priority over larger ones
            mask_areas = rearrange(instance_masks[b], "N T H W -> N (T H W)").sum(1, dtype=torch.int64)  # [N]
            sort_idx = mask_areas.argsort(descending=True)
            instance_masks_b = instance_masks[b][sort_idx]
            class_ids_b = class_ids[b][sort_idx]

            for mask, cls_id in zip(instance_masks_b.bool(), class_ids_b):
                semantic_masks[b] = torch.where(mask, cls_id, semantic_masks[b])

        return semantic_masks

    def forward(self, pred_mask_logits: Tensor, gt_instance_masks: List[Tensor], gt_classes: List[Tensor]):
        """
        Forward method
        :param pred_mask_logits: tensor of shape [L, B, 1+num_classes, T, H, W]
        :param gt_instance_masks: List of tensors, each of shape [num_instances, T, H, W]
        :param gt_classes: list of tensors, each of shape [num_instances] with value denoting class IDs
        `gt_instance_masks` and `gt_classes` will be ignored.
        :return: loss dict
        """
        num_layers = pred_mask_logits.shape[0]

        gt_semantic_masks = self.instance_masks_to_semantic_masks(gt_instance_masks, gt_classes)  # [B, H, W]

        gt_semantic_masks = gt_semantic_masks.to(pred_mask_logits.dtype)
        losses = defaultdict(list)

        # merge batch and time dimensions
        pred_mask_logits = rearrange(pred_mask_logits, "L B num_classes T H W -> L (B T) num_classes H W")
        gt_semantic_masks = rearrange(gt_semantic_masks, "B T H W -> (B T) 1 H W")

        for l in range(num_layers):
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    pred_mask_logits[l], uncertainty_fn_multiclass, self.num_points, self.oversample_ratio,
                    self.importance_sample_ratio
                )

                point_labels = point_sample(
                    gt_semantic_masks, point_coords, align_corners=False
                ).to(torch.int64).squeeze(1)  # [B*T, P]

            point_logits = point_sample(pred_mask_logits[l], point_coords, align_corners=False)  # [B*T, C, P]

            assert not torch.any(torch.logical_or(point_labels < 0, point_labels >= point_logits.size(1)))

            if self.ignore_null_class_logits:
                point_logits = self.eliminate_null_class_logits(point_logits, point_labels)

            loss_ce = F.cross_entropy(point_logits, point_labels)
            losses["loss_ce"].append(loss_ce)

            if self.logit_regularization_loss:
                loss_reg = self.compute_logit_regularization_loss(point_logits)
                losses["loss_reg"].append(loss_reg)
            else:
                loss_reg = 0.0

            if self.w_dice > 0:
                loss_dice = multiclass_dice_loss(point_logits, point_labels)
                losses["loss_dice"].append(loss_dice)
            else:
                loss_dice = 0.

            losses["loss_total"].append(
                (self.w_ce * loss_ce) + (self.w_dice * loss_dice) + (self.w_reg * loss_reg)
            )

        return {
            k: torch.stack(v) for k, v in losses.items()
        }
        
    def eliminate_null_class_logits(self, pred: Tensor, gt: Tensor):
        # pred: [B*T, num_classses, P]
        # gt: [B*T, P]
        num_classes = pred.size(1)
        present_class_ids = gt.unique().tolist()
        absent_class_ids = [iid for iid in range(num_classes) if iid not in present_class_ids and iid >= 0]
        if len(absent_class_ids) == 0:
            return pred
            
        with torch.no_grad():
            pred_labels = pred.max(1).indices
            pred_labels_count_absent_classes = torch.stack([(pred_labels == iid).sum(dtype=torch.long) for iid in absent_class_ids])
            
            absent_class_ids = torch.tensor(absent_class_ids).to(pred.device)
            null_classes = absent_class_ids[pred_labels_count_absent_classes == 0]
        
        if null_classes.numel() > 0:
            pred[:, null_classes] = -1000
            
        return pred

    @torch.cuda.amp.autocast(enabled=False)
    def compute_logit_regularization_loss(self, pred_logits: Tensor):
        # pred_logits: [B*T, C, P]
        norm = torch.linalg.norm(pred_logits, dim=1, keepdim=True)
        return norm.mean()
