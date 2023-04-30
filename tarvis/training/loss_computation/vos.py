from collections import defaultdict
from torch import Tensor
from einops import rearrange, repeat
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tarvis.modelling.point_sampling_utils import get_uncertain_point_coords_with_randomness, point_sample


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


# def uncertainty_fn_multiclass(logits: Tensor):
#     # logits: [N, C, ...]
#     top2 = logits.topk(k=2, dim=1).values
#     diffs = top2[:, 1] - top2[:, 0]
#     return diffs.unsqueeze(1)


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


class VOSLoss(nn.Module):
    def __init__(self, num_points: int, apply_bg_loss: bool, loss_type: str, w_dice: float, w_ce: float):
        super().__init__()

        self.w_dice = w_dice
        self.w_ce = w_ce

        self.num_points = num_points
        self.oversample_ratio = 3
        self.importance_sample_ratio = 0.75
        self.apply_bg_loss = apply_bg_loss

        assert loss_type in ("sparse", "dense")
        self.loss_type = loss_type
        self.max_dense_masks = 64

    def forward(self, pred_mask_logits: Tensor, gt_masks: List[Tensor]):
        """
        Forward method
        :param pred_mask_logits: tensor of shape [L, B, 1+num_instances, T, H, W]
        :param gt_masks: List of tensors, each of shape [num_instances, T, H, W]
        :return: loss dict
        """
        num_layers = pred_mask_logits.shape[0]

        assert len(set([m.size(0) for m in gt_masks])) == 1, f"All batch samples must have the same number of objects"
        gt_masks = torch.stack(gt_masks)  # [B, num_instances, T, H, W]

        if self.apply_bg_loss:
            with torch.no_grad():
                bg_mask = torch.logical_not(torch.any(gt_masks, 1, keepdim=True).bool())  # [B, 1, T, H, W]
                gt_masks = torch.cat((bg_mask, gt_masks), 1)  # [B, 1+num_instances, T, H, W]
        else:
            pred_mask_logits = pred_mask_logits[:, :, 1:]  # [L, B, num_instances, T, H, W]

        assert pred_mask_logits.shape[1:4] == gt_masks.shape[:3], \
            f"Shape mismatch: {pred_mask_logits.shape}, {gt_masks.shape}"

        # merge batch, time and class dims
        gt_masks = rearrange(gt_masks, "B C T H W -> (B C T) H W").to(pred_mask_logits.dtype)
        pred_mask_logits = rearrange(pred_mask_logits, "L B C T H W -> L (B C T) H W")

        if self.loss_type == "dense":
            if pred_mask_logits.size(1) > self.max_dense_masks:
                perm = torch.linspace(0, pred_mask_logits.size(1) - 1, self.max_dense_masks).round().long()
                pred_mask_logits = pred_mask_logits[:, perm]
                gt_masks = gt_masks[perm]

            pred_mask_logits = F.interpolate(pred_mask_logits, scale_factor=4.0, mode='bilinear', align_corners=False)
        else:
            gt_masks = gt_masks.unsqueeze(1)  # [B*C*T, 1, H, W]
            pred_mask_logits = pred_mask_logits.unsqueeze(2)  # [L, B*C*T,  1, H, W]

        outputs = defaultdict(list)

        for l in range(num_layers):
            if self.loss_type == "sparse":
                with torch.no_grad():
                    point_coords = get_uncertain_point_coords_with_randomness(
                        pred_mask_logits[l], calculate_uncertainty, self.num_points, self.oversample_ratio,
                        self.importance_sample_ratio
                    ).squeeze(1)  # [B*C*T, P]

                    point_labels = point_sample(
                        gt_masks, point_coords, align_corners=False
                    ).squeeze(1)  # [B*C*T, P]

                point_logits = point_sample(pred_mask_logits[l], point_coords, align_corners=False).squeeze(1)  # [B, P]

            else:
                point_logits = pred_mask_logits[l].flatten(1)  # [B*C*T, H*W]
                point_labels = gt_masks.flatten(1)  # [B*C*T, H*W]

            outputs["loss_ce"].append(F.binary_cross_entropy_with_logits(point_logits, point_labels))
            outputs["loss_dice"].append(dice_loss_jit(point_logits, point_labels))
            outputs["loss_total"].append((self.w_ce * outputs["loss_ce"][-1] + (self.w_dice * outputs["loss_dice"][-1])))

        return {
            k: torch.stack(v) for k, v in outputs.items()
        }


# @GlobalRegistry.register("VOSLosses", "per_object_sparse")
# class VOSPerObjectSparseLoss(VOSPerObjectLoss):
#     def __init__(self, num_points: int, apply_bg_loss: bool, w_dice: float = 5.0, w_ce: float = 5.0):
#         super().__init__(
#             num_points=num_points,
#             apply_bg_loss=apply_bg_loss,
#             loss_type="sparse",
#             w_dice=w_dice,
#             w_ce=w_ce
#         )


# @GlobalRegistry.register("VOSLosses", "per_object_dense")
# class VOSPerObjectSparseLoss(VOSPerObjectLoss):
#     def __init__(self, num_points: int, apply_bg_loss: bool, w_dice: float = 5.0, w_ce: float = 5.0):
#         super().__init__(
#             num_points=num_points,
#             apply_bg_loss=apply_bg_loss,
#             loss_type="dense",
#             w_dice=w_dice,
#             w_ce=w_ce
#         )


# @GlobalRegistry.register("VOSLosses", "per_object_bootstrapped")
# class VOSPerObjectBootstrappedLoss(nn.Module):
#     def __init__(self, num_points: int, apply_bg_loss: bool):
#         super().__init__()

#         self.w_dice = 5.0
#         self.w_ce = 5.0

#         self.bootstrap_factor = 0.25
#         self.num_points = num_points
#         self.apply_bg_loss = apply_bg_loss
#         self.max_dense_masks = 32

#     def forward(self, pred_mask_logits: Tensor, gt_masks: List[Tensor]):
#         """
#         Forward method
#         :param pred_mask_logits: tensor of shape [L, B, 1+num_instances, T, H, W]
#         :param gt_masks: List of tensors, each of shape [num_instances, T, H, W]
#         :return: loss dict
#         """
#         num_layers = pred_mask_logits.shape[0]

#         assert len(set([m.size(0) for m in gt_masks])) == 1, f"All batch samples must have the same number of objects"
#         gt_masks = torch.stack(gt_masks)  # [B, num_instances, T, H, W]

#         if self.apply_bg_loss:
#             with torch.no_grad():
#                 bg_mask = torch.logical_not(torch.any(gt_masks, 1, keepdim=True).bool())  # [B, 1, T, H, W]
#                 gt_masks = torch.cat((bg_mask, gt_masks), 1)  # [B, 1+num_instances, T, H, W]
#         else:
#             pred_mask_logits = pred_mask_logits[:, :, 1:]  # [L, B, num_instances, T, H, W]

#         assert pred_mask_logits.shape[1:4] == gt_masks.shape[:3], \
#             f"Shape mismatch: {pred_mask_logits.shape}, {gt_masks.shape}"

#         # merge batch, time and class dims
#         gt_masks = rearrange(gt_masks, "B N T H W -> (B N) (T H W)").to(pred_mask_logits.dtype)
#         pred_mask_logits = rearrange(pred_mask_logits, "L B N T H W -> L (B N) T H W")

#         if pred_mask_logits.size(1) > self.max_dense_masks:
#             perm = torch.linspace(0, pred_mask_logits.size(1) - 1, self.max_dense_masks).round().long()
#             pred_mask_logits = pred_mask_logits[:, perm]
#             gt_masks = gt_masks[perm]

#         pred_mask_logits = F.interpolate(pred_mask_logits, scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
#         pred_mask_logits = rearrange(pred_mask_logits, "L BN T H W -> L BN (T H W)")
#         assert pred_mask_logits.shape[1:] == gt_masks.shape, f"Shape mismatch: {pred_mask_logits.shape}, {gt_masks.shape}"

#         outputs = defaultdict(list)

#         for l in range(num_layers):
#             logits_l = pred_mask_logits[l]  # [B*N, T*H*W]

#             error_map = self.compute_logit_errors(logits_l, gt_masks)
#             topk_indices = error_map.topk(k=int(round(self.bootstrap_factor * logits_l.size(1))), dim=1,largest=True,
#                                           sorted=False).indices

#             logits_l = torch.gather(logits_l, 1, topk_indices)
#             labels_l = torch.gather(gt_masks, 1, topk_indices)

#             outputs["loss_ce"].append(F.binary_cross_entropy_with_logits(logits_l, labels_l))
#             outputs["loss_dice"].append(dice_loss_jit(logits_l, labels_l))
#             outputs["loss_total"].append((self.w_ce * outputs["loss_ce"][-1] + (self.w_dice * outputs["loss_dice"][-1])))

#         return {
#             k: torch.stack(v) for k, v in outputs.items()
#         }

#     @torch.no_grad()
#     def compute_logit_errors(self, logits: Tensor, gt: Tensor):
#         # logits and gt should have identical shape
#         assert logits.shape == gt.shape
#         return (logits.sigmoid() - gt).abs()


# class VOSMulticlassLoss(nn.Module):
#     def __init__(self, num_points: int, apply_bg_loss: bool, loss_type: str):
#         super().__init__()

#         self.w_dice = 5.0
#         self.w_ce = 2.0

#         self.num_points = num_points
#         self.oversample_ratio = 3
#         self.importance_sample_ratio = 0.75

#         assert apply_bg_loss
#         assert loss_type in ("sparse", "dense")
#         self.loss_type = loss_type
#         self.max_dense_masks = 64

#     def forward(self, pred_mask_logits: Tensor, gt_masks: List[Tensor]):
#         """
#         Forward method
#         :param pred_mask_logits: tensor of shape [L, B, 1+num_instances, T, H, W]
#         :param gt_masks: List of tensors, each of shape [num_instances, T, H, W]
#         :return: loss dict
#         """
#         num_layers, batch_sz, num_objs, clip_len = pred_mask_logits.shape[:4]
#         num_masks = batch_sz * num_objs * clip_len

#         assert len(set([m.size(0) for m in gt_masks])) == 1, f"All batch samples must have the same number of objects"
#         gt_masks = torch.stack(gt_masks)  # [B, num_instances, T, H, W]

#         with torch.no_grad():
#             bg_mask = torch.logical_not(torch.any(gt_masks, 1, keepdim=True).bool())  # [B, 1, T, H, W]
#             gt_masks = torch.cat((bg_mask, gt_masks), 1)  # [B, 1+num_instances, T, H, W]
#             gt_masks = gt_masks.to(torch.int64).argmax(1)  # [B, T, H, W]

#         # merge batch, time and class dims
#         gt_masks = rearrange(gt_masks, "B T H W -> (B T) H W")
#         pred_mask_logits = rearrange(pred_mask_logits, "L B N T H W -> L (B T) N H W")

#         if self.loss_type == "dense":
#             if num_masks > self.max_dense_masks:
#                 n_retain = max(1, self.max_dense_masks // num_objs)
#                 perm = torch.linspace(0, (batch_sz * clip_len) - 1, n_retain).round().long()
#                 pred_mask_logits = pred_mask_logits[:, perm]
#                 gt_masks = gt_masks[perm]

#             pred_mask_logits = F.interpolate(pred_mask_logits, scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
#         else:
#             gt_masks = gt_masks.unsqueeze(1).to(pred_mask_logits.dtype)  # [B*T, 1, H, W]

#         outputs = defaultdict(list)

#         for l in range(num_layers):
#             if self.loss_type == "sparse":
#                 with torch.no_grad():
#                     point_coords = get_uncertain_point_coords_with_randomness(
#                         pred_mask_logits[l], uncertainty_fn_multiclass, self.num_points, self.oversample_ratio,
#                         self.importance_sample_ratio
#                     ).squeeze(1)  # [B*T, P]

#                     point_labels = point_sample(
#                         gt_masks.float(), point_coords, align_corners=False
#                     ).squeeze(1).to(torch.int64)  # [B*T, P]

#                 point_logits = point_sample(pred_mask_logits[l], point_coords, align_corners=False)  # [B, N, P]

#             else:
#                 point_logits = pred_mask_logits[l]
#                 point_labels = gt_masks

#             outputs["loss_ce"].append(F.cross_entropy(point_logits, point_labels))
#             outputs["loss_dice"].append(multiclass_dice_loss(point_logits, point_labels, ignore_zero_class=True))

#             outputs["loss_total"].append((self.w_ce * outputs["loss_ce"][-1] + (self.w_dice * outputs["loss_dice"][-1])))

#         return {
#             k: torch.stack(v) for k, v in outputs.items()
#         }


# @GlobalRegistry.register("VOSLosses", "multi_object_sparse")
# class VOSMulticlassSparseLoss(VOSMulticlassLoss):
#     def __init__(self, num_points: int, apply_bg_loss: bool):
#         super().__init__(
#             num_points=num_points,
#             apply_bg_loss=apply_bg_loss,
#             loss_type="sparse"
#         )


# @GlobalRegistry.register("VOSLosses", "multi_object_dense")
# class VOSMulticlassDenseLoss(VOSMulticlassLoss):
#     def __init__(self, num_points: int, apply_bg_loss: bool):
#         super().__init__(
#             num_points=num_points,
#             apply_bg_loss=apply_bg_loss,
#             loss_type="dense"
#         )
