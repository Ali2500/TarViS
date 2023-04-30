# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict
import torch.distributed

from torch import Tensor
from typing import List, Dict, Optional
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from tarvis.training.loss_computation.instance_matcher import HungarianMatcher
from tarvis.utils import distributed as dist_utils
# from tarvis.training.misc_m2f import nested_tensor_from_tensor_list
# from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
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
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


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


class InstanceSegLoss(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, bg_coef, num_points, oversample_ratio, importance_sample_ratio):
        super().__init__()

        self.matcher = HungarianMatcher(
            cost_class=2.0, cost_mask=5.0, cost_dice=5.0, num_points=num_points
        )
        self.weight_dict = {
            "loss_mask_ce": 5.0,
            "loss_mask_dice": 5.0,
            "loss_cls": 2.0
        }

        self.bg_coef = bg_coef

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks, ignore_mask):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        ignore_mask: None or tensor of shape [B, Q]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  # [B, Q, 1+num_classes]
        num_classes_p1 = src_logits.size(2)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)

        if ignore_mask is not None:
            target_classes = torch.where(ignore_mask, torch.full_like(target_classes, -100), target_classes)

        target_classes[idx] = target_classes_o

        with torch.no_grad():
            loss_weights = torch.ones(num_classes_p1, dtype=torch.float32, device=src_logits.device, requires_grad=False)
            loss_weights[0] = self.bg_coef

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, loss_weights, ignore_index=-100)
        losses = {"loss_cls": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]

        assert masks[0].ndim == 4, f"Invalid masks shape: {masks[0].shape}"
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_ce": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_mask_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, pred_mask_logits: Tensor, gt_masks: List[Tensor], pred_class_logits: Tensor,
                gt_classes: List[Tensor], cls_ignore_mask: Optional[Tensor] = None):
        """
        Forward method
        :param pred_mask_logits: tensor of shape [L, B, Qf, H, W] or [L, B, Qf, T, H, W]
        :param gt_masks: List of masks for each batch samples of shape [N, H, W]
        :param pred_class_logits: tensor of shape [L, B, Qf, 1+num_classes]
        :param gt_classes: list of tensors, each containing the ground-truth class labels for the instances of shape [N]
        :param cls_ignore_mask: optional tensor of shape [L, B, Qf]. Instance classification loss for instances will
        be ignored wherever cls_ignore_mask is True.
        :return: loss dict
        """
        num_layers, batch_sz = pred_mask_logits.shape[:2]

        if cls_ignore_mask is None:
            cls_ignore_mask = [None for _ in range(num_layers)]
        else:
            assert cls_ignore_mask.shape == pred_mask_logits.shape[:3], \
                f"Shape mismatch: {cls_ignore_mask.shape}, {pred_mask_logits.shape}"

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum([len(gt_classes_per_sample) for gt_classes_per_sample in gt_classes])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=pred_mask_logits.device
        )
        if dist_utils.is_distributed():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / dist_utils.get_world_size(), min=1).item()

        num_masks *= pred_mask_logits.size(3)  # multiply with clip length

        outputs = defaultdict(list)

        gt_targets = [
            {
                "labels": gt_classes_per_sample,
                "masks": gt_masks_per_sample
            }
            for gt_classes_per_sample, gt_masks_per_sample in zip(gt_classes, gt_masks)
        ]

        for l in range(num_layers):
            pred_outputs = {
                    "pred_masks": pred_mask_logits[l],
                    "pred_logits": pred_class_logits[l]
            }

            indices = self.matcher(outputs=pred_outputs, targets=gt_targets)

            losses = self.loss_masks(
                outputs=pred_outputs, targets=gt_targets, indices=indices, num_masks=num_masks
            )

            losses.update(self.loss_labels(
                outputs=pred_outputs, targets=gt_targets, indices=indices, num_masks=num_masks,
                ignore_mask=cls_ignore_mask[l]
            ))

            outputs["loss_total"].append(sum([losses[k] * self.weight_dict[k] for k in losses.keys()]))

            for k, loss in losses.items():
                outputs[k].append(loss)

        return {
            k: torch.stack(v) for k, v in outputs.items()
        }

    def __repr__(self):
        head = "Instance Segmentation Loss " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "weight_dict: {}".format(self.weight_dict),
            "bg_coef: {}".format(self.bg_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
