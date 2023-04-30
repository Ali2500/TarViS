from collections import defaultdict
from einops import rearrange
from typing import List, Optional
from torch import Tensor
from tarvis.training.loss_computation.instance_seg import InstanceSegLoss
from tarvis.modelling.point_sampling_utils import get_uncertain_point_coords_with_randomness, point_sample
from tarvis.training.loss_computation.common import multiclass_dice_loss, bootstrapped_cross_entropy

import torch
import torch.nn as nn
import torch.nn.functional as F


class PanopticSegLoss(nn.Module):
    def __init__(self, num_points_instance, num_points_semantic, oversample_ratio, importance_sample_ratio,
                 semantic_loss_type, semantic_w_ce, semantic_w_dice, ignore_null_class_logits,
                 semantic_logit_regularization_loss):
        super().__init__()

        self.instance_seg_loss = InstanceSegLoss(
            bg_coef=0.1, num_points=num_points_instance, oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio
        )

        self.semantic_seg_loss = SemanticSegLoss(
            num_points=num_points_semantic, oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio, loss_type=semantic_loss_type,
            w_ce=semantic_w_ce, w_dice=semantic_w_dice, ignore_null_class_logits=ignore_null_class_logits,
            logit_regularization_loss=semantic_logit_regularization_loss
        )

    def forward(self, pred_instance_mask_logits: Tensor, gt_instance_masks: List[Tensor],
                pred_instance_class_logits: Tensor, gt_instance_classes: List[Tensor],
                pred_semantic_mask_logits: Tensor, gt_semantic_masks: Tensor,
                ignore_masks: Tensor, instance_cls_ignore_mask: Optional[Tensor] = None):
        """
        Forward method
        :param pred_instance_mask_logits: tensor of shape [L, B, Qf, H, W] or [L, B, Qf, T, H, W]
        :param gt_instance_masks: List of masks for each batch samples of shape [N, H, W]
        :param pred_instance_class_logits: tensor of shape [L, B, Qf, 1+num_classes]
        :param gt_instance_classes: list of tensors, each containing the ground-truth class labels for the instances of shape [N]
        :param pred_semantic_mask_logits: tensor of shape [B, 1+num_classes, T, H, W]
        :param gt_semantic_masks: tensor of shape [B, num_classes, T, H, W].
        :param ignore_masks: tensor of shape [B, T, H, W] with ignore regions for semantic seg
        :param instance_cls_ignore_mask: optional tensor of shape [L, B, Qf] for masking instance classification loss
        :return: loss dict
        """
        instance_losses = self.instance_seg_loss(
            pred_mask_logits=pred_instance_mask_logits, gt_masks=gt_instance_masks,
            pred_class_logits=pred_instance_class_logits, gt_classes=gt_instance_classes,
            cls_ignore_mask=instance_cls_ignore_mask
        )

        semantic_losses = self.semantic_seg_loss(
            pred_mask_logits=pred_semantic_mask_logits, gt_masks=gt_semantic_masks, ignore_masks=ignore_masks
        )

        # ensure no duplicate keys
        assert all([k not in semantic_losses for k in instance_losses.keys()])

        key_mapping = {
            "loss_mask_dice": "loss_instance_mask_dice",
            "loss_mask_ce": "loss_instance_mask_ce",
            "loss_cls": "loss_instance_cls",
            "loss_total": "loss_instance_total"
        }

        combined_losses = {key_mapping[k]: v for k, v in instance_losses.items()}
        combined_losses.update(semantic_losses)
        combined_losses["loss_total"] = instance_losses["loss_total"] + semantic_losses["loss_semantic_total"]

        return combined_losses


def condense_semantic_masks(masks: Tensor):
    # masks: [B, num_classes, T, H, W]
    condensed_mask = torch.zeros(masks.shape[1:])
    for cls_id, cls_mask in enumerate(masks):
        condensed_mask = torch.where(cls_mask, cls_id, condensed_mask)
    return condensed_mask


def uncertainty_fn(logits: Tensor):
    # logits: [N, 1+C, ...]. 1st channel for dim1 is the ignore mask
    ignore_mask, logits = logits.split((1, logits.size(1) - 1), 1)
    ignore_mask = ignore_mask.bool()

    top2 = logits.topk(k=2, dim=1, largest=True, sorted=True).values
    uncertainty = top2[:, 1] - top2[:, 0]  # [N, ...]
    uncertainty = uncertainty.unsqueeze(1)  # [N, 1, ....]

    # assign very low uncertainty to points which are supposed to be ignored
    uncertainty = torch.where(ignore_mask, torch.full_like(uncertainty, -1e3), uncertainty)
    return uncertainty


class SemanticSegLoss(nn.Module):
    def __init__(self, num_points, oversample_ratio, importance_sample_ratio, loss_type, w_ce, w_dice,
                 ignore_null_class_logits, logit_regularization_loss):
        super().__init__()

        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_reg = 1e-3

        assert loss_type in ("sparse", "dense", "dense_full_res")
        self.type = loss_type  # sparse or dense

        self.ignore_null_class_logits = ignore_null_class_logits
        self.logit_regularization_loss = logit_regularization_loss

    def forward(self, pred_mask_logits: Tensor, gt_masks: Tensor, ignore_masks: Tensor):
        """
        Forward method
        :param pred_mask_logits: tensor of shape [L, B, num_classes, T, H, W]
        :param gt_masks: tensor of shape [B, T, H, W] with pixel value denoting class IDs
        :param ignore_masks: tensor of shape [B, T, H, W] of type bool
        :return: loss dict
        """
        if self.type == "sparse":
            return self.forward_sparse(pred_mask_logits, gt_masks, ignore_masks)
        else:
            assert self.type in ("dense", "dense_full_res")
            return self.forward_dense(pred_mask_logits, gt_masks, ignore_masks, self.type == "dense_full_res")

    def forward_dense(self, pred_mask_logits: Tensor, gt_masks: Tensor, ignore_masks: Tensor, full_res: bool):
        # up-sampling logits to full res is too memory intensive. Instead we down-sample the gt by 4x to match the
        # pred logits size.
        gt_masks = torch.where(ignore_masks, -100, gt_masks)

        if not full_res:
            gt_masks = F.interpolate(gt_masks.float(), scale_factor=0.25, mode='nearest-exact').long()
            assert pred_mask_logits.shape[-3:] == gt_masks.shape[-3:], \
                f"Shape mismatch: {pred_mask_logits.shape}, {gt_masks.shape}"

        losses = defaultdict(list)

        for l in range(pred_mask_logits.size(0)):
            pred_l = pred_mask_logits[l]  # [B, num_classes, T, H, W]
            if full_res:
                pred_l = F.interpolate(pred_l, scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
                assert pred_l.shape[-3:] == gt_masks.shape[-3:], f"Shape mismatch: {pred_l.shape}, {gt_masks.shape}"

            loss_ce = bootstrapped_cross_entropy(pred_l, gt_masks, alpha=0.25, ignore_index=-100)

            if torch.isnan(loss_ce) or torch.isinf(loss_ce):
                print(f"Logits nan: {torch.any(torch.isnan(pred_mask_logits))}")
                print(f"Logits inf: {torch.any(torch.isinf(pred_mask_logits))}")

            losses["loss_semantic_ce"].append(loss_ce)

            if self.w_dice > 0:
                loss_dice = multiclass_dice_loss(pred_l, gt_masks)
                losses["loss_semantic_dice"].append(loss_dice)
            else:
                loss_dice = 0.

            losses["loss_semantic_total"].append(
                (self.w_ce * loss_ce) + (self.w_dice * loss_dice)
            )

        return {
            k: torch.stack(v) for k, v in losses.items()
        }

    def forward_sparse(self, pred_mask_logits: Tensor, gt_masks: Tensor, ignore_masks: Tensor):
        num_layers = pred_mask_logits.size(0)
        losses = defaultdict(list)

        # self.debug_viz_gt(gt_masks, ignore_masks)

        # merge batch and time dimensions
        pred_mask_logits = rearrange(pred_mask_logits, "L B num_classes T H W -> L (B T) num_classes H W")
        gt_masks = rearrange(gt_masks, "B T H W -> (B T) 1 H W")
        ignore_masks = rearrange(ignore_masks, "B T H W -> (B T) 1 H W")

        # resize ignore mask to image size of pred_mask_logits
        ignore_masks_ds = F.interpolate(
            ignore_masks.type_as(pred_mask_logits), scale_factor=0.25, mode='bilinear', align_corners=False
        )
        ignore_masks_ds = (ignore_masks_ds > 0.5).type_as(pred_mask_logits).detach()
        assert ignore_masks_ds.shape[-2:] == pred_mask_logits.shape[-2:], \
            f"Shape mismatch: {ignore_masks.shape}, {pred_mask_logits.shape}"

        for l in range(num_layers):
            with torch.no_grad():
                concat_ignore_mask_logits = torch.cat((ignore_masks_ds, pred_mask_logits[l]), 1)
                point_coords = get_uncertain_point_coords_with_randomness(
                    concat_ignore_mask_logits, uncertainty_fn, self.num_points, self.oversample_ratio,
                    self.importance_sample_ratio
                )
                
                point_labels = point_sample(gt_masks.float(), point_coords, mode='nearest', align_corners=False).long().squeeze(1)  # [B*T, P]
                point_ignore = point_sample(ignore_masks.float(), point_coords, mode='nearest', align_corners=False).bool().squeeze(1)  # [B*T, P]

            point_logits = point_sample(pred_mask_logits[l], point_coords, align_corners=False)  # [B*T, C, P]

            # Weird shit can happen when working on so many datasets. Let's ensure that the GT class Ids are not
            # out-of-bounds. -100 is the ignore class label so that's fine, but all other class values should be in
            # the range [0, num_classes-1]
            gt_class_ids = point_labels.unique()
            gt_class_ids = gt_class_ids[gt_class_ids != -100]
            assert not torch.any(torch.logical_or(gt_class_ids < 0, gt_class_ids >= point_logits.size(1))), \
                f"logits shape: {point_logits.shape}, Unique labels: {point_labels.unique().tolist()}"

            point_labels = torch.where(point_ignore, torch.full_like(point_labels, -100), point_labels)

            if self.ignore_null_class_logits:
                point_logits = self.eliminate_null_class_logits(point_logits, point_labels)
                
            loss_ce = F.cross_entropy(point_logits, point_labels, ignore_index=-100)
            losses["loss_semantic_ce"].append(loss_ce)

            if self.logit_regularization_loss:
                loss_reg = self.compute_logit_regularization_loss(point_logits)
                losses["loss_semantic_reg"].append(loss_reg)
            else:
                loss_reg = 0.0

            if self.w_dice > 0:
                loss_dice = multiclass_dice_loss(point_logits, point_labels)
                losses["loss_semantic_dice"].append(loss_dice)
            else:
                loss_dice = 0.

            losses["loss_semantic_total"].append(
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

    def debug_viz_gt(self, gt_masks, ignore_masks):
        from tarvis.inference.dataset_parser.kitti_step import KittiStepParser
        import cv2
        # gt_masks: [B, T, H, W]
        # ignore_masks: [B, T, H, W]
        gt_masks = gt_masks[0]
        ignore_masks = ignore_masks[0]
        labels = KittiStepParser._category_labels
        T = gt_masks.size(0)

        for t in range(T):
            cv2.namedWindow(f"Image {t}", cv2.WINDOW_NORMAL)
            cv2.namedWindow(f"ignore {t}", cv2.WINDOW_NORMAL)

        ignore_masks = ignore_masks.byte().cpu().numpy() * 255
        for t in range(T):
            cv2.imshow(f"ignore {t}", ignore_masks[t])

        for cls_id, label in labels.items():
            m = (gt_masks == cls_id).byte().cpu().numpy()
            print(label)
            for t, m_t in enumerate(m):
                cv2.imshow(f"Image {t}", m_t * 255)

            cv2.waitKey(0)
