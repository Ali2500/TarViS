from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.layers import ShapeSpec
from einops import rearrange

from tarvis.modelling.backbone.temporal_neck.module import TemporalNeck
from tarvis.config import cfg

import logging
import torch.nn as nn
import os.path as osp


def create_default_cfg():
    _C = CN()
    _C.VERSION = 2

    _C.MODEL = CN()
    _C.MODEL.WEIGHTS = ""

    _C.MODEL.RESNETS = CN()
    _C.MODEL.BACKBONE = CN()

    _C.MODEL.BACKBONE.FREEZE_AT = 0

    _C.MODEL.RESNETS.DEPTH = 50
    _C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone

    # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
    _C.MODEL.RESNETS.NUM_GROUPS = 1

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.RESNETS.NORM = "FrozenBN"

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    _C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

    # Place the stride 2 conv on the 1x1 filter
    # Use True only for the original MSRA ResNet; use False for C2 and Torch models
    _C.MODEL.RESNETS.STRIDE_IN_1X1 = True

    # Apply dilation in stage "res5"
    _C.MODEL.RESNETS.RES5_DILATION = 1

    # Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
    # For R18 and R34, this needs to be set to 64
    _C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    _C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on Res2, Res3, Res4, Res5
    _C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
    # Use False for DeformableV1.
    _C.MODEL.RESNETS.DEFORM_MODULATED = False
    # Number of groups in deformable conv.
    _C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1

    return _C


class Backbone(nn.Module):
    def __init__(self, cfg_path, pretrained=True):
        super().__init__()

        d2_cfg = create_default_cfg()
        d2_cfg.merge_from_file(cfg_path)

        input_shape = ShapeSpec(channels=3)
        model = build_resnet_backbone(d2_cfg, input_shape)
        if pretrained:
            checkpointer = DetectionCheckpointer(model)
            checkpointer.logger.setLevel(logging.ERROR)
            checkpointer.resume_or_load(d2_cfg.MODEL.WEIGHTS, resume=True)
            # DetectionCheckpointer(model).resume_or_load(d2_cfg.MODEL.WEIGHTS, resume=True)

        self.backbone = model

        input_shape = {
            "res2": ShapeSpec(channels=256, stride=4),
            "res3": ShapeSpec(channels=512, stride=8),
            "res4": ShapeSpec(channels=1024, stride=16),
            "res5": ShapeSpec(channels=2048, stride=32)
        }

        self.fpn = TemporalNeck(
            input_shape=input_shape,
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm='GN',
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
            temporal_attn_patches_per_dim=8
        )

        self.image_mean = [123.675, 116.28, 103.53]
        self.image_std = [58.395, 57.12, 57.375]
        self.normalize_image = False

    def forward(self, images):
        assert images.ndim == 5, f"images: {images.shape}"
        batch_sz, clip_len = images.shape[:2]

        images = rearrange(images, "B T C H W -> (B T) C H W")

        fmaps = self.backbone(images)

        if self.fpn.is_3d:
            fmaps = {
                key: rearrange(f, "(B T) C H W -> B T C H W", B=batch_sz, T=clip_len)
                for key, f in fmaps.items()
            }

        fmaps = self.fpn(fmaps)
        fmaps = [rearrange(f, "(B T) C H W -> B T C H W", B=batch_sz, T=clip_len) for f in fmaps]

        return {
            "backbone_features": None,
            "output_features": fmaps
        }


def build_resnet50_backbone():
    cfg_path = osp.join(osp.dirname(__file__), "d2_configs", "resnet50.yaml")
    # return _build(cfg_path)
    return Backbone(cfg_path)


def build_resnet101_backbone():
    cfg_path = osp.join(osp.dirname(__file__), "d2_configs", "resnet101.yaml")
    return Backbone(cfg_path)
