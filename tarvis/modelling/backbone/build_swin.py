from tarvis.modelling.backbone.swin import D2SwinTransformer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.layers import ShapeSpec
from einops import rearrange

from tarvis.modelling.backbone.temporal_neck.module import TemporalNeck

from tarvis.utils.paths import Paths

import logging
import torch.nn as nn
import os.path as osp


def build_cfg(variant: str):
    _C = CN()
    _C.MODEL = CN()

    _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.PATCH_SIZE = 4

    _C.MODEL.SWIN.MLP_RATIO = 4.0
    _C.MODEL.SWIN.QKV_BIAS = True
    _C.MODEL.SWIN.QK_SCALE = None
    _C.MODEL.SWIN.DROP_RATE = 0.0
    _C.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    _C.MODEL.SWIN.DROP_PATH_RATE = 0.3
    _C.MODEL.SWIN.APE = False
    _C.MODEL.SWIN.PATCH_NORM = True
    _C.MODEL.SWIN.USE_CHECKPOINT = False
    _C.MODEL.SWIN.OUT_FEATURES = [
        "res2",
        "res3",
        "res4",
        "res5"
      ]

    append_variant_specific_cfg(_C, variant)
    return _C


def append_variant_specific_cfg(d2_cfg: CN, variant: str):
    if variant == "tiny":
        d2_cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
        d2_cfg.MODEL.SWIN.EMBED_DIM = 96
        d2_cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
        d2_cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
        d2_cfg.MODEL.SWIN.WINDOW_SIZE = 7

    elif variant == "small":
        d2_cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
        d2_cfg.MODEL.SWIN.EMBED_DIM = 96
        d2_cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
        d2_cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
        d2_cfg.MODEL.SWIN.WINDOW_SIZE = 7

    elif variant == "base":
        d2_cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 384
        d2_cfg.MODEL.SWIN.EMBED_DIM = 128
        d2_cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
        d2_cfg.MODEL.SWIN.NUM_HEADS = [4, 8, 16, 32]
        d2_cfg.MODEL.SWIN.WINDOW_SIZE = 12

    elif variant == "large":
        d2_cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 384
        d2_cfg.MODEL.SWIN.EMBED_DIM = 192
        d2_cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
        d2_cfg.MODEL.SWIN.NUM_HEADS = [6, 12, 24, 48]
        d2_cfg.MODEL.SWIN.WINDOW_SIZE = 12

    else:
        raise ValueError(f"Invalid variant: {variant}")


CHECKPOINT_FILENAME = {
    "tiny": "swin_tiny_patch4_window7_224.pkl",
    "small": "swin_small_patch4_window7_224.pkl",
    "base": "swin_base_patch4_window12_384_22k.pkl",
    "large": "swin_large_patch4_window12_384_22k.pkl"
}


class Backbone(nn.Module):
    def __init__(self, d2_cfg: CN, checkpoint_filename: str):
        super().__init__()

        input_shape = ShapeSpec(channels=3)
        model = D2SwinTransformer(d2_cfg, input_shape)

        checkpoint_path = osp.join(Paths.pretrained_backbones_dir(), checkpoint_filename)

        checkpointer = DetectionCheckpointer(model)
        checkpointer.logger.setLevel(logging.ERROR)
        checkpointer.resume_or_load(checkpoint_path, resume=True)

        self.backbone = model

        self.fpn = TemporalNeck(
            input_shape=model.output_shape(),
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


def build_swin_tiny_backbone():
    d2_cfg = build_cfg("tiny")
    return Backbone(d2_cfg, CHECKPOINT_FILENAME["tiny"])


def build_swin_small_backbone():
    d2_cfg = build_cfg("small")
    return Backbone(d2_cfg, CHECKPOINT_FILENAME["small"])


def build_swin_base_backbone():
    d2_cfg = build_cfg("base")
    return Backbone(d2_cfg, CHECKPOINT_FILENAME["base"])


def build_swin_large_backbone():
    d2_cfg = build_cfg("large")
    return Backbone(d2_cfg, CHECKPOINT_FILENAME["large"])
