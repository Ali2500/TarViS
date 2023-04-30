from tarvis.modelling.backbone.convnext import ConvNeXt, LayerNorm
from einops import rearrange

from tarvis.modelling.backbone.temporal_neck.module import TemporalNeck
from detectron2.layers import ShapeSpec
from tarvis.config import cfg

import torch
import torch.nn as nn
import tarvis.utils.distributed as dist_utils


class Backbone(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.backbone = model

        fmap_keys = ["res2", "res3", "res4", "res5"]

        self.norm_layers = nn.ModuleDict({
            key: LayerNorm(dim, eps=1e-6, data_format="channels_first")
            for key, dim in zip(fmap_keys, model.output_dims)
        })

        backbone_output_shape = {
            name: ShapeSpec(
                channels=ch, stride=scale
            )
            for name, ch, scale in zip(fmap_keys, model.output_dims, [4, 8, 16, 32])
        }

        self.fpn = TemporalNeck(
            input_shape=backbone_output_shape,
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

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.normalize_image = True

    def forward(self, images):
        assert images.ndim == 5, f"images: {images.shape}"
        batch_sz, clip_len = images.shape[:2]

        images = rearrange(images, "B T C H W -> (B T) C H W")

        fmaps = self.backbone(images)
        fmaps = {k: self.norm_layers[k](f) for k, f in fmaps.items()}

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


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def load_state_dict(model, checkpoint):
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    assert len(missing) == 0
    allowed_unexpected = [
        "norm.weight", "norm.bias", "head.weight", "head.bias"
    ]
    assert all([x in allowed_unexpected for x in unexpected])


def load_checkpoint(url):
    if dist_utils.is_main_process():
        # download checkpoint on main process only
        _ = torch.hub.load_state_dict_from_url(url=url, progress=True)
    dist_utils.synchronize()
    return torch.hub.load_state_dict_from_url(url=url, map_location="cpu")


# @GlobalRegistry.register("Backbone", "ConvNextTiny")
def convnext_tiny(in_22k=True):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
    checkpoint = load_checkpoint(url)
    load_state_dict(model, checkpoint)
    return Backbone(model)


# @GlobalRegistry.register("Backbone", "ConvNextSmall")
def convnext_small(in_22k=True):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
    url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
    checkpoint = load_checkpoint(url)
    load_state_dict(model, checkpoint)
    return Backbone(model)


# @GlobalRegistry.register("Backbone", "ConvNextBase")
def convnext_base(in_22k=True):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
    checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
    load_state_dict(model, checkpoint)
    return Backbone(model)


# @GlobalRegistry.register("Backbone", "ConvNextLarge")
def convnext_large(in_22k=True):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
    url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
    checkpoint = load_checkpoint(url)
    load_state_dict(model, checkpoint)
    return Backbone(model)


# @GlobalRegistry.register("Backbone", "ConvNextXLarge")
def convnext_xlarge(in_22k=True):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
    assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
    url = model_urls['convnext_xlarge_22k']
    checkpoint = load_checkpoint(url)
    load_state_dict(model, checkpoint)
    return Backbone(model)
