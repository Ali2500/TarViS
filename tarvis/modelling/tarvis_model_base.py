from typing import List, Union, Optional, Dict
from einops import repeat
from torch import Tensor

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tarvis.config import cfg
from tarvis.modelling.backbone import backbone_builder
from tarvis.modelling.embedding_container import EmbeddingContainer
from tarvis.modelling.transformer_decoder import TransformerDecoder
from tarvis.modelling.vos_pet_object_encoder import VOSPETObjectEncoder


class TarvisModelBase(nn.Module):
    def __init__(self, semantic_query_group_sizes: Dict[str, int], num_obj_queries: int, num_bg_queries: int):
        super().__init__()

        self.backbone = backbone_builder[cfg.MODEL.BACKBONE]()

        self.embedding_container = EmbeddingContainer(
            num_dims=256, 
            num_classes=semantic_query_group_sizes, 
            num_objs=num_obj_queries, 
            num_bg=num_bg_queries,
            # shared_cityscapes_vps_queries=self.shared_semseg_queries_for_cityscapes
        )

        self.decoder = TransformerDecoder(
            in_channels=256, 
            hidden_dim=256, 
            nheads=8, 
            dim_feedforward=2048, 
            dec_layers=9,
            pre_norm=False, 
            enforce_input_project=False, 
            cross_attention_type=cfg.MODEL.CROSS_ATTENTION_TYPE
        )

        self.vos_query_extractor = VOSPETObjectEncoder(
            num_dims=256, 
            num_layers=4, 
            bg_grid_size=(4, 4),
            num_queries_per_object=cfg.MODEL.NUM_VOS_QUERIES_PER_OBJECT,
            backprop_through_fmap=False, 
            point_vos_likelihood=cfg.TRAINING.POINT_VOS_TASK_SAMPLE_LIKELIHOOD
        )

        self.register_buffer("pixel_mean", torch.tensor(self.backbone.image_mean))  # in RGB format
        self.register_buffer("pixel_std", torch.tensor(self.backbone.image_std))

    @property
    def fmap_multiple_of(self):
        return getattr(self.backbone.fpn, "fmap_multiple_of", 32)

    @property
    def num_vos_queries_per_object(self):
        return self.vos_query_extractor.num_queries_per_object

    def get_query_group_names(self, task_type):
        if task_type == "vos":
            return ["background_vos", "vos"]

        elif task_type in ("instance_seg", "panoptic_seg"):
            return ["instance", "background", "semantic"]

        else:
            raise ValueError(f"Invalid task type: {task_type}")

    def get_query_set(self, query_group_names: List[str], batch_sz: int, dataset_name: str,
                      vos_query_init: Optional[Tensor] = None, vos_bg_query_init: Optional[Tensor] = None):
        query_embeds = []
        query_inits = []
        query_counts = []

        def repeat_batch(x):
            assert x.ndim == 2
            return repeat(x, "Q C -> B Q C", B=batch_sz)

        for name in query_group_names:
            if name == "background":
                embed, init = self.embedding_container.background_queries

            elif name == "instance":
                embed, init = self.embedding_container.instance_queries

            elif name == "semantic":
                embed, init = self.embedding_container.semantic_queries(dataset_name)

            elif name == "background_vos":
                embed, init = self.embedding_container.background_vos_queries

                if vos_bg_query_init is not None:
                    num_bg_queries = vos_bg_query_init.shape[1]
                    assert embed.size(0) == 1
                    embed = repeat(embed.squeeze(0), "C -> B N C", B=batch_sz, N=num_bg_queries)
                    query_embeds.append(embed)
                    query_inits.append(vos_bg_query_init)
                    query_counts.append(num_bg_queries)
                    continue

            elif name == "vos":
                assert vos_query_init is not None
                num_objects = vos_query_init.shape[1]
                embed, init = self.embedding_container.vos_queries(num_objects, vos_query_init)  # [Q, C]

                query_embeds.append(repeat_batch(embed))
                query_inits.append(init)
                query_counts.append(num_objects)
                continue

            else:
                raise ValueError(f"Invalid query type: {name}")

            query_embeds.append(repeat_batch(embed))
            query_inits.append(repeat_batch(init))
            query_counts.append(embed.shape[0])

        return {
            "embeds": torch.cat(query_embeds, 1),
            "inits": torch.cat(query_inits, 1),
            "counts": query_counts
        }

    def compute_padded_dims(self, height, width):
        padded_width = (int(math.ceil(width / float(self.fmap_multiple_of))) * self.fmap_multiple_of)
        padded_height = (int(math.ceil(height / float(self.fmap_multiple_of))) * self.fmap_multiple_of)
        return padded_height, padded_width

    def pad_tensors(self, tensors: Union[Tensor, List[Tensor]], pad_value: int, stack: Union[None, bool],
                    return_padding: bool = False):

        if torch.is_tensor(tensors):
            tensors = [tensors]
            return_tensor = True
            assert stack is None
        else:
            return_tensor = False

        max_height = max([x.shape[-2] for x in tensors])
        max_width = max([x.shape[-1] for x in tensors])
        padded_height, padded_width = self.compute_padded_dims(max_height, max_width)
        padded_images = []

        for x in tensors:
            pad_right = padded_width - x.shape[-1]
            pad_bottom = padded_height - x.shape[-2]

            # handle variable number of dimensions in tensor. Allow any tensor with at least 2 dims.
            assert x.ndim >= 2, f"Image tensor must have at least 2 dimensions, but got tensor of shape {x.shape}"
            non_spatial_shape = x.shape[:-2]

            if x.ndim < 4:
                num_padded_dims = 4 - x.ndim
                view_shape = [1 for _ in range(num_padded_dims)] + list(x.shape)
                x = x.view(*view_shape)

            elif x.ndim > 4:
                num_compressed_dims = x.ndim - 4
                x = x.flatten(0, num_compressed_dims)

            assert x.ndim == 4
            x = F.pad(x, (0, pad_right, 0, pad_bottom), mode='constant', value=pad_value)
            padded_images.append(x.reshape(list(non_spatial_shape) + list(x.shape[-2:])))

        if return_tensor:
            assert len(padded_images) == 1
            padded_images = padded_images[0]
        else:
            assert stack is not None
            if stack:
                padded_images = torch.stack(padded_images, 0)

        if return_padding:
            return padded_images, (pad_right, pad_bottom)
        else:
            return padded_images

    def condense_vos_masks(self, vos_masks: Tensor, query_dim: int, object_query_counts: List[int]):
        assert sum(object_query_counts) == vos_masks.size(query_dim), \
            f"Mismatch: {object_query_counts}, {vos_masks.shape}, {query_dim}"

        vos_masks = vos_masks.split(object_query_counts, query_dim)
        vos_masks = torch.cat([
            masks_per_object.max(query_dim, keepdim=True).values for masks_per_object in vos_masks
        ], query_dim)
        return vos_masks

    def resize_vos_ref_masks_for_attention_masking(self, masks: Tensor) -> Dict[int, Tensor]:
        # masks: [B, N, T, H, W]
        masks = masks.float()
        masks = F.interpolate(masks, scale_factor=(1.0, 0.25, 0.25), mode='trilinear', align_corners=False) > 0.0

        # resize in steps of factor 2 and always threshold at 0. to make sure masks don't disappear
        # 0: 32x, 1: 16x, 2: 8x
        resized_masks = dict()
        for scale_index in range(2, -1, -1):
            masks = F.interpolate(masks.float(), scale_factor=(1.0, 0.5, 0.5),  mode='trilinear', align_corners=False) > 0.0
            masks = masks.repeat_interleave(self.num_vos_queries_per_object, 1)
            resized_masks[scale_index] = masks

        return resized_masks
