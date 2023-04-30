from einops import rearrange, repeat
from torch import Tensor
from typing import List, Tuple, Union
from tarvis.modelling.vos_pet_helpers.pet_query_initializer import PETQueryInitializer
from tarvis.modelling.pos_embeddings import PositionEmbeddingSine2D
from tarvis.modelling.point_sampling_utils import point_sample
from tarvis.modelling.vos_pet_helpers.attention_layers import CrossAttentionLayer, SelfAttentionLayer, FFNLayer

import torch
import torch.nn as nn
import tarvis.modelling.vos_pet_helpers.utils as utils


class VOSPETObjectEncoder(nn.Module):
    def __init__(self, num_dims: int, num_layers: int, num_queries_per_object: int,
                 bg_grid_size: Union[None, Tuple[int, int]], point_vos_likelihood: float,
                 backprop_through_fmap: bool = False):

        super().__init__()

        self.pos_embedding = PositionEmbeddingSine2D(num_dims // 2)

        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(num_dims, num_heads=8) for _ in range(num_layers)
        ])

        self.self_attn_layers = nn.ModuleList([
            SelfAttentionLayer(num_dims) for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            FFNLayer(num_dims) for _ in range(num_layers)
        ])

        self.proj_point_features = nn.ModuleList([
            nn.Linear(num_dims, num_dims) for _ in range(4)
        ])

        if bg_grid_size is None:
            self.bg_grid_size = None
        else:
            self.register_buffer("bg_grid_size", torch.tensor(list(bg_grid_size), dtype=torch.int64))

        self._max_points_per_object_train = 2048
        self._max_points_per_object_inference = 2**14  # ~16k

        self._num_queries_per_object = num_queries_per_object
        self.backprop_through_fmap = backprop_through_fmap

        self.register_parameter("mask_object_query_embed", nn.Parameter(torch.zeros(num_dims, dtype=torch.float32)))
        self.register_parameter("bg_query_embed", nn.Parameter(torch.zeros(num_dims, dtype=torch.float32)))
        self.register_parameter("point_object_query_embed", nn.Parameter(torch.zeros(num_dims, dtype=torch.float32)))

        self.point_vos_query_extractor = PETQueryInitializer(dim=num_dims, n_scales=4)
        self.point_vos_likelihood = point_vos_likelihood

        self.rng = torch.Generator()
        self.rng.manual_seed(22021994)  # fixed seed across all GPUs

        self._reset_parameters()

    @property
    def num_queries_per_object(self):
        if self.training:
            return self._num_queries_per_object
        else:
            return self._num_queries_per_object * 2  # use more queries per object during inference

    @property
    def provides_background_queries(self):
        return self.num_bg_queries > 0

    @property
    def num_bg_queries(self):
        if self.bg_grid_size is None:
            return 0
        else:
            return int(self.bg_grid_size[0] * self.bg_grid_size[1])

    @property
    def max_points_per_object(self):
        if self.training:
            return self._max_points_per_object_train
        else:
            return self._max_points_per_object_inference

    def _reset_parameters(self):
        nn.init.normal_(self.mask_object_query_embed)
        nn.init.normal_(self.point_object_query_embed)
        nn.init.normal_(self.bg_query_embed)

    def forward(self, **kwargs):
        mode = kwargs.pop("mode")
        if mode == "train":
            return self.forward_train(**kwargs)
        elif mode == "infer":
            return self.forward_infer(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward_infer(self, fmap_mask_vos: Tensor, ref_masks: Tensor, fmaps_point_vos: List[Tensor],
                      point_coords: Tensor, image_padding: Tuple[int, int]):
        """
        Forward for inference with predicted masks from previous clip
        :param fmap_mask_vos: tensor of shape [C, T, H, W]
        :param ref_masks: tensor of shape [N, T, H, W] (N = num. of objects)
        :param fmaps_point_vos: list of tensors, each of shape [N, C, H, W]
        :param point_coords: tensor of shape [N, 2] with normalized coords in [0, 1] and (y, x) format.
        :param image_padding: point_coords are normalized w.r.t the unpadded image. This param should be in the format
        (pad_right, pad_bottom) and will be used to adjust the normalized point coordinates for the padded feature maps
        :return:
        """

        all_query_inits = []
        all_query_embeds = []
        all_object_features = []
        all_pos_embeddings = []
        
        # Here VOS is referred to as 'mask VOS' and PET is referred to as 'point VOS'
        if self.provides_background_queries or ref_masks.numel() > 0:
            fmap_mask_vos = fmap_mask_vos.unsqueeze(0)  # [1, C, T, H, W]
            num_mask_vos_samples = ref_masks.size(0)

            mask_query_inits, mask_object_features, mask_pos_embeddings, mask_query_embeds = \
                self.initialize_queries_for_vos(fmap=fmap_mask_vos, ref_masks=[ref_masks])

            all_query_inits.append(mask_query_inits)
            all_query_embeds.append(mask_query_embeds)

            all_object_features.extend(mask_object_features[0])
            all_pos_embeddings.extend(mask_pos_embeddings[0])
        else:
            num_mask_vos_samples = 0

        if point_coords is None:
            num_point_vos_samples = 0
        else:
            num_point_vos_samples = point_coords.size(0)
            assert all([f.size(0) == num_point_vos_samples for f in fmaps_point_vos])

            point_coords = point_coords.unsqueeze(1)  # [N, 1, 2]

            padded_height, padded_width = [int(x * 4) for x in fmap_mask_vos.shape[-2:]]
            point_coords = self.point_vos_query_extractor.adapt_point_coords(
                point_coords, image_padding, (padded_height, padded_width)
            )

            point_query_inits, point_object_features, point_pos_embeddings, point_query_embeds = \
                self.initialize_queries_for_pet_from_points(fmaps=fmaps_point_vos, point_coords=point_coords)

            all_query_inits.append(rearrange(point_query_inits, "N one C -> one N C"))
            all_query_embeds.append(rearrange(point_query_embeds, "N one C -> one N C"))

            all_object_features.extend([point_object_features[n][0] for n in range(num_point_vos_samples)])
            all_pos_embeddings.extend([point_pos_embeddings[n][0] for n in range(num_point_vos_samples)])

        all_query_inits = torch.cat(all_query_inits, 1)  # [1, N, C]
        all_query_embeds = torch.cat(all_query_embeds, 1)  # [1, N, C]

        # pad all features to same length
        all_object_features, all_pos_embeddings, padding_mask = utils.pad_batched_sampled_features(
            features=[all_object_features], pos_embeddings=[all_pos_embeddings]
        )

        all_queries = self.refine_query_inits(
            inits=all_query_inits, query_embed=all_query_embeds, object_features=all_object_features,
            object_pos_embeddings=all_pos_embeddings, attention_mask=padding_mask
        )  # [1, N, C] (N = #background + #mask VOS + #point VOS queries)

        bg_queries, object_queries = all_queries.split((self.num_bg_queries, all_queries.size(1) - self.num_bg_queries), 1)

        object_query_counts = [self.num_queries_per_object for _ in range(num_mask_vos_samples)] + \
                              [1 for _ in range(num_point_vos_samples)]

        mask_vos_flag = [True for _ in range(self.num_queries_per_object * num_mask_vos_samples)] + \
                        [False for _ in range(num_point_vos_samples)]
        mask_vos_flag = torch.tensor(mask_vos_flag, dtype=torch.bool, device=object_queries.device)

        return {
            "object_queries": object_queries.squeeze(0),  # [N, C]
            "bg_queries": bg_queries.squeeze(0) if self.provides_background_queries else None,  # [M, C]
            "object_query_counts": object_query_counts,
            "mask_vos_flag": mask_vos_flag
        }

    def forward_train(self, fmaps: List[Tensor], ref_masks: List[Tensor]):
        """
        Initialize VOS queries from reference frame feature map and object masks by average pooling
        :param fmaps: lsit of multi-scale feature maps, each of shape [B, C, T, H', W']
        :param ref_masks: list of masks for each batch sample of shape [N, T, H, W]
        :return:
        """
        fmap = fmaps[-1]  # largest feature map at 4x scale
        assert fmap.size(0) == len(ref_masks)
        assert len(set([m.size(0) for m in ref_masks])) == 1  # batch samples must contain the same number of objects
        assert fmap.ndim == 5, f"{fmap.shape}"
        batch_sz = fmap.shape[0]
        num_objects = ref_masks[0].size(0)

        # Here VOS is referred to as 'mask VOS' and PET is referred to as 'point VOS'
        num_point_vos_samples = (torch.rand((num_objects,), generator=self.rng) < self.point_vos_likelihood).sum().item()
        num_mask_vos_samples = num_objects - num_point_vos_samples

        ref_masks_for_mask_vos = [object_masks[:num_mask_vos_samples] for object_masks in ref_masks]
        ref_masks_for_point_vos = [object_masks[num_mask_vos_samples:].squeeze(1) for object_masks in ref_masks]

        all_query_inits = []
        all_query_embeds = []
        all_object_features = [[] for _ in range(batch_sz)]
        all_pos_embeddings = [[] for _ in range(batch_sz)]

        if self.provides_background_queries or num_mask_vos_samples > 0:
            mask_query_inits, mask_object_features, mask_pos_embeddings, mask_query_embeds = \
                self.initialize_queries_for_vos(fmap=fmaps[-1], ref_masks=ref_masks_for_mask_vos)

            all_query_inits.append(mask_query_inits)
            all_query_embeds.append(mask_query_embeds)

            for b in range(len(mask_object_features)):
                all_object_features[b].extend(mask_object_features[b])
                all_pos_embeddings[b].extend(mask_pos_embeddings[b])

        if num_point_vos_samples > 0:
            fmaps_2d = [f.squeeze(2) for f in fmaps]
            point_query_inits, point_object_features, point_pos_embeddings, point_query_embeds = \
                self.initialize_queries_for_pet_from_masks(fmaps=fmaps_2d, ref_masks=ref_masks_for_point_vos)

            all_query_inits.append(point_query_inits)
            all_query_embeds.append(point_query_embeds)

            for b in range(len(point_object_features)):
                all_object_features[b].extend(point_object_features[b])
                all_pos_embeddings[b].extend(point_pos_embeddings[b])

        all_query_inits = torch.cat(all_query_inits, 1)  # [B, N, C]
        all_query_embeds = torch.cat(all_query_embeds, 1)  # [B, N, C]

        # pad all features to same length
        all_object_features, all_pos_embeddings, padding_mask = utils.pad_batched_sampled_features(
            features=all_object_features, pos_embeddings=all_pos_embeddings
        )

        all_queries = self.refine_query_inits(
            inits=all_query_inits, query_embed=all_query_embeds, object_features=all_object_features,
            object_pos_embeddings=all_pos_embeddings, attention_mask=padding_mask
        )  # [B, N, C] (N = #background + #mask VOS + #point VOS queries)

        bg_queries, object_queries = all_queries.split((self.num_bg_queries, all_queries.size(1) - self.num_bg_queries), 1)

        object_query_counts = [self.num_queries_per_object for _ in range(num_mask_vos_samples)] + \
                              [1 for _ in range(num_point_vos_samples)]

        mask_vos_flag = [True for _ in range(self.num_queries_per_object * num_mask_vos_samples)] + \
                        [False for _ in range(num_point_vos_samples)]
        mask_vos_flag = torch.tensor(mask_vos_flag, dtype=torch.bool, device=object_queries.device)

        return {
            "object_queries": object_queries,
            "bg_queries": bg_queries if self.provides_background_queries else None,
            "object_query_counts": object_query_counts,
            "mask_vos_flag": mask_vos_flag
        }

    def initialize_queries_for_vos(self, fmap: Tensor, ref_masks: List[Tensor]):
        """
        Initialize queries using the full object masks
        :param fmap: tensor of shape [B, C, T, H, W]
        :param ref_masks: List of object masks, each of shape [N, T, H, W]
        :return:
        """
        assert self.provides_background_queries or ref_masks[0].size(0) > 0
        assert fmap.size(0) == len(ref_masks)
        assert len(set([m.size(0) for m in ref_masks])) == 1  # batch samples must contain the same number of objects
        assert fmap.ndim == 5, f"{fmap.shape}"
        batch_sz, num_dims, clip_len, height, width = fmap.shape

        all_masks = []

        if self.provides_background_queries:
            bg_masks = torch.stack(utils.generate_background_masks(ref_masks, self.bg_grid_size))  # [B, N_b, T, H, W]
            all_masks.append(bg_masks)

        if ref_masks[0].size(0) > 0:
            ref_masks = torch.stack(utils.divide_object_masks(ref_masks, self.num_queries_per_object))  # [B, N * num_queries_per_object, T, H, W]]
            all_masks.append(ref_masks)

        all_masks = torch.cat(all_masks, 1)  # [B, N_b + (N * num_queries_per_object), T, H, W]
        assert all_masks.numel() > 0

        with torch.set_grad_enabled(self.backprop_through_fmap):
            pos_embeddings = self.pos_embedding(None, dims=(batch_sz, height, width), device=fmap.device)
            pos_embeddings = repeat(pos_embeddings, "B C H W -> B C T H W", T=clip_len)

            object_features, object_pos_embeddings = utils.sample_features_from_masks(
                fmap, pos_embeddings, all_masks.unbind(0), max_allowed_num_points=self.max_points_per_object,
                num_bg_masks=self.num_bg_queries
            )
            query_inits = utils.average_batched_sampled_features(object_features)

        query_embeds = torch.cat((
            repeat(self.bg_query_embed, "C -> B N C", B=batch_sz, N=self.num_bg_queries),
            repeat(self.mask_object_query_embed, "C -> B M C", B=batch_sz, M=ref_masks[0].size(0))
        ), 1)

        return query_inits, object_features, object_pos_embeddings, query_embeds

    def initialize_queries_for_pet_from_masks(self, fmaps: List[Tensor], ref_masks: List[Tensor]):
        """
        Initialize queries using only the centroid point from the mask
        :param fmaps: List of multi-scale features of shape [B, C, H', W']
        :param ref_masks: List of object masks, each of shape [N, H, W]
        :return:
        """
        assert all([m.ndim == 3 for m in ref_masks])
        centroid_point_coords = self.point_vos_query_extractor.mask_to_point(ref_masks, coord_offset=0.5)  # List[B, [N, 2]]
        centroid_point_coords = torch.stack(centroid_point_coords)  # [B, N, 2]
        return self.initialize_queries_for_pet_from_points(fmaps, centroid_point_coords)

    def initialize_queries_for_pet_from_points(self, fmaps: List[Tensor], point_coords: Tensor):
        """
        Initialize queries using the given point coordinates
        :param fmaps: List of multi-scale features of shape [B, C, H', W']
        :param point_coords: tensor of shape [B, N, 2] where N = number of objects. Coordinates must be normalized
        to [0, 1] and be in (y, x) format.
        :return:
        """
        assert all([f.ndim == 4 for f in fmaps])
        query_inits, query_point_features = self.point_vos_query_extractor(fmaps=fmaps, point_coords=point_coords)  # [B, N, C]
        query_point_features = torch.stack([
            self.proj_point_features[i](rearrange(query_point_features[i], "B C N -> B N C")) for i in range(4)
        ], 2)  # [B, N, P, C] (P = 4)

        batch_sz, _, height, width = fmaps[-1].shape
        pos_embeddings = self.pos_embedding(None, dims=(batch_sz, height, width), device=fmaps[-1].device)

        point_coords = point_coords.flip([2])  # (y, x) -> (x, y)
        point_pos_embeddings = point_sample(pos_embeddings, point_coords, mode='nearest', align_corners=False)
        point_pos_embeddings = repeat(point_pos_embeddings, "B C N -> B N P C", P=4)

        # split query_point_features and point_pos_embeddings into lists for latter padding
        query_point_features = [x.unbind(0) for x in query_point_features.unbind(0)]
        point_pos_embeddings = [x.unbind(0) for x in point_pos_embeddings.unbind(0)]

        query_embeds = repeat(self.point_object_query_embed, "C -> B N C", B=batch_sz, N=point_coords.size(1))

        return query_inits, query_point_features, point_pos_embeddings, query_embeds

    def refine_query_inits(self, inits: Tensor, query_embed: Tensor, object_features: Tensor,
                           object_pos_embeddings: Tensor, attention_mask: Tensor):
        """
        Forward method
        :param inits: tensor of shape [B, bg+N, C]  (N = num objects, bg = num of background queries)
        :param query_embed: tensor of shape [B, bg+N, C]
        :param object_features: tensor of shape [B, bg+N, P, C] (P = max num points  per object)
        :param object_pos_embeddings: tensor of identical shape to `object_features`.
        :param attention_mask: tensor of shape [B, bg+N, P]  (False at valid points, True at invalid/padded points)
        :return: Tensor of shape [B, bg+N, C]
        """
        batch_sz, num_queries = inits.shape[:2]
        queries = rearrange(inits, "B N C -> (B N) 1 C")

        object_features = rearrange(object_features, "B N P C -> (B N) P C")
        object_pos_embeddings = rearrange(object_pos_embeddings, "B N P C -> (B N) P C")
        key_padding_mask = rearrange(attention_mask, "B N P -> (B N) P")
        query_embed_cross_attn = rearrange(query_embed, "B N C -> (B N) 1 C")

        for layer_cross_attn, layer_self_attn, layer_ffn in zip(self.cross_attn_layers, self.self_attn_layers, self.ffn_layers):
            # self attention
            queries = rearrange(queries.squeeze(1), "(B N) C -> B N C", B=batch_sz, N=num_queries)
            queries = layer_self_attn(queries, embed=query_embed)
            queries = rearrange(queries, "B N C -> (B N) 1 C")

            # cross-attention
            k = object_features + object_pos_embeddings
            v = object_features
            queries = layer_cross_attn(query=queries,
                                       key=k,
                                       value=v,
                                       query_embed=query_embed_cross_attn,
                                       key_padding_mask=key_padding_mask)

            queries = layer_ffn(queries)

        return rearrange(queries.squeeze(1), "(B N) C -> B N C", B=batch_sz, N=num_queries)
