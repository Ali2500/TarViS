# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from einops import rearrange
from typing import Optional, List, Dict
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform_

from tarvis.modelling.pos_embeddings import PositionEmbeddingSine3D
from tarvis.modelling.cross_attention import SoftMaskedAttention, HardMaskedAttention
from tarvis.modelling.utils import split_by_query_group
from tarvis.modelling.constants import Constants


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, queries,
                query_embed: Optional[Tensor] = None):
        """
        Forward method
        :param queries: Tensor of shape [B, Q, C]
        :param query_embed: Optional tensor of shape [B, Q, C]
        :return:
        """
        # reshape to [Q, B, C] because this is the format expected by nn.MultiheadAttention
        queries = rearrange(queries, "B Q C -> Q B C")
        if query_embed is not None:
            query_embed = rearrange(query_embed, "B Q C -> Q B C")

        if self.normalize_before:
            output = self.forward_pre(queries, query_pos=query_embed)
        else:
            output = self.forward_post(queries, query_pos=query_embed)

        return rearrange(output, "Q B C -> B Q C")


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False,
                 cross_attention_type="hard_masked"):
        super().__init__()

        if cross_attention_type == "hard_masked":
            self.multihead_attn = HardMaskedAttention(embed_dim=d_model, num_heads=nhead, dropout=0.0)
        elif cross_attention_type == "soft_masked":
            self.multihead_attn = SoftMaskedAttention(embed_dim=d_model, num_heads=nhead, dropout=0.0)
        else:
            raise ValueError(f"Invalid cross attention type: {cross_attention_type}")

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     queries: Tensor,
                     vid_features: Tensor,
                     vid_mask: Tensor,
                     pos_embed: Optional[Tensor] = None,
                     query_embed: Optional[Tensor] = None):
        # video features have shape [B, C, T*H*W] but masked attention expects [B, T*H*W, C]
        vid_features = rearrange(vid_features, "B C N -> B N C")
        pos_embed = rearrange(pos_embed, "B C N -> B N C")

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(queries, query_embed),
            key=self.with_pos_embed(vid_features, pos_embed),
            value=vid_features, mask=vid_mask, return_attn_weights=False
        )
        queries = queries + self.dropout(tgt2)
        queries = self.norm(queries)

        return queries

    def forward_pre(self,
                    queries: Tensor,
                    vid_features: Tensor,
                    vid_mask: Tensor,
                    pos_embed: Optional[Tensor] = None,
                    query_embed: Optional[Tensor] = None):
        tgt2 = self.norm(queries)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_embed),
                                   key=self.with_pos_embed(vid_features, pos_embed),
                                   value=vid_features, mask=vid_mask)
        queries = queries + self.dropout(tgt2)

        return queries

    def forward(self,
                queries: Tensor,
                vid_features: Tensor,
                vid_mask: Tensor,
                pos_embed: Optional[Tensor] = None,
                query_embed: Optional[Tensor] = None):

        if self.normalize_before:
            return self.forward_pre(
                queries=queries, vid_features=vid_features, vid_mask=vid_mask,
                pos_embed=pos_embed, query_embed=query_embed
            )

        else:
            return self.forward_post(
                queries=queries, vid_features=vid_features, vid_mask=vid_mask,
                pos_embed=pos_embed, query_embed=query_embed
            )


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim: int,
                 nheads: int,
                 dim_feedforward: int,
                 dec_layers: int,
                 pre_norm: bool,
                 enforce_input_project: bool,
                 cross_attention_type: str,
                 ):
        super().__init__()

        self.pos_embed_generator = PositionEmbeddingSine3D(hidden_dim // 2, normalize=True)

        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    cross_attention_type=cross_attention_type
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.scale_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv3d(in_channels, hidden_dim, kernel_size=(1, 1, 1)))

                # weight_init.c2_xavier_fill(self.input_proj[-1])
                kaiming_uniform_(self.input_proj[-1].weight)
            else:
                self.input_proj.append(nn.Identity())

        self.mask_embeds = nn.ModuleDict()
        self.proj_layers = nn.ModuleDict()

        self.class_embed = self.proj_instance = self.proj_semantic = self.proj_background = None

        # if ffn_for_cls:
        #     self.mask_embeds["instance"] = MLP(hidden_dim, hidden_dim, 256, 3)
        #     self.class_embed = nn.Linear(hidden_dim, 81)  # TODO: 81 is hard-coded
        # else:
        def make_proj_layer():
            return nn.Linear(hidden_dim, hidden_dim)  # nn.Identity()

        for query_type in Constants.all_query_types:
            self.mask_embeds[query_type] = MLP(hidden_dim, hidden_dim, 256, 3)
            if query_type not in ["free"]:
                self.proj_layers[query_type] = make_proj_layer()

    def forward(self, fmaps: List[Tensor], queries: Tensor, query_embed: Tensor, query_group_names: List[str],
                query_group_sizes: List[int]) -> Dict[str, Tensor]:
        """
        Forward method
        :param fmaps: List of multi-scale feature maps of shape [B, C, T, H_i, W_i]
        :param queries: Queries of shape [B, Q, C]
        :param query_embed: Query embeddings of shape [B, Q, C]
        :param query_group_names:
        :param query_group_sizes:
        :return:
        """
        assert len(fmaps) == self.num_feature_levels + 1        

        # separate the highest-scale feature map. This will only be used for mask logit generation. For the
        # transformer layers we'll use the remaining, smaller feature maps (8x, 16x, 32x).
        fmap_for_mask = fmaps[-1]
        fmaps = fmaps[:-1]

        vid_features = []
        pos_encodings = []
        size_list = []

        for i in range(self.num_feature_levels):
            assert fmaps[i].ndim == 5 
            size_list.append(fmaps[i].shape[-3:])

            pos_embedding_i = self.pos_embed_generator(fmaps[i])
            pos_encodings.append(rearrange(pos_embedding_i, "B C T H W -> B C (T H W)"))

            features_i = self.input_proj[i](fmaps[i])  # [B, C, T, H, W]
            scale_embed_i = self.scale_embed.weight[i][None, :, None]  # [1, C, 1]
            vid_features.append(rearrange(features_i, "B C T H W -> B C (T H W)") + scale_embed_i)

        # compute outputs for the initial queries
        layer_0_outputs = self.compute_layer_outputs(
            queries=queries,
            vid_features_for_mask=fmap_for_mask,
            fmap_size=list(size_list[0]),
            query_group_names=query_group_names,
            query_group_sizes=query_group_sizes
        )

        outputs = {
            "queries": [queries],  # [B, Q, C]
            "mask_logits": [layer_0_outputs["mask_logits"]],  # [B, Q, H/4, W/4]
        }
        if "class_logits" in layer_0_outputs:
            outputs["class_logits"] = [layer_0_outputs["class_logits"]]  # [B, Q, 1+num_classes]

        current_attention_mask = layer_0_outputs["attention_mask"]

        for i in range(self.num_layers):
            scale_index = i % self.num_feature_levels

            # Cross-attention
            queries = self.transformer_cross_attention_layers[i](
                queries=queries,
                vid_features=vid_features[scale_index],
                vid_mask=current_attention_mask,
                pos_embed=pos_encodings[scale_index],
                query_embed=query_embed
            )

            # Self-attention
            queries = self.transformer_self_attention_layers[i](
                queries=queries,
                query_embed=query_embed
            )

            # FFN
            queries = self.transformer_ffn_layers[i](queries)

            next_scale_index = (i + 1) % self.num_feature_levels
            layer_i_output = self.compute_layer_outputs(
                queries=queries,
                vid_features_for_mask=fmap_for_mask,
                fmap_size=list(size_list[next_scale_index]),  # give fmap size for next layer
                query_group_names=query_group_names,
                query_group_sizes=query_group_sizes
            )

            outputs["queries"].append(queries)
            outputs["mask_logits"].append(layer_i_output["mask_logits"])

            if "class_logits" in layer_i_output:
                outputs["class_logits"].append(layer_i_output["class_logits"])

            current_attention_mask = layer_i_output["attention_mask"]

        return {
            k: torch.stack(v, 0) for k, v in outputs.items()
        }

    def compute_layer_outputs(self, queries: Tensor, vid_features_for_mask: Tensor, fmap_size: List[int],
                              query_group_names: List[str], query_group_sizes: List[int]):
        queries = self.decoder_norm(queries)

        queries_dict = split_by_query_group(queries, 1, query_group_names, query_group_sizes)

        if "semantic" in queries_dict and "instance" in queries_dict:
            # maybe should pass the task type as an argument?
            instance_queries = self.proj_layers["instance"](queries_dict["instance"])
            semantic_queries = self.proj_layers["semantic"](queries_dict["semantic"])  # [B, Q, C]
            bg_queries = self.proj_layers["background"](queries_dict["background"])

            semantic_dotprod = torch.einsum("bic,bsc->bis", instance_queries, semantic_queries)  # [B, I, num_classes]
            bg_dotprod = torch.einsum("bic,bsc->bis", instance_queries, bg_queries)  # [B, I, #bg]
            bg_dotprod = bg_dotprod.max(2, keepdim=True)[0]  # [B, I, 1], max-pool over all background queries

            class_preds = torch.cat((bg_dotprod, semantic_dotprod), 2)  # [B, I, 1+C], background logits comes first!

        elif "vos" in queries_dict and "background_vos" in queries_dict and "instance" in queries_dict:
            instance_queries = self.proj_layers["instance"](queries_dict["instance"])
            vos_queries = self.proj_layers["vos"](queries_dict["vos"])  # [B, Q, C]
            bg_queries = self.proj_layers["background_vos"](queries_dict["background_vos"])

            vos_cls_dotprod = torch.einsum("bic,bsc->bis", instance_queries, vos_queries)  # [B, I, num_objects]
            bg_dotprod = torch.einsum("bic,bsc->bis", instance_queries, bg_queries)  # [B, I, #bg]
            bg_dotprod = bg_dotprod.max(2, keepdim=True)[0]  # [B, I, 1], max-pool over all background queries

            class_preds = torch.cat((bg_dotprod, vos_cls_dotprod), 2)  # [B, I, 1+C], background logits come first!

        else:
            class_preds = None

        mask_queries = [
            self.mask_embeds[name](queries_dict[name])
            for name in query_group_names
        ]

        mask_queries = torch.cat(mask_queries, 1)  # [B, Q, C]

        # [B, C, T, H, W] * [B, Q, C] -> [B, Q, T, H, W]
        mask_logits = torch.einsum("bcthw,bqc->bqthw", vid_features_for_mask, mask_queries)

        # To obtain the attention masks, we apply sigmoid and resize the mask to the feature map scale at which
        # the current layer operates
        with torch.no_grad():
            attn_mask = self.resize_masks_for_attention(mask_logits, fmap_size)

            # if vos_query_attn_masks is not None:
            #     attn_mask = self.apply_vos_query_attn_masks(attn_mask, query_group_names, query_group_sizes,
            #                                                 vos_query_attn_masks, vos_query_attn_masks_valid)

            attn_mask = rearrange(attn_mask, "B Q T H W -> B Q (T H W)")

        outputs = {
            "mask_logits": mask_logits,
            "attention_mask": attn_mask
        }

        if class_preds is not None:
            outputs["class_logits"] = class_preds

        return outputs

    @staticmethod
    def resize_masks_for_attention(mask_logits: Tensor, fmap_size: List[int]):
        return F.interpolate(mask_logits, fmap_size, mode='trilinear', align_corners=False).sigmoid().detach()

    # @staticmethod
    # def apply_vos_query_attn_masks(attn_masks: Tensor, query_group_names: List[str], query_group_sizes: List[int],
    #                                vos_query_attn_masks: Tensor, vos_query_attn_masks_valid: Tensor):
    #     """
    #     Apply ref object masks as attention masks to VOS queries
    #     :param attn_masks: [B, Q, T, H, W]
    #     :param query_group_names:
    #     :param query_group_sizes:
    #     :param vos_query_attn_masks: [B, Qv, T, H, W]
    #     :param vos_query_attn_masks_valid: [B, Qv, T, 1, 1]
    #     :return:
    #     """
    #     vos_query_attn_masks = vos_query_attn_masks.type_as(attn_masks)
    #     attn_masks = split_by_query_group(attn_masks, 1, query_group_names, query_group_sizes)
    #     assert attn_masks["vos"].shape[:3] == vos_query_attn_masks_valid.shape[:3]
    #     attn_masks["vos"] = torch.where(vos_query_attn_masks_valid, vos_query_attn_masks, attn_masks["vos"])
    #     return torch.cat([attn_masks[k] for k in query_group_names], 1)
