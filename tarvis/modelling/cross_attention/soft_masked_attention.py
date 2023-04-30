from einops import rearrange, repeat
from typing import Optional

import math
import torch
import torch.nn as nn
from torch import Tensor


class SoftMaskedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, learnable_scaling_factors=False):
        super().__init__()

        self._attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, bias=bias,
                                           batch_first=True)
        if learnable_scaling_factors:
            self.scaling_factors = nn.Parameter(torch.as_tensor(self.get_default_scaling_factors()))
        else:
            self.register_buffer("scaling_factors", torch.as_tensor(self.get_default_scaling_factors()))

    def get_default_scaling_factors(self):
        assert self._attn.num_heads == 8
        return [32., 32., 16., 16., 8., 8., 4., 4.]

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor,
                key_padding_mask: Optional[Tensor] = None, return_attn_weights: Optional[bool] = False):
        """
        Forward method
        :param query: [bs, num_instances, embed_dim]
        :param key: [bs, src_len, embed_dim]
        :param value: [bs, src_len, embed_dim]
        :param mask: tensor of shape [bs, num_instances, src_len] of dtype torch.bool or float values in range [0, 1]
        :param key_padding_mask: optional tensor of shape [bs, src_len] of type bool. Value should be true for padded
        locations and false for valid locations.
        :param return_attn_weights: if True, attention weights will also be returned as a tensor of shape
        [bs, num_heads, num_instances, src_len]
        :return: tensor of shape [bs, num_instances, embed_dim]
        """
        assert key.shape == value.shape, f"Shape mismatch: {key.shape}, {value.shape}"
        assert query.shape[:2] == mask.shape[:2], f"Shape mismatch: {query.shape}, {mask.shape}"

        mask = repeat(mask, "bs tgt_len src_len -> bs num_heads tgt_len src_len", num_heads=self._attn.num_heads)
        mask = mask * self.scaling_factors[None, :, None, None]
        mask = rearrange(mask, "bs num_heads tgt_len src_len -> (bs num_heads) tgt_len src_len")

        output = self._attn(query=query, key=key, value=value, attn_mask=mask, key_padding_mask=key_padding_mask)

        if not return_attn_weights:
            output = output[0]

        return output
