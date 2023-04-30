from torch import Tensor
from typing import List, Optional

import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, num_dims, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(num_dims, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(num_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query: Tensor,
                key: Tensor,
                value: Tensor,
                query_embed: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None):
        # query: [B, Q, C], key: [B, Q, C]
        q = query + query_embed if query_embed is not None else query
        attn_output = self.attn(query=q, key=key, value=value, key_padding_mask=key_padding_mask)[0]
        return self.norm(query + attn_output)


class SelfAttentionLayer(nn.Module):
    def __init__(self, num_dims, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(num_dims, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(num_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor, embed: Optional[Tensor] = None):
        # x: [B, Q, C]
        q = k = x + embed if embed is not None else x
        attn_output = self.attn(query=q, key=k, value=x)[0]
        return self.norm(x + attn_output)


class FFNLayer(nn.Module):
    def __init__(self, num_dims, num_hidden_dims=2048):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_dims, num_hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden_dims, num_dims)
        )
        self.norm = nn.LayerNorm(num_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor):
        return self.norm(x + self.mlp(x))
