from einops import rearrange, repeat
from typing import Optional, Union

import math
import torch
import torch.nn as nn
from torch import Tensor


class HardMaskedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()

        self._attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, bias=bias,
                                           batch_first=True)

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

        if mask.dtype in (torch.float16, torch.float32):
            mask = (mask < 0.5).bool().detach()  # positions that are True will be masked out

        # disable masking for queries whose mask is all ones otherwise we will get NaNs in the output
        mask[torch.where(mask.sum(-1) == mask.shape[-1])] = False

        mask = repeat(mask, "bs tgt_len src_len -> (bs nhead) tgt_len src_len", nhead=self._attn.num_heads)

        if self.training:
            output = self._attn(query=query, key=key, value=value, attn_mask=mask, key_padding_mask=key_padding_mask)
        else:
            output = self.attention_for_inference(query=query, key=key, value=value, mask=mask, key_padding_mask=key_padding_mask)

        if not return_attn_weights:
            output = output[0]

        return output

    def attention_for_inference(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor, key_padding_mask: Union[Tensor, None]):
        # first try to process everything at once
        try:
            output = self._attn(query=query, key=key, value=value, attn_mask=mask, key_padding_mask=key_padding_mask)
            return output
        except RuntimeError as err:
            if "CUDA out of memory. Tried to allocate" in str(err):
                print(f"Encountered OOM error. {query.shape}, {key.shape}, {value.shape}")
                pass
            else:
                raise err

        # divide query and mask into chunks
        query = query.chunk(2, dim=1)
        mask = mask.chunk(2, dim=1)

        output = []
        output_weights = []
        for chunk_query, chunk_mask in zip(query, mask):
            chunk_output, chunk_output_weights = self._attn(
                query=chunk_query, key=key, value=value, attn_mask=chunk_mask, key_padding_mask=key_padding_mask
            )

            output.append(chunk_output)
            output_weights.append(chunk_output_weights)

        return torch.cat(output, 1), torch.cat(output_weights, 1)


def _test():
    vanilla_attn = nn.MultiheadAttention(256, 8, 0.0, batch_first=True).cuda()
    my_attn = HardMaskedAttention(256, 8, 0.0).cuda()

    # copy weights
    with torch.no_grad():
        my_attn._attn.in_proj_weight.copy_(vanilla_attn.in_proj_weight)
        my_attn._attn.in_proj_bias.copy_(vanilla_attn.in_proj_bias)

        my_attn._attn.out_proj.weight.copy_(vanilla_attn.out_proj.weight)
        my_attn._attn.out_proj.bias.copy_(vanilla_attn.out_proj.bias)
        # my_attn.q_proj.weight.copy_(vanilla_attn.in_proj_weight[:256])
        # my_attn.q_proj.bias.copy_(vanilla_attn.in_proj_bias[:256])
        #
        # my_attn.k_proj.weight.copy_(vanilla_attn.in_proj_weight[256:512])
        # my_attn.k_proj.bias.copy_(vanilla_attn.in_proj_bias[256:512])
        #
        # my_attn.v_proj.weight.copy_(vanilla_attn.in_proj_weight[512:])
        # my_attn.v_proj.bias.copy_(vanilla_attn.in_proj_bias[512:])
        #
        # my_attn.out_proj.weight.copy_(vanilla_attn.out_proj.weight)
        # my_attn.out_proj.bias.copy_(vanilla_attn.out_proj.bias)

    SRC_LEN = 20
    TGT_LEN = 10
    DIMS = 256

    q = torch.randn((1, TGT_LEN, DIMS)).cuda()
    k = torch.randn((1, SRC_LEN, DIMS)).cuda()
    v = torch.randn((1, SRC_LEN, DIMS)).cuda()

    mask = torch.rand(1, TGT_LEN, SRC_LEN).cuda()
    mask_expanded = repeat(mask, "B T C -> (B H) T C", H=8)

    output_vanilla = vanilla_attn(query=q, key=k, value=v, attn_mask=mask_expanded < 0.5, need_weights=False)[0]
    output_mine = my_attn(query=q, key=k, value=v, mask=mask)

    print((output_mine - output_vanilla).abs().sum().item())
    assert torch.allclose(output_vanilla, output_mine)


if __name__ == '__main__':
    _test()
