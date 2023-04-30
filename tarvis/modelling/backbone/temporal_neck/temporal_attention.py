from einops import rearrange, repeat
from torch import Tensor
from typing import Tuple, List
from torch.cuda.amp import autocast

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TemporalAttentionLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.0, activation="relu", n_heads=8):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    @torch.no_grad()
    def chunked_self_attention(self, kq, value):
        assert not self.training
        num_chunks = 8
        kq = kq.chunk(num_chunks, dim=0)
        value = value.chunk(num_chunks, dim=0)

        attn_output = []
        for i in range(num_chunks):
            attn_output.append(self.self_attn(query=kq[i], key=kq[i], value=value[i])[0])

        return torch.cat(attn_output)

    def forward(self, src: Tensor, pos: Tensor, patch_mask_indices: Tensor):
        """
        Forward method
        :param src: tensor of shape [B*T, L*H*W, C]
        :param pos: tensor of shape [B, T, L*H*W, C]
        :param patch_mask_indices: tensor of shape [P, N] (int64) (N = patch area)
        :return:
        """
        # print("temporal: ", src.dtype, pos.dtype)
        batch_sz, clip_len = pos.shape[:2]
        num_patches = patch_mask_indices.size(0)
        assert batch_sz * clip_len == src.size(0)
        assert src.shape[-2:] == pos.shape[-2:], f"{src.shape}, {pos.shape}"

        src = rearrange(src, "(B T) LHW C -> LHW T B C", B=batch_sz, T=clip_len)
        pos = rearrange(pos, "B T LHW C -> LHW T B C")

        patch_indices = rearrange(patch_mask_indices, "P N -> (P N)")

        src = src[patch_indices]  # [P*N, T B C]
        src = rearrange(src, "(P N) T B C -> (B P) (N T) C", B=batch_sz, P=num_patches, T=clip_len)

        pos = pos[patch_indices]
        pos = rearrange(pos, "(P N) T B C -> (B P) (N T) C", B=batch_sz, P=num_patches, T=clip_len)

        kq = self.with_pos_embed(src, pos)

        if kq.size(1) > 500 and not self.training:  # split operation to avoid OOM during inference
            attn_output = self.chunked_self_attention(kq, src)
        else:
            attn_output = self.self_attn(query=kq, key=kq, value=src)[0]  # [B*P, N*T C]

        src = src + self.dropout1(attn_output)

        # restore input dimensions and remove padded points
        src = rearrange(src, "(B P) (N T) C -> (P N) (B T) C", B=batch_sz, P=num_patches, T=clip_len)

        unique_idxes = patch_indices.unique()
        assert unique_idxes.numel() == patch_indices.numel(), f"{unique_idxes.shape} =/= {patch_indices.shape}"

        reverse_indices = patch_indices.argsort()
        src = src[reverse_indices]  # [L*H*W, B*T C]
        src = rearrange(src, "LHW BT C -> BT LHW C")

        src = self.norm1(src)
        src = self.forward_ffn(src)

        return src


@torch.no_grad()
def get_patch_mask_indices(fmaps: List[Tensor], ksize_offset: int = 0) -> List[Tensor]:
    """
    Get masks for when the feature maps are divided into a grid where the size of each grid cell is (ksize_offset + 1) * (ksize_offset + 1)
    :param fmaps: list of multi-scale features, each of shape [B, T, C, H_i, W_i] in ascending order of spatial dimensions
    :return: tensor of shape [num_patches, H, W] of type bool
    """
    # batch_sz, num_frames, _, H_min, W_min = fmaps[0].shape
    H_min, W_min = fmaps[0].shape[-2:]
    assert ksize_offset in (0, 1)
    if ksize_offset == 1:
        assert H_min % 2 == 0 and W_min % 2 == 0, f"{H_min, W_min}"
    num_levels = len(fmaps)
    device = fmaps[0].device

    all_patch_indices = []
    index_offset = 0

    for l in range(num_levels):
        h_l, w_l = fmaps[l].shape[-2:]
        coords = torch.stack(torch.meshgrid([
            torch.arange(0, h_l, dtype=torch.int64, device=device),
            torch.arange(0, w_l, dtype=torch.int64, device=device)
        ], indexing='ij'))  # [C, H, W]

        ks = 2**(l + ksize_offset)
        patches_indices = F.unfold(coords[None].float(), kernel_size=ks, stride=ks, padding=0).squeeze(0).to(torch.int64)  # [2 * patch_sz, num_patches]
        patches_indices = rearrange(patches_indices, "(C patch_sz) num_patches -> C patch_sz num_patches", C=2)

        # flatten (y, x) coords into (y * width + x)
        patches_indices = (patches_indices[0] * w_l) + patches_indices[1] + index_offset  # [patch_sz, num_patches]

        # try:
        assert patches_indices.size(1) == H_min * W_min / (ksize_offset + 1)**2, f"{fmaps[l].shape}, {patches_indices.shape}, {(H_min, W_min)}, {ksize_offset}"
        # except AssertionError as err:
        #     breakpoint()
        all_patch_indices.append(patches_indices.transpose(0, 1).to(torch.int64))  # [num_patches, patch_sz]
        index_offset += (h_l * w_l)

    return all_patch_indices  # List[num_levels, [num_patches, patch_size]


def _test():
    h_min, w_min = 2, 4
    num_lvls = 2

    fmaps = []
    for l in range(1, num_lvls+1):
        f = torch.arange(0, h_min*w_min*l*l).reshape(h_min*l, w_min*l)[None, None, None]  # [1, 1, 1, H', W']
        fmaps.append(f)

    y = get_patch_mask_indices(fmaps, ksize_offset=1)
    breakpoint()


if __name__ == '__main__':
    _test()
