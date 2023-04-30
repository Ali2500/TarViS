from typing import Dict, Optional, List, Tuple, Union
from torch import Tensor
from einops import repeat

import torch
import torch.nn as nn

class EmbeddingContainer(nn.Module):
    def __init__(self, num_dims: int, num_classes: Dict[str, int], num_objs: int, num_bg: int):
        super().__init__()
        num_semantic_classes = sum(num_classes.values())
        self._semantic_queries = self._make_param(1 + num_semantic_classes, num_dims)

        self._semantic_indices = dict()
        current_offset = 1
        for name in sorted(num_classes.keys()):
            self._semantic_indices[name] = (current_offset, current_offset + num_classes[name])
            current_offset += num_classes[name]

        self._background_queries = None
        self._background_vos_queries = None
        if num_bg > 0:
            self._background_queries = self._make_param(num_bg + 1, num_dims)
            self._background_vos_queries = self._make_param(num_bg + 1, num_dims)

        self._instance_queries = self._make_param(num_objs + 1, num_dims)

        self._vos_queries = self._make_param(1, num_dims)

        self._semantic_queries_keep_indices = dict()

        self._num_dims = num_dims
        self.reset_parameters()

    @staticmethod
    def _make_param(*size):
        return nn.Parameter(torch.zeros(*size, requires_grad=True), requires_grad=True)

    def reset_parameters(self):
        for name, p in self.named_parameters():
            nn.init.normal_(p)

    @property
    def num_instance_queries(self):
        return self._instance_queries.size(0) - 1

    @property
    def instance_queries(self):
        return self._split_and_repeat(self._instance_queries)

    @property
    def background_queries(self):
        assert self._background_queries is not None
        return self._split_and_repeat(self._background_queries)

    @property
    def background_vos_queries(self):
        assert self._background_vos_queries is not None
        return self._split_and_repeat(self._background_vos_queries)

    def vos_queries(self, num_objects: int, init: Tensor):
        embed = self._vos_queries.repeat(num_objects, 1)
        return embed, init

    def semantic_queries(self, dataset: str):
        embed = self._semantic_queries[0]  # [C]

        start, end = self._semantic_indices[dataset]
        init = self._semantic_queries[start:end]

        if dataset in self._semantic_queries_keep_indices:
            init = init[self._semantic_queries_keep_indices[dataset]]

        num_queries = init.size(0)
        return embed[None].repeat(num_queries, 1), init

    @torch.no_grad()
    def assign_semantic_queries(self, dataset: str, queries: Tensor):
        start, end = self._semantic_indices[dataset]
        assert list(queries.shape) == [end - start, self._num_dims], f"Shape mismatch: {end - start}, {queries.shape}"
        self._semantic_queries[start:end] = queries

    @staticmethod
    def _split_and_repeat(x: Tensor):
        assert x.size(0) > 1
        embed, init = x[0], x[1:]
        num_queries = init.size(0)
        return embed[None].repeat(num_queries, 1), init

    @torch.no_grad()
    def prune_semantic_query_set(self, dataset_name: str, indices_to_keep: Union[Tensor, List[int]]):
        if not torch.is_tensor(indices_to_keep):
            indices_to_keep = torch.as_tensor(indices_to_keep, dtype=torch.int64)

        # sanity check:
        start, end = self._semantic_indices[dataset_name]
        assert torch.all(indices_to_keep < end - start)

        self._semantic_queries_keep_indices[dataset_name] = indices_to_keep
