from collections import defaultdict
from typing import Union, Optional, List, Dict
from torch.utils.data import Sampler, BatchSampler

from tarvis.data.training_dataset_base import TrainingDatasetBase, ConcatDataset

import math
import torch
import torch.distributed as dist
import tarvis.utils.distributed as dist_utils


class TaskTypeAwareBatchSampler(BatchSampler):
    """
    We train on a heterogeneous training set containing samples from different datasets and different types of
    tasks. To simplify the forward pass logic, we ensure that all samples within a given batch come from the same
    dataset.

    (1) For the VOS/PET, we ensure that all samples in a batch contain the same number of
        instances. This negates the need for complicated masking operations in the forward pass.

    (2) With multi-GPU training using DDP, the forward pass has to be identical across all processes. This imposes the
        additional constraint that for a given training iteration, the batches samples across all processes must be of
        the same task type.
    """

    def __init__(self, sampler: Union[None, Sampler[int]],
                 dataset: Union[ConcatDataset, TrainingDatasetBase],
                 total_iterations: int,
                 batch_size: int,
                 post_shuffle: bool,
                 # max_allowed_ar_diff: Optional[float] = 0.1,
                 elapsed_batches: Optional[int] = 0,
                 chunking_factor: Optional[int] = 1,
                 sub_batch_size: Optional[int] = None) -> None:
        super().__init__(sampler, batch_size, drop_last=True)

        if sampler is None:
            sample_idxes = list(range(len(dataset)))
        else:
            sample_idxes = list(sampler)

        assert len(sample_idxes) % batch_size == 0, \
            f"Number of samples {len(sample_idxes)} must be exactly divisible by batch size per GPU ({batch_size})"

        # separate the samples into groups based on the task type
        sample_type_ids = dataset.sample_type_ids()
        grouped_sample_idxes = defaultdict(list)

        for i in sample_idxes:
            grouped_sample_idxes[sample_type_ids[i]].append(i)

        # Check the validity of the sampler's output. For multi-GPU training, the number of samples per group must be
        # exactly the same across all processes.
        grouped_sample_idxes = self.check_sampler_correctness(grouped_sample_idxes)

        batch_sample_indices = []
        batch_group_ids = []

        batch_size_cf = batch_size * chunking_factor

        # for each group, create batches in a way that samples within the batch have similar image aspect ratios
        for group_id, group_name in enumerate(sorted(grouped_sample_idxes.keys())):
            idxes = torch.as_tensor(grouped_sample_idxes[group_name], dtype=torch.long)  # [N]

            batch_sample_indices.append(self.create_minibatches(
                sample_idxes=idxes,
                batch_size=batch_size_cf
            ))

            batch_group_ids.append(torch.full((batch_sample_indices[-1].size(0),), group_id, dtype=torch.long))

        # concatenate the batch indices
        batch_sample_indices = torch.cat(batch_sample_indices, 0)
        batch_group_ids = torch.cat(batch_group_ids, 0)

        num_expected_batches = total_iterations // chunking_factor

        if batch_sample_indices.size(0) < num_expected_batches:
            n_pad = num_expected_batches - batch_sample_indices.size(0)
            batch_sample_indices = torch.cat((batch_sample_indices, batch_sample_indices[:n_pad]))

        elif batch_sample_indices.size(0) > num_expected_batches:
            batch_sample_indices = batch_sample_indices[:num_expected_batches]
            batch_group_ids = batch_group_ids[:num_expected_batches]

        assert len(batch_sample_indices) == num_expected_batches  # sanity check

        # sort the batches by group ID. We need to ensure that the batch group ID will be the same for a given index
        # across all GPU processes
        indices_sorted_by_group_id = batch_group_ids.argsort()
        batch_sample_indices = batch_sample_indices[indices_sorted_by_group_id]
        batch_group_ids = batch_group_ids[indices_sorted_by_group_id]
        # The DistributedSampler ensures that the number of samples per group is the same across all processes. So now
        # the group IDs for a given index in `batch_sample_indices` should be the same across all processes.

        if post_shuffle:
            # Perform deterministic shuffle by using the same seed. This will preserve the batch group ID consistency
            # across processes
            g = torch.Generator()
            g.manual_seed(123987)
            randperm = torch.randperm(num_expected_batches, generator=g)
            batch_sample_indices = batch_sample_indices[randperm]
            batch_group_ids = batch_group_ids[randperm]

        # Sanity check: ensure that group IDs are consistent across all processes
        self.check_group_id_consistency(batch_group_ids)

        # reverse the chunking factor
        batch_sample_indices = batch_sample_indices.reshape(-1, batch_size)

        # Important: `elapsed_batches` should refer to the number of iterations of optimizer step calls, NOT the number
        # of total sub-batches elapsed so far.
        if elapsed_batches > 0:
            assert elapsed_batches < len(batch_sample_indices), f"{elapsed_batches}, {batch_sample_indices.shape}"
            batch_sample_indices = batch_sample_indices[elapsed_batches:]

        # Divide batches into sub-batches if needed. This is done when training with accumulated gradients.
        if sub_batch_size is not None:
            assert 0 < sub_batch_size <= batch_size
            assert batch_size % sub_batch_size == 0
            batch_sample_indices = batch_sample_indices.reshape(-1, sub_batch_size)

        self.batch_sample_indices = batch_sample_indices

    @staticmethod
    def create_minibatches(sample_idxes, batch_size):
        n_pad = int((math.ceil(float(sample_idxes.numel()) / batch_size) * batch_size) - sample_idxes.numel())

        if n_pad > 0:
            # sample_ars = torch.cat((sample_ars, sample_ars[:n_pad]))
            sample_idxes = torch.cat((sample_idxes, sample_idxes[:n_pad]))

        return sample_idxes.reshape(-1, batch_size)

    @staticmethod
    def check_sampler_correctness(grouped_sample_idxes: Dict[str, List[int]]):
        # grouped_sample_idex: Dict[str, List[int]]
        if not dist.is_initialized():
            return grouped_sample_idxes

        if dist.get_world_size() <= 1:
            return grouped_sample_idxes

        keys = sorted(list(grouped_sample_idxes.keys()))
        counts = torch.as_tensor([len(grouped_sample_idxes[k]) for k in keys],
                                 dtype=torch.long, device=f"cuda:{dist_utils.get_local_rank()}")  # [num_groups]

        counts_all = [torch.zeros_like(counts) for _ in range(dist.get_world_size())]
        dist.all_gather(counts_all, counts)

        counts_all = torch.stack(counts_all, 1)  # [num_groups, world_size]
        max_count = counts_all.max(1).values.tolist()  # [num_groups]

        for k, k_count, k_max_count in zip(keys, counts.tolist(), max_count):
            n_pad = k_max_count - k_count
            if n_pad == 0:
                continue

            pad_idxes = grouped_sample_idxes[k][:n_pad]
            grouped_sample_idxes[k].extend(pad_idxes)

        return grouped_sample_idxes

    @staticmethod
    def check_group_id_consistency(batch_group_ids):
        if not dist.is_initialized():
            return

        if dist.get_world_size() <= 1:
            return

        batch_group_ids = batch_group_ids.float().to(device=f"cuda:{dist_utils.get_local_rank()}")
        current_process_batch_group_ids = batch_group_ids.clone()
        dist.all_reduce(batch_group_ids, dist.ReduceOp.AVG)
        equality_flag = batch_group_ids == current_process_batch_group_ids
        if not torch.all(equality_flag):
            num_violations = equality_flag.numel() - equality_flag.sum().item()
            raise RuntimeError(f"Batch sample group IDs are inconsistent across processes. "
                               f"Number of violations: {num_violations}")

    def __iter__(self):
        for i in range(len(self)):
            yield self.batch_sample_indices[i].tolist()

    def __len__(self):
        return len(self.batch_sample_indices)
