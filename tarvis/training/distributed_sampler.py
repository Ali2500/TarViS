import math
import torch
from collections import defaultdict
from torch.utils.data import Sampler
import tarvis.utils.distributed as dist_utils


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(None)

        if num_replicas is None:
            num_replicas = dist_utils.get_world_size()

        if rank is None:
            rank = dist_utils.get_rank()

        assert len(dataset) % num_replicas == 0, \
            f"Dataset length = {len(dataset)}, number of processes = {num_replicas}"

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

    def __iter__(self):
        sample_ds_names = self.dataset.sample_dataset_names()

        # Group the samples by dataset. We want that each process should get an equal number of samples from each
        # dataset
        grouped_sample_idxes = defaultdict(list)
        for i, ds_name in enumerate(sample_ds_names):
            grouped_sample_idxes[ds_name].append(i)

        # sample separately per process rank within each group
        rank_sample_idxes = []
        rank_samples_created_so_far = 0

        for i, ds_name in enumerate(sorted(grouped_sample_idxes.keys())):
            idxes = torch.as_tensor(list(grouped_sample_idxes[ds_name]), dtype=torch.long)

            if i == len(grouped_sample_idxes) - 1:
                target_num_ds_samples = (len(self.dataset) // self.num_replicas) - rank_samples_created_so_far
            else:
                target_num_ds_samples = int(round(len(idxes) / float(self.num_replicas)))

            rank_ds_idxes = idxes[self.rank:len(idxes):self.num_replicas]

            if len(rank_ds_idxes) < target_num_ds_samples:
                n_pad = target_num_ds_samples - len(rank_ds_idxes)
                rank_ds_idxes = torch.cat((rank_ds_idxes, rank_ds_idxes[:n_pad]))

            elif len(rank_ds_idxes) > target_num_ds_samples:
                rank_ds_idxes = rank_ds_idxes[:target_num_ds_samples]

            assert len(rank_ds_idxes) == target_num_ds_samples, \
                f"Sanity check failed. ds_name={ds_name}, len(rank_ds_idxes)={len(rank_ds_idxes)}, " \
                f"target_num_ds_samples={target_num_ds_samples}"

            rank_sample_idxes.append(rank_ds_idxes)
            rank_samples_created_so_far += target_num_ds_samples

        rank_sample_idxes = torch.cat(rank_sample_idxes)
        if self.shuffle:
            rank_sample_idxes = rank_sample_idxes[torch.randperm(rank_sample_idxes.numel())]

        return iter(rank_sample_idxes.tolist())

    def __len__(self):
        return len(self.dataset) // self.num_replicas

    def set_epoch(self, epoch):
        pass


def _test():
    class DummyDataset:
        def __init__(self):
            self.ds_names = [
                "COCO",
                "COCO",
                "ADE20k",
                "ADE20k",
                "ADE20k",
                "YouTube-VIS",
                "COCO",
                "COCO",
                "YouTube-VIS",
                "YouTube-VIS",
                "COCO",
                "COCO"
            ]

        def __len__(self):
            return len(self.ds_names)

        def sample_dataset_names(self):
            return self.ds_names

    world_size = 3
    for rank in range(world_size):
        sampler = DistributedSampler(DummyDataset(), world_size, rank)
        print(f"Rank {rank}: {list(sampler)}")
        print(f"Rank {rank} names: {[DummyDataset().ds_names[i] for i in sampler]}")


if __name__ == '__main__':
    _test()
