from typing import List
from torch.utils.data import Dataset, ConcatDataset as _ConcatDatasetBase

import itertools


class TrainingDatasetBase(Dataset):
    def __init__(self, name: str, task_type: str):
        super().__init__()

        self.name = name
        self.task_type = task_type
        self.sample_image_dims = []
        self.sample_instance_counts = []

    def sample_type_ids(self):
        if self.task_type in ("instance_seg", "panoptic_seg"):
            return [f"{self.name}_{self.task_type}"] * len(self.sample_image_dims)

        elif self.task_type == "vos":
            return [f"{self.name}_{self.task_type}_{num_objects}" for num_objects in self.sample_instance_counts]

    def sample_dataset_names(self):
        return [self.name] * len(self.sample_image_dims)


class ConcatDataset(_ConcatDatasetBase):
    def __init__(self, datasets: List[TrainingDatasetBase]):
        super().__init__(datasets)

        self.sample_image_dims = list(itertools.chain(*[ds.sample_image_dims for ds in self.datasets]))

    def sample_type_ids(self):
        return list(itertools.chain(*[ds.sample_type_ids() for ds in self.datasets]))

    def sample_dataset_names(self):
        return list(itertools.chain(*[ds.sample_dataset_names() for ds in self.datasets]))
