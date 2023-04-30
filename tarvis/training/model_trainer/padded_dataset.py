from torch.utils.data import Dataset
from typing import List, Optional

import math
import torch
import warnings


class PaddedDataset(Dataset):
    def __init__(self, dataset: Dataset, size: int, attribs_to_inherit: Optional[List[str]] = []):
        """
        Constructor
        :param dataset:
        :param size:
        :param attribs_to_inherit: If `dataset` has any custom attributes (properties or functions) which we want the
        instance of PaddedDataset to also have, then provide the list of those attributes here.
        """
        super().__init__()
        self._dataset = dataset
        if size < len(self._dataset):
            warnings.warn("Given size for padded dataset is smaller than the original size of the dataset."
                          "Some samples will be omitted to achieve this.", UserWarning)
        self.size = size
        whole_reps = int(math.floor(float(size) / float(len(self._dataset))))
        self.sample_idxes = list(range(len(self._dataset))) * whole_reps

        n_padded_samples = size % len(self._dataset)

        # pad with the same random samples across all parallel processes
        rng_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        self.sample_idxes.extend(torch.randint(len(self._dataset), (n_padded_samples,), dtype=torch.long).tolist())
        torch.random.set_rng_state(rng_state)

        for name in attribs_to_inherit:
            assert not hasattr(self, name), f"Attribute {name} already exists in PaddedDataset instance"
            assert hasattr(dataset, name), f"Attribute {name} not found in given dataset instance"
            setattr(self, name, getattr(dataset, name))

    def get_orig_idx(self, index):
        return self.sample_idxes[index]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self._dataset[self.sample_idxes[index]]
