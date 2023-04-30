from abc import abstractmethod
from typing import List, Dict, Optional
from tarvis.inference.misc import split_by_job_id

import os
import os.path as osp


class DatasetParserBase:
    def __init__(self, task_type, name):
        assert task_type in ("instance_seg", "vos", "panoptic_seg")
        self.task_type = task_type
        self.name = name

        self.images_base_dir: str = ""
        self.sequence_dirnames: List[str] = []
        self.sequence_image_filenames: Dict[str, List[str]] = dict()
        self.sequence_image_dims: Dict[str, List[int]] = dict()

        self.first_frame_mask_paths: Dict[int, str] = dict()  # only used for VOS and PET

    def get_base_seq_info(self):
        return {
            "task_type": self.task_type,
            "dataset_name": self.name
        }

    def populate_image_paths(self):
        self.sequence_dirnames = [
            d for d in os.listdir(self.images_base_dir) if osp.isdir(osp.join(self.images_base_dir, d))
        ]

        for dirname in self.sequence_dirnames:
            self.sequence_image_filenames[dirname] = sorted([
                f for f in os.listdir(osp.join(self.images_base_dir, dirname)) if
                f.endswith(".jpg") or f.endswith(".png")
            ])

    def partition_sequences(self, split_spec: Optional[str] = None):
        start_idx, end_idx, _ = split_by_job_id(len(self), split_spec)
        self.sequence_dirnames = self.sequence_dirnames[start_idx:end_idx]
        print(f"Partitioned sequence range: {start_idx}-{end_idx} out of {len(self.sequence_dirnames)}")

    def isolate_sequences(self, seq_names: List[str]):
        assert len(seq_names) > 0
        for name in seq_names:
            if name not in self.sequence_dirnames:
                raise ValueError(f"Sequence {name} does not exist")
        self.sequence_dirnames = seq_names

    def __len__(self):
        return len(self.sequence_dirnames)

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def total_frames(self):
        return sum([len(image_filenames) for image_filenames in self.sequence_image_filenames.values()])

    category_labels = None
