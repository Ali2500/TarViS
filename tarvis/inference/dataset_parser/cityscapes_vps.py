from collections import defaultdict
from glob import glob
from tarvis.inference.dataset_parser import DatasetParserBase

import cv2
import json
import os.path as osp


class CityscapesVPSParser(DatasetParserBase):
    def __init__(self, images_base_dir: str, info_file: str, sparse_frames: bool):
        super().__init__("panoptic_seg", "CITYSCAPES_VPS")

        self.images_base_dir = images_base_dir

        with open(info_file, 'r') as fh:
            info = json.load(fh)

        clip_length = 6 if sparse_frames else 30

        # group images into sequences
        # print(f"Total images: {len(info['images'])}")
        sequences = dict()
        for img in info['images']:
            seq_id = img['id'] // 10000
            frame_id = img['id'] % 10000  # 1-based index

            if sparse_frames:
                if frame_id % 5 != 0:
                    continue
                frame_id = frame_id // 5

            if seq_id not in sequences:
                sequences[seq_id] = {
                    "filenames": [None for _ in range(clip_length)],
                    "img_dims": (img['height'], img['width'])
                }
            sequences[seq_id]["filenames"][frame_id - 1] = img['file_name']

        # populate image paths from JSON file contents
        self.dirname_to_video_id = dict()
        self.dirname_to_image_dims = dict()

        for seq_id, info in sequences.items():
            assert all([f is not None for f in info['filenames']])
            self.sequence_dirnames.append(str(seq_id))
            self.sequence_image_filenames[str(seq_id)] = info['filenames']
            self.dirname_to_video_id[str(seq_id)] = seq_id
            self.dirname_to_image_dims[str(seq_id)] = info['img_dims']  # height, width

    @property
    def category_labels(self):
        return self._category_labels

    def __getitem__(self, index):
        dirname = self.sequence_dirnames[index]

        seq_info = self.get_base_seq_info()
        seq_info.update({
            "dirname": dirname,
            "image_paths": [osp.join(self.images_base_dir, f) for f in self.sequence_image_filenames[dirname]],
            "video_id": self.dirname_to_video_id[dirname],
            "image_dims": self.dirname_to_image_dims[dirname],
            "thing_class_ids": [11, 12, 13, 14, 15, 16, 17, 18]
        })

        return seq_info

    _category_labels = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
       # 255: "void"
    }
