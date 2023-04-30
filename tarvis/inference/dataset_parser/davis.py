import os
import os.path as osp

from PIL import Image
from typing import List, Dict
from tarvis.inference.dataset_parser import DatasetParserBase


class DavisDatasetParser(DatasetParserBase):
    def __init__(self, images_base_dir: str, annotations_base_dir: str, image_set_file_path: str):
        super().__init__("vos", "DAVIS")

        self.images_base_dir = images_base_dir

        self.annotations_base_dir = annotations_base_dir
        self.sequence_annotation_filenames: Dict[str, List[str]] = dict()
        self.dirname_to_image_dims = dict()

        with open(image_set_file_path, 'r') as fh:
            self.sequence_dirnames = [x.strip() for x in fh.readlines()]

        self.populate_image_paths()

    def populate_image_paths(self):
        for dirname in self.sequence_dirnames:
            self.sequence_image_filenames[dirname] = sorted([
                f for f in os.listdir(osp.join(self.images_base_dir, dirname)) if
                f.endswith(".jpg")
            ])

            self.sequence_annotation_filenames[dirname] = sorted([
                f for f in os.listdir(osp.join(self.annotations_base_dir, dirname)) if
                f.endswith(".png")
            ])[:1]  # only take the first file

            # open up one file to get image dims
            with open(osp.join(self.annotations_base_dir, dirname, self.sequence_annotation_filenames[dirname][0]),
                      'rb') as fh:
                img = Image.open(fh)
                width, height = img.size
                self.dirname_to_image_dims[dirname] = (height, width)

    def __getitem__(self, index):
        dirname = self.sequence_dirnames[index]

        seq_info = self.get_base_seq_info()

        seq_info.update({
            "dirname": dirname,
            "image_paths": [osp.join(self.images_base_dir, dirname, f) for f in self.sequence_image_filenames[dirname]],
            "image_dims": self.dirname_to_image_dims[dirname],
            "first_frame_mask_paths": [
                osp.join(self.annotations_base_dir, dirname, f) for f in self.sequence_annotation_filenames[dirname]
            ],
            "first_ref_mask_frame_index": 0
        })

        return seq_info
