from glob import glob
from tarvis.inference.dataset_parser import DatasetParserBase

import cv2
import json
import os.path as osp


class KittiStepParser(DatasetParserBase):
    def __init__(self, images_base_dir: str):
        super().__init__("panoptic_seg", "KITTI_STEP")

        self.images_base_dir = images_base_dir

        # load dataset filenames from metainfo file in tarvis/data/format_mappers/metainfo/kittimots_filenames.json
        # NOTE: KITTI-STEP shares the same images as KITTI-MOTS
        fpath = osp.join(
            osp.dirname(__file__), osp.pardir, osp.pardir, "data", "format_mappers", "metainfo",
            "kitti_step_filenames.json"
        )
        fpath = osp.realpath(fpath)
        assert osp.exists(fpath), f"Filenames JSON file not found at expected path: {fpath}"

        with open(fpath, 'r') as fh:
            content = json.load(fh)

        # populate image paths from JSON file contents
        self.dirname_to_video_id = dict()
        self.dirname_to_image_dims = dict()

        for seq_name in self._validation_seqs:
            filenames = content[seq_name]
            self.sequence_dirnames.append(seq_name)
            self.sequence_image_filenames[seq_name] = filenames
            self.dirname_to_video_id[seq_name] = int(seq_name)

            # load one image to get image dims
            first_image = cv2.imread(glob(osp.join(images_base_dir, seq_name, "*.png"))[0], cv2.IMREAD_COLOR)
            self.dirname_to_image_dims[seq_name] = first_image.shape[:2]  # height, width

        print(f"Maximum sequence length: {max([len(fnames) for fnames in self.sequence_image_filenames.values()])}")

    @property
    def category_labels(self):
        return self._category_labels

    def __getitem__(self, index):
        dirname = self.sequence_dirnames[index]

        seq_info = self.get_base_seq_info()
        seq_info.update({
            "dirname": dirname,
            "image_paths": [osp.join(self.images_base_dir, dirname, f) for f in self.sequence_image_filenames[dirname]],
            "video_id": self.dirname_to_video_id[dirname],
            "image_dims": self.dirname_to_image_dims[dirname],
            "thing_class_ids": [11, 13]
        })

        return seq_info

    _validation_seqs = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']

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
