import json
import os.path as osp
from tarvis.inference.dataset_parser import DatasetParserBase


class YoutubeVISParser(DatasetParserBase):
    def __init__(self, images_base_dir: str, json_info_path: str):
        super().__init__("instance_seg", "YOUTUBE_VIS")

        self.images_base_dir = images_base_dir

        with open(json_info_path, 'r') as fh:
            content = json.load(fh)

        self.year = content['info']['year']

        # populate image paths from JSON file contents
        self.dirname_to_video_id = dict()
        self.dirname_to_image_dims = dict()

        for vid in content["videos"]:
            dirname = vid["file_names"][0].split("/")[0]
            filenames = [f.split("/")[1] for f in vid["file_names"]]

            self.sequence_dirnames.append(dirname)
            self.sequence_image_filenames[dirname] = filenames
            self.dirname_to_video_id[dirname] = vid["id"]
            self.dirname_to_image_dims[dirname] = (vid["height"], vid["width"])

        print(f"Maximum sequence length: {max([len(fnames) for fnames in self.sequence_image_filenames.values()])}")

    @property
    def category_labels(self):
        if self.year == 2019:
            return self._category_labels_2019
        elif self.year in (2021, 2022):
            return self._category_labels_2021
        else:
            raise ValueError(f"Unexpected dataset version/year: {self.year}")

    def __getitem__(self, index):
        dirname = self.sequence_dirnames[index]
        seq_info = self.get_base_seq_info()

        seq_info.update({
            "dirname": dirname,
            "image_paths": [osp.join(self.images_base_dir, dirname, f) for f in self.sequence_image_filenames[dirname]],
            "video_id": self.dirname_to_video_id[dirname],
            "image_dims": self.dirname_to_image_dims[dirname]
        })

        return seq_info

    _category_labels_2019 = {
        1: 'person',
        2: 'giant_panda',
        3: 'lizard',
        4: 'parrot',
        5: 'skateboard',
        6: 'sedan',
        7: 'ape',
        8: 'dog',
        9: 'snake',
        10: 'monkey',
        11: 'hand',
        12: 'rabbit',
        13: 'duck',
        14: 'cat',
        15: 'cow',
        16: 'fish',
        17: 'train',
        18: 'horse',
        19: 'turtle',
        20: 'bear',
        21: 'motorbike',
        22: 'giraffe',
        23: 'leopard',
        24: 'fox',
        25: 'deer',
        26: 'owl',
        27: 'surfboard',
        28: 'airplane',
        29: 'truck',
        30: 'zebra',
        31: 'tiger',
        32: 'elephant',
        33: 'snowboard',
        34: 'boat',
        35: 'shark',
        36: 'mouse',
        37: 'frog',
        38: 'eagle',
        39: 'earless_seal',
        40: 'tennis_racket'
    }

    _category_labels_2021 = {
        1: 'airplane',
        2: 'bear',
        3: 'bird',
        4: 'boat',
        5: 'car',
        6: 'cat',
        7: 'cow',
        8: 'deer',
        9: 'dog',
        10: 'duck',
        11: 'earless_seal',
        12: 'elephant',
        13: 'fish',
        14: 'flying_disc',
        15: 'fox',
        16: 'frog',
        17: 'giant_panda',
        18: 'giraffe',
        19: 'horse',
        20: 'leopard',
        21: 'lizard',
        22: 'monkey',
        23: 'motorbike',
        24: 'mouse',
        25: 'parrot',
        26: 'person',
        27: 'rabbit',
        28: 'shark',
        29: 'skateboard',
        30: 'snake',
        31: 'snowboard',
        32: 'squirrel',
        33: 'surfboard',
        34: 'tennis_racket',
        35: 'tiger',
        36: 'train',
        37: 'truck',
        38: 'turtle',
        39: 'whale',
        40: 'zebra'
    }


def parse_json(filepath: str):
    with open(filepath, 'r') as fh:
        content = json.load(fh)

    seq_dirs = []
    meta_info = {
        "seq_dirname_to_index_mapping": dict(),
        "json_content": content
    }

    for i, seq in enumerate(content["videos"]):
        dirname = seq["file_names"][0].split("/")[0]
        seq_dirs.append(dirname)
        meta_info["seq_dirname_to_index_mapping"][dirname] = i

    return seq_dirs, meta_info
