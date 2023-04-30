import json
import os.path as osp
from tarvis.inference.dataset_parser import DatasetParserBase


class OVISParser(DatasetParserBase):
    def __init__(self, images_base_dir: str, json_info_path: str):
        super().__init__("instance_seg", "OVIS")

        self.images_base_dir = images_base_dir

        with open(json_info_path, 'r') as fh:
            content = json.load(fh)

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
        return self._category_labels

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

    _category_labels = {
        1: "Person",
        2: "Bird",
        3: "Cat",
        4: "Dog",
        5: "Horse",
        6: "Sheep",
        7: "Cow",
        8: "Elephant",
        9: "Bear",
        10: "Zebra",
        11: "Giraffe",
        12: "Poultry",
        13: "Giant_panda",
        14: "Lizard",
        15: "Parrot",
        16: "Monkey",
        17: "Rabbit",
        18: "Tiger",
        19: "Fish",
        20: "Turtle",
        21: "Bicycle",
        22: "Motorcycle",
        23: "Airplane",
        24: "Boat",
        25: "Vehicle"
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
