from typing import Tuple
from PIL import Image
from typing import List, Dict
from tarvis.inference.dataset_parser import DatasetParserBase

import json
import numpy as np
import os
import os.path as osp


class BurstDatasetParser(DatasetParserBase):
    def __init__(self, images_base_dir: str, first_frame_annotations_file: str, mode: str, num_interleaved_frames: int):
        super().__init__("vos", "BURST")

        assert mode in ("mask", "point")
        self.mode = mode
        self.images_base_dir = images_base_dir

        with open(first_frame_annotations_file, 'r') as fh:
            content = json.load(fh)

        self.first_frame_annotations_file = first_frame_annotations_file
        self.sequence_image_filenames: Dict[str, List[str]] = dict()
        self.sequence_object_points: Dict[str, List[Dict[int, Tuple[int, int]]]] = dict()
        self.sequence_object_mask_rles: Dict[str, List[Dict[int, bytes]]] = dict()
        self.sequence_image_dims = dict()
        self.sequence_first_object_frame_index = dict()
        self.sequence_dirnames = []

        for seq in content["sequences"]:
            dirname = f"{seq['dataset']}/{seq['seq_name']}"
            # if dirname != "YFCC100M/v_25685519b728afd746dfd1b2fe77c":
            #     continue

            image_filenames = self.sample_images(seq["all_image_paths"], seq["fps"], num_interleaved_frames)

            seq_length = len(image_filenames)
            image_dims = (seq['height'], seq['width'])

            self.sequence_dirnames.append(dirname)
            self.sequence_image_filenames[dirname] = image_filenames
            self.sequence_image_dims[dirname] = image_dims
            self.sequence_object_points[dirname] = [dict() for _ in range(seq_length)]
            self.sequence_object_mask_rles[dirname] = [dict() for _ in range(seq_length)]
            self.sequence_first_object_frame_index[dirname] = int(1e6)

            frame_index_mapping = self.get_frame_index_mapping(image_filenames, seq['annotated_image_paths'])

            for t, segs_t in enumerate(seq['segmentations']):
                t = frame_index_mapping[t]

                for track_id, seg in segs_t.items():
                    track_id = int(track_id)
                    point_coords = (seg["point"]["y"], seg["point"]["x"])

                    # normalize point_coords by image size
                    # the '+0.5' is for accuracy when using F.grid_sample to sample features at normalized coordinates
                    point_coords = [(float(x) + 0.5) / dim for x, dim in zip(point_coords, image_dims)]

                    self.sequence_object_points[dirname][t][track_id] = point_coords  # (y, x), in range [0, 1]
                    self.sequence_object_mask_rles[dirname][t][track_id] = seg["rle"]  # rle mask
                    self.sequence_first_object_frame_index[dirname] = min(self.sequence_first_object_frame_index[dirname], t)

    def sample_images(self, all_image_filenames: List[str], video_fps: int, num_interleaved_frames: int):
        assert num_interleaved_frames >= 0
        num_interleaved_frames = min(num_interleaved_frames, video_fps - 2)
        target_sample_rate = np.round(np.linspace(0, video_fps, num_interleaved_frames + 2)[:-1])
        target_sample_rate = set(target_sample_rate.round().astype(int).tolist())

        sampled_indices = [t for t in range(len(all_image_filenames)) if t % video_fps in target_sample_rate]
        return [all_image_filenames[t] for t in sampled_indices]

    def get_frame_index_mapping(self, sampled_filenames: List[str], annotated_filenames: List[str]):
        assert len(sampled_filenames) >= len(annotated_filenames)
        mapping = dict()
        for t, fname in enumerate(annotated_filenames):
            assert fname in sampled_filenames, f"File {fname} not found in sampled image filenames"
            sampled_t = sampled_filenames.index(fname)
            mapping[t] = sampled_t

        return mapping

    def __getitem__(self, index):
        dirname = self.sequence_dirnames[index]

        seq_info = self.get_base_seq_info()

        seq_info.update({
            "dirname": dirname,
            "image_paths": [osp.join(self.images_base_dir, dirname, f) for f in self.sequence_image_filenames[dirname]],
            "image_dims": self.sequence_image_dims[dirname],
            "first_ref_mask_frame_index": self.sequence_first_object_frame_index[dirname]
        })

        if self.mode == "point":
            seq_info["first_frame_object_points"] = self.sequence_object_points[dirname]
        elif self.mode == "mask":
            seq_info["first_frame_mask_rles"] = self.sequence_object_mask_rles[dirname]
        else:
            raise ValueError("Should not be here")

        return seq_info
