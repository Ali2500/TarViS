from abc import abstractmethod
from typing import Dict, Any, List, Optional
from PIL import Image

from tarvis.inference.result_formatters import ResultFormatterBase

import pycocotools.mask as mt
import json
import numpy as np
import os
import os.path as osp
import shutil
import zipfile


class VOSResultFormatter(ResultFormatterBase):
    def __init__(self, output_dir: str, color_palette_filename: str, eval_frames: Optional[Dict[str, List[str]]] = None):
        super().__init__(output_dir)

        color_palette_path = osp.join(osp.dirname(__file__), "color_palettes", color_palette_filename)
        assert osp.exists(color_palette_path)
        self.color_palette = np.load(color_palette_path)
        self.eval_frames = eval_frames

    def add_sequence_result(self, accumulator_output: Dict[str, Any], sequence_info: Dict[str, Any]):
        seq_track_rles = accumulator_output["track_mask_rles"]
        assert len(seq_track_rles) == len(sequence_info["image_paths"])
        seq_output_dir = osp.join(self.output_dir, sequence_info["dirname"])
        os.makedirs(seq_output_dir, exist_ok=True)

        for t, (rles_t, img_path_t) in enumerate(zip(seq_track_rles, sequence_info["image_paths"])):
            filename = osp.split(img_path_t)[-1]
            if self.eval_frames is not None:
                if filename not in self.eval_frames[sequence_info["dirname"]]:
                    continue

            output_mask = np.zeros(sequence_info["image_dims"], np.uint8)

            for instance_id, rle in rles_t.items():
                instance_mask = mt.decode(rle).astype(np.uint8)
                assert instance_mask.shape == output_mask.shape, \
                    f"Shape mismatch: {instance_mask.shape} =/= {output_mask.shape}"
                output_mask = np.where(instance_mask, instance_id, output_mask)

            output_mask_path = osp.join(seq_output_dir, filename.replace(".jpg", ".png"))
            output_mask = Image.fromarray(output_mask)
            output_mask.putpalette(self.color_palette)

            with open(output_mask_path, 'wb') as fh:
                output_mask.save(fh)

        return accumulator_output

    @abstractmethod
    def finalize_output(self):
        pass


class DavisResultFormatter(VOSResultFormatter):
    def __init__(self, output_dir: str, split: str):
        super().__init__(output_dir=output_dir, color_palette_filename="davis.npy")
        assert split in ("val", "testdev")
        self.split = split

    def finalize_output(self):
        if self.split == "val":
            return

        zip_out_path = osp.join(osp.dirname(self.output_dir), "formatted_results.zip")

        with zipfile.ZipFile(zip_out_path, 'w') as zf:
            for dirname in os.listdir(self.output_dir):
                if not osp.isdir(osp.join(self.output_dir, dirname)):
                    continue

                filenames = os.listdir(osp.join(self.output_dir, dirname))
                for f in filenames:
                    zf.write(osp.join(self.output_dir, dirname, f), arcname=osp.join(dirname, f))
