from typing import Dict, Any, Optional

from tarvis.inference.result_formatters import ResultFormatterBase
from tarvis.inference.misc import split_by_job_id

import json
import os
import os.path as osp


class BURSTResultFormatter(ResultFormatterBase):
    def __init__(self, output_dir: str, annotations_json: str, seq_split_spec: Optional[str] = None):
        super().__init__(output_dir)

        with open(annotations_json, 'r') as fh:
            self.json_content = json.load(fh)

        self.sequences_content = self.json_content["sequences"]
        self.json_content["sequences"] = []

        for seq in self.sequences_content:
            del seq["segmentations"]

        self.output_file_suffix = ""

        # handle distributed inference with slurm job array 
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            self.output_file_suffix = f"_part_{os.environ['SLURM_ARRAY_TASK_ID']}"

        if seq_split_spec:
            start_idx, end_idx, _ = split_by_job_id(len(self.sequences_content), seq_split_spec)
            self.output_file_suffix = f"_part_{start_idx}_{end_idx}"

    def add_sequence_result(self, accumulator_output: Dict[str, Any], sequence_info: Dict[str, Any]):
        seq_track_rles = accumulator_output["track_mask_rles"]
        assert len(seq_track_rles) == len(sequence_info["image_paths"])
        dataset, seq_name = sequence_info["dirname"].split("/")

        seq_dict = [seq for seq in self.sequences_content if seq["dataset"] == dataset and seq["seq_name"] == seq_name]
        assert len(seq_dict) == 1, f"seq_dict length = {len(seq_dict)} for dataset = {dataset}, seq_name = {seq_name}"
        seq_dict = seq_dict[0]
        seq_dict["segmentations"] = [dict() for _ in range(len(seq_dict["annotated_image_paths"]))]

        for t, (segs_t, img_path_t) in enumerate(zip(seq_track_rles, sequence_info["image_paths"])):
            filename_t = osp.split(img_path_t)[-1]
            if filename_t not in seq_dict["annotated_image_paths"]:
                continue

            t_eval = seq_dict["annotated_image_paths"].index(filename_t)

            for instance_id, seg in segs_t.items():
                seq_dict["segmentations"][t_eval][instance_id] = {
                    "rle": seg["counts"].decode("utf-8"),
                    "is_gt": False
                }

        self.json_content["sequences"].append(seq_dict)

        return accumulator_output

    def finalize_output(self):
        with open(osp.join(self.output_dir, f"results{self.output_file_suffix}.json"), 'w') as fh:
            json.dump(self.json_content, fh)
