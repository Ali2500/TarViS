from typing import Dict, Any

from tarvis.inference.visualization.instance_seg_and_vos import SequenceVisualizer as InstanceSegAndVOSVisualizer
from tarvis.inference.visualization.panoptic_seg import SequenceVisualizer as PanopticSequenceVisualizer
from tarvis.utils.timer import Timer


@Timer.log_duration("vizualization")
def save_vizualization(task_type: str,
                       output_dir: str,
                       sequence_results: Dict[str, Any],
                       sequence_info: Dict[str, Any],
                       category_labels: Dict[int, str],
                       num_processes: int):

    if task_type in ("instance_seg", "vos"):
        InstanceSegAndVOSVisualizer(
            task_type=task_type,
            sequence_results=sequence_results,
            sequence_info=sequence_info,
            category_labels=category_labels,
            num_processes=num_processes
        ).save(output_dir)

    elif task_type == "panoptic_seg":
        PanopticSequenceVisualizer(
            sequence_results=sequence_results,
            sequence_info=sequence_info,
            category_labels=category_labels,
            num_processes=num_processes
        ).save(output_dir)

    else:
        raise ValueError(f"Invalid task type: {task_type}")
