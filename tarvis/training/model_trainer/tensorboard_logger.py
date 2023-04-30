import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore annoying tensorflow/tensorboard info messages

from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
from tarvis.utils.distributed import is_main_process


class TensorboardLogger:
    def __init__(self, output_dir):
        self._enabled = True
        self._init_success = False
        self._output_dir = output_dir

        if not is_main_process():
            return

        os.makedirs(output_dir, exist_ok=True)

        self.writer = SummaryWriter(output_dir)
        self._init_success = True

    def __del__(self):
        if self._init_success:
            self.writer.close()

    @property
    def is_initialized(self):
        return self._init_success

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def log(self, log_dict: Dict[str, float], elapsed_iterations: int):
        if not self._init_success or not self._enabled:
            return

        for name, value in log_dict.items():
            self.writer.add_scalar(tag=name, scalar_value=value, global_step=elapsed_iterations, new_style=True)

    def state_dict(self):
        return {
            "output_dir": self._output_dir,
            "type": "tensorboard"
        }

    @classmethod
    def create(cls, output_dir: str):
        return cls(
            output_dir=output_dir
        )

    @classmethod
    def restore_from_checkpoint(cls, state_dict: Dict[str, Any], override_output_dir: Optional[str] = None):
        if override_output_dir:
            return cls(output_dir=override_output_dir)

        return cls(
            output_dir=state_dict["output_dir"]
        )
