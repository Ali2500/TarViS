from typing import Optional, Dict, Any
from tarvis.utils.distributed import is_main_process

import datetime
import os
import logging
import wandb
import wandb.util


class WeightsAndBiasesLogger:
    NUM_INSTANCES = 0

    def __init__(self,
                 project,
                 run_name,
                 run_id,
                 resume_arg,
                 config,
                 suppress_console_output,
                 suppress_failure):

        self._init_success = False
        self._enabled = True

        self._run_id = run_id
        self._project = project
        self._run_name = run_name

        if not is_main_process():
            return

        if WeightsAndBiasesLogger.NUM_INSTANCES > 0:
            raise ValueError("This class should only be instantiated once!")

        try:
            if suppress_console_output:
                os.environ["WANDB_SILENT"] = "true"
                logging.getLogger("wandb").setLevel(logging.ERROR)

            wandb.login()

            if not run_name:
                run_name = datetime.datetime.now().strftime("%m.%d.%Y_%H.%M.%S")

            # self._run_id = wandb.util.generate_id()
            wandb.init(
                project=project,
                name=run_name,
                config=config,
                id=run_id,
                resume=resume_arg,
            )
            self._init_success = True
            WeightsAndBiasesLogger.NUM_INSTANCES += 1

        except Exception as exc:
            if suppress_failure:
                if is_main_process():
                    print(f"wandb initialization failed with the following exception:\n"
                          f"{str(exc)}")
                pass

    def __del__(self):
        if self._init_success:
            wandb.finish()

    @property
    def is_initialized(self):
        return self._init_success

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def log(self, log_dict: Dict[str, float], elapsed_iterations: Optional[int] = None):
        if not self._init_success or not self._enabled:
            return

        wandb.log(data=log_dict, step=elapsed_iterations)

    def state_dict(self):
        return {
            "run_id": self._run_id,
            "project": self._project,
            "run_name": self._run_name,
            "type": "wandb"
        }

    @classmethod
    def create(cls,
               project: str,
               run_name: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None,
               suppress_console_output: bool = True,
               suppress_failure: bool = False):

        run_id = wandb.util.generate_id()
        return cls(
            project=project,
            run_name=run_name,
            run_id=run_id,
            resume_arg=False,
            config=config,
            suppress_console_output=suppress_console_output,
            suppress_failure=suppress_failure
        )

    @classmethod
    def restore_from_checkpoint(cls,
                                state_dict: Dict[str, Any],
                                suppress_console_output: bool = True,
                                suppress_failure: bool = False):

        return cls(
            project=state_dict["project"],
            run_name=state_dict["run_name"],
            run_id=state_dict["run_id"],
            resume_arg="must",
            config=None,
            suppress_console_output=suppress_console_output,
            suppress_failure=suppress_failure
        )
