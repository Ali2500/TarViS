from typing import Dict, Any, Optional
from tarvis.utils import distributed as dist_utils

import os
import os.path as osp
import logging
import torch


class CheckpointManager:
    def __init__(self,
                 logger: logging.Logger,
                 checkpoint_dir: str,
                 save_interval: int,
                 start_saving_after: int,
                 max_num_to_keep: int):

        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.start_saving_after = start_saving_after
        self.max_num_to_keep = max_num_to_keep

        self._save_history = []

    def saving_required(self, elapsed_iters: int):
        return elapsed_iters % self.save_interval == 0 and elapsed_iters > self.start_saving_after

    def save(self, state_dict: Dict[str, Any], elapsed_iters: int):
        filename = f"{elapsed_iters:06d}.pth"
        save_path = osp.join(self.checkpoint_dir, filename)

        # add the checkpoint which we're going to save to the state dict
        state_dict["checkpoint_manager"]["_save_history"].append((elapsed_iters, filename))
        
        if dist_utils.is_main_process():
            torch.save(state_dict, save_path)
            self.logger.info(f"Checkpoint saved to: {save_path}")

        self._save_history.append((elapsed_iters, filename))

        if self.max_num_to_keep <= 0:
            return

        if len(self._save_history) > self.max_num_to_keep:
            num_to_remove = len(self._save_history) - self.max_num_to_keep
            if dist_utils.is_main_process():
                for i in range(num_to_remove):
                    path = osp.join(self.checkpoint_dir, self._save_history[i][1])
                    self.logger.info(f"Deleting outdated checkpoint: {path}")
                    try:
                        os.remove(path)
                    except FileNotFoundError as _:
                        pass  # file was already deleted in a previous run or manually by user
                    except PermissionError as _:
                        self.logger.warn(f"Could not delete outdated checkpoint because of directory permissions.")
                    except Exception as exc:
                        self.logger.warn(f"Failed to delete outdated checkpoint at {path}\n"
                                         f"Error message: {str(exc)}")
                
            self._save_history = self._save_history[num_to_remove:]

    def get_save_iterations(self):
        return [x[0] for x in self._save_history]

    def get_last_save_iterations(self):
        save_iterations = self.get_save_iterations()
        if len(save_iterations) > 0:
            return self.get_save_iterations()[-1]
        else:
            return -1

    def state_dict(self):
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "save_interval": self.save_interval,
            "start_saving_after": self.start_saving_after,
            "max_num_to_keep": self.max_num_to_keep,
            "_save_history": self._save_history.copy()
        }

    @classmethod
    def create(cls,
               logger: logging.Logger,
               checkpoint_dir: str,
               save_interval: int,
               start_saving_after: int = 0,
               max_num_to_keep: int = -1
               ):
        return cls(
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            save_interval=save_interval,
            start_saving_after=start_saving_after,
            max_num_to_keep=max_num_to_keep
        )

    @classmethod
    def restore_from_checkpoint(cls,
                                logger: logging.Logger,
                                state_dict: Dict[str, Any],
                                checkpoint_dir: Optional[str] = None,
                                save_interval: Optional[int] = None,
                                start_saving_after: Optional[int] = None,
                                max_num_to_keep: Optional[int] = None
                                ):
        inst = cls(
            logger=logger,
            checkpoint_dir=state_dict["checkpoint_dir"],
            save_interval=state_dict["save_interval"],
            start_saving_after=state_dict["start_saving_after"],
            max_num_to_keep=state_dict["max_num_to_keep"]
        )

        inst._save_history = state_dict["_save_history"]

        if checkpoint_dir is not None:
            inst.checkpoint_dir = checkpoint_dir

        if save_interval is not None:
            inst.save_interval = save_interval

        if start_saving_after is not None:
            inst.start_saving_after = start_saving_after

        if max_num_to_keep is not None:
            inst.max_num_to_keep = max_num_to_keep

        return inst
