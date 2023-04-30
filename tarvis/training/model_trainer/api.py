from abc import ABCMeta, abstractmethod
from colorama import Fore, Style, Back
from glob import glob
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
from collections import defaultdict, OrderedDict
from multiprocessing import cpu_count
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast

from tarvis.training.model_trainer.interrupt_detector import InterruptDetector
from tarvis.training.model_trainer.padded_dataset import PaddedDataset
from tarvis.training.model_trainer.default_distributed_sampler import DistributedSampler
from tarvis.training.model_trainer.wandb_logger import WeightsAndBiasesLogger
from tarvis.training.model_trainer.tensorboard_logger import TensorboardLogger
from tarvis.training.model_trainer.checkpoint_manager import CheckpointManager
from tarvis.training.model_trainer.timer_utils import ETAEstimator
from tarvis.training.model_trainer import misc_utils
from tarvis.utils import distributed as dist_utils

import contextlib
import datetime as dt
import time
import logging
import os
import os.path as osp
import torch
import torch.distributed as dist
import torch.nn as nn
import tarvis.training.model_trainer.timer_utils as timer_utils
import tarvis.training.model_trainer.logging_utils as log_utils


class InterruptException(RuntimeError):
    pass


class ModelTrainer(metaclass=ABCMeta):
    def __init__(self,
                 model: nn.Module,
                 model_save_dir: str,
                 log_dir: str,
                 console_logger: logging.Logger,
                 metrics_logger: Union[WeightsAndBiasesLogger, TensorboardLogger],
                 eta_estimator: ETAEstimator,
                 checkpoint_manager: CheckpointManager,
                 grad_scaler: Union[GradScaler, None],
                 state_params: Dict[str, Any]):

        self.model = model
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        self.console_logger = console_logger

        self.optimizers_and_lr_schedulers = []
        self.create_optimizers(self._model)

        self.metrics_logger = metrics_logger
        self.eta_estimator = eta_estimator
        self.checkpoint_manager = checkpoint_manager
        self.interrupt_detector = InterruptDetector()
        self.grad_scaler = grad_scaler
        self._state_params = state_params

        self.logging_buffer = dict()
        self.current_session_start_time = time.time()
        self.detect_anomaly = False
        self.ignore_oom_errors = False
        self.log_time_durations = False

        self.display_interval = 1
        self.summary_interval = -1
        self.image_summary_interval = -1

        self.setup()

    @classmethod
    def new(cls,
            model: nn.Module,
            model_save_dir: str,
            total_iterations: int,
            save_interval: int,
            restore_model_weights: Optional[str] = None,
            use_mixed_precision: Optional[bool] = False,
            find_model_unused_parameters: Optional[bool] = True,
            convert_sync_batchnorm: Optional[bool] = False,
            start_saving_checkpoints_after: Optional[int] = 0,
            max_checkpoints_to_keep: Optional[int] = -1,
            wandb_logging: Optional[bool] = False,
            wandb_project: Optional[str] = None,
            wandb_run: Optional[str] = None,
            wandb_config_dict: Optional[Dict[str, Any]] = None,
            console_logger: Optional[logging.Logger] = None,
            max_runtime: Optional[dt.timedelta] = dt.timedelta(days=1000)
            ):
        log_dir = osp.join(model_save_dir, "logs")

        if dist_utils.is_main_process():
            os.makedirs(log_dir, exist_ok=True)
        dist_utils.synchronize()  # wait until main process has created the log directory

        # initialize console logger
        if console_logger is None:
            if dist_utils.is_main_process():
                log_txt_file = osp.join(log_dir, "out.log")
            else:
                log_txt_file = osp.join(log_dir, f"out_rank{dist_utils.get_rank()}.log")

            console_logger = log_utils.create_console_logger(logging.INFO, logging.WARN, file_output_path=log_txt_file)
        else:
            console_logger = console_logger

        assert start_saving_checkpoints_after < total_iterations
        assert save_interval < total_iterations

        # ETA Estimator
        eta_estimator = ETAEstimator.create(
            total_iterations=total_iterations,
            num_iterations_to_discard=50
        )

        # Checkpoint manager
        checkpoint_manager = CheckpointManager.create(
            logger=console_logger,
            checkpoint_dir=model_save_dir,
            save_interval=save_interval,
            start_saving_after=start_saving_checkpoints_after,
            max_num_to_keep=max_checkpoints_to_keep
        )

        # WeightsAndBiases/Tensorboard logger
        if wandb_logging:
            assert wandb_project is not None
            metrics_logger = WeightsAndBiasesLogger.create(
                project=wandb_project,
                run_name=wandb_run,
                config=wandb_config_dict,
                suppress_console_output=True,
                suppress_failure=False
            )
        else:
            metrics_logger = TensorboardLogger(
                output_dir=osp.join(log_dir, "tensorboard")
            )

        # Gradient scaler for mixed precision
        gradient_scaler = None
        if use_mixed_precision:
            gradient_scaler = GradScaler()

        # Wrap model with DDP
        if dist_utils.is_distributed():
            if convert_sync_batchnorm:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            # For multi-node training, it is important to use the local rank here
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[dist_utils.get_local_rank()], output_device=dist_utils.get_local_rank(),
                find_unused_parameters=find_model_unused_parameters
            )

        max_runtime = (max_runtime.days * 3600 * 24) + max_runtime.seconds
        trainer_state_params = {
            "find_model_unused_parameters": find_model_unused_parameters,
            "convert_sync_batchnorm": convert_sync_batchnorm,
            "total_iterations": total_iterations,
            "elapsed_iterations": 0,
            "max_runtime": int(max_runtime)
        }

        trainer = cls(
            model=model,
            model_save_dir=model_save_dir,
            log_dir=log_dir,
            console_logger=console_logger,
            metrics_logger=metrics_logger,
            eta_estimator=eta_estimator,
            checkpoint_manager=checkpoint_manager,
            grad_scaler=gradient_scaler,
            state_params=trainer_state_params,
        )

        if restore_model_weights:
            checkpoint_data = torch.load(restore_model_weights, map_location=dist_utils.get_device())
            trainer.load_model_weights(checkpoint_data["model"])
            console_logger.info(f"Restored model weights from {restore_model_weights}")

        return trainer

    @classmethod
    def restore_from_checkpoint(cls,
                                model: nn.Module,
                                checkpoint_path: str,
                                console_logger: Optional[logging.Logger] = None,
                                max_runtime: Optional[dt.timedelta] = None
                                ):
        model_save_dir = osp.dirname(checkpoint_path)

        if dist_utils.is_main_process():
            # create a new log dir for every restoration to preserve the previous outputs
            log_dir_suffix = 1
            while True:
                log_dir = osp.join(model_save_dir, f'logs_{log_dir_suffix}')
                if osp.exists(log_dir):
                    log_dir_suffix += 1
                else:
                    break

            if dist_utils.is_main_process():
                os.makedirs(log_dir, exist_ok=True)

        dist_utils.synchronize()

        if not dist_utils.is_main_process():
            log_dir = sorted(glob(osp.join(model_save_dir, "logs_*")))[-1]

        # initialize console logger
        if console_logger is None:
            if dist_utils.is_main_process():
                log_txt_file = osp.join(log_dir, "out.log")
            else:
                log_txt_file = osp.join(log_dir, f"out_rank{dist_utils.get_rank()}.log")

            console_logger = log_utils.create_console_logger(logging.INFO, logging.WARN,
                                                             file_output_path=log_txt_file)
        else:
            console_logger = console_logger

        checkpoint_data = torch.load(checkpoint_path, map_location=dist_utils.get_device())

        def verify_checkpoint_key(key):
            if key not in checkpoint_data:
                raise KeyError(f"Required key '{key}' does not exist in checkpoint saved at {checkpoint_path}")

        verify_checkpoint_key("trainer_state_params")
        trainer_state_params = checkpoint_data["trainer_state_params"]

        # ETA estimator
        verify_checkpoint_key("eta_estimator")
        eta_estimator = ETAEstimator.restore_from_checkpoint(checkpoint_data["eta_estimator"])

        # Checkpoint manager
        verify_checkpoint_key("checkpoint_manager")
        checkpoint_manager = CheckpointManager.restore_from_checkpoint(
            logger=console_logger,
            state_dict=checkpoint_data["checkpoint_manager"]
        )

        # WeightsAndBiases/Tensorboard logger
        verify_checkpoint_key("metrics_logger")

        if checkpoint_data["metrics_logger"]["type"] == "tensorboard":

            metrics_logger = TensorboardLogger.restore_from_checkpoint(
                state_dict=checkpoint_data["metrics_logger"],
                override_output_dir=osp.join(log_dir, "tensorboard")
            )
        elif checkpoint_data["metrics_logger"]["type"] == "wandb":
            metrics_logger = WeightsAndBiasesLogger.restore_from_checkpoint(
                state_dict=checkpoint_data["metrics_logger"],
                suppress_console_output=True,
                suppress_failure=False
            )
        else:
            raise ValueError(f"Invalid metrics logger type found in checkpoint: "
                             f"'{checkpoint_data['metrics_logger']['type']}'")

        # Gradient scaler
        if "grad_scaler" in checkpoint_data:
            gradient_scaler = GradScaler()
            gradient_scaler.load_state_dict(checkpoint_data["grad_scaler"])
        else:
            gradient_scaler = None

        # Wrap model with DDP
        if dist_utils.is_distributed():
            if trainer_state_params["convert_sync_batchnorm"]:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            # For multi-node training, it is important to use the local rank here
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[dist_utils.get_local_rank()], output_device=dist_utils.get_local_rank(),
                find_unused_parameters=trainer_state_params["find_model_unused_parameters"]
            )

        if max_runtime is not None:
            max_runtime = (max_runtime.days * 3600 * 24) + max_runtime.seconds
            trainer_state_params["max_runtime"] = max_runtime

        trainer = cls(
            model=model,
            model_save_dir=model_save_dir,
            log_dir=log_dir,
            console_logger=console_logger,
            metrics_logger=metrics_logger,
            eta_estimator=eta_estimator,
            checkpoint_manager=checkpoint_manager,
            grad_scaler=gradient_scaler,
            state_params=trainer_state_params
        )

        # model weights
        verify_checkpoint_key("model")
        trainer.load_model_weights(checkpoint_data["model"])

        # restore state_dicts for optimizers and LR schedulers
        verify_checkpoint_key("optimizers_and_lr_schedulers")
        assert len(checkpoint_data["optimizers_and_lr_schedulers"]) == len(trainer.optimizers_and_lr_schedulers)
        for i, (optim_state_dict, lr_state_dicts) in enumerate(checkpoint_data["optimizers_and_lr_schedulers"]):
            trainer.optimizers_and_lr_schedulers[i][0].load_state_dict(optim_state_dict)

            if lr_state_dicts is not None:
                assert len(lr_state_dicts) == len(trainer.optimizers_and_lr_schedulers[i][1])
                for j, lrs_dict in enumerate(lr_state_dicts):
                    trainer.optimizers_and_lr_schedulers[i][1][j].load_state_dict(lrs_dict)

        # extra checkpoint state dict
        trainer.load_extra_checkpoint_state_dict(checkpoint_data.get("extra", dict()))

        return trainer

    @property
    def total_iterations(self):
        return self._state_params["total_iterations"]

    @property
    def elapsed_iterations(self):
        return self._state_params["elapsed_iterations"]

    @property
    def training_complete(self):
        return self._state_params["total_iterations"] <= self._state_params["elapsed_iterations"]

    @property
    def max_runtime(self):
        return self._state_params["max_runtime"]

    @property
    def num_gpus(self):
        return dist_utils.get_world_size()

    @property
    def local_rank(self):
        return dist_utils.get_local_rank()

    @property
    def global_rank(self):
        return dist_utils.get_rank()

    @property
    def local_device(self):
        return dist_utils.get_device()

    @property
    def mixed_precision_enabled(self):
        return self.grad_scaler is not None

    @property
    def _model(self):
        return self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

    def print(self, msg, *args, **kwargs):
        level = logging.getLevelName(kwargs.pop("level", "INFO"))
        assert isinstance(level, int), f"Given log level '{level}' is invalid"
        self.console_logger.log(level, msg, *args, **kwargs)

    def setup(self, *args, **kwargs):
        pass

    def register_optimizer(self, optimizer: torch.optim.Optimizer,
                           lr_schedulers: Optional[List[torch.optim.lr_scheduler._LRScheduler]] = None):
        assert isinstance(lr_schedulers, list) or lr_schedulers is None
        self.optimizers_and_lr_schedulers.append([optimizer, lr_schedulers])

    @abstractmethod
    def create_optimizers(self, model) -> Any:
        pass

    @abstractmethod
    def create_training_dataset(self, total_samples: int) -> Union[Dataset, Any]:
        pass

    @abstractmethod
    def forward(self, training_sample: Any, subiter: int) -> Any:
        pass

    @abstractmethod
    def apply_criterion(self, model_output: Any, training_sample: Any) -> Any:
        pass

    @abstractmethod
    def compute_loss_scalar(self, model_output, training_sample, criterion_output) -> Tensor:
        pass

    @abstractmethod
    def get_log_vars(self, loss_scalar: Tensor, training_sample: Any, model_output: Any, criterion_output) \
            -> Union[OrderedDict, Dict[str, Tensor]]:
        pass

    def backup_session(self):
        optimizers_and_lr_schedulers_state_dicts = [
            (opt.state_dict(), [lrs.state_dict() for lrs in lr_schedulers] if lr_schedulers is not None else None)
            for opt, lr_schedulers in self.optimizers_and_lr_schedulers
        ]

        save_dict = {
            'model': self._model.state_dict(),
            'optimizers_and_lr_schedulers': optimizers_and_lr_schedulers_state_dicts,
            'eta_estimator': self.eta_estimator.state_dict(),
            'checkpoint_manager': self.checkpoint_manager.state_dict(),
            'metrics_logger': self.metrics_logger.state_dict(),
            'trainer_state_params': self._state_params
        }

        if self.mixed_precision_enabled:
            save_dict['grad_scaler'] = self.grad_scaler.state_dict()

        extra_state_dict = self.get_extra_checkpoint_state_dict()
        save_dict["extra"] = extra_state_dict

        self.checkpoint_manager.save(save_dict, self.elapsed_iterations)
        dist_utils.synchronize()

    def load_model_weights(self, state_dict: Dict[str, Tensor]):
        self._model.load_state_dict(state_dict, strict=True)

    def create_training_data_loader(self,
                                    dataset: Dataset,
                                    sub_iter_batch_size_per_gpu: int,
                                    batch_size: int,
                                    optimizer_step_interval: int,
                                    num_workers: int,
                                    collate_fn: Union[Callable, None]) -> DataLoader:

        num_samples = batch_size * self.total_iterations
        if len(dataset) != num_samples:
            dataset = PaddedDataset(dataset, num_samples)

        if dist_utils.is_distributed():
            shuffle = False  # shuffling happens inside the sampler
            sampler = DistributedSampler(
                dataset, self.num_gpus, dist_utils.get_rank(), shuffle=True,
                elapsed_iters=(self.elapsed_iterations * batch_size)
            )
        else:
            sampler = None
            shuffle = True

        if num_workers < 0:
            num_workers = max(cpu_count() // self.num_gpus, 1)

        return DataLoader(dataset, sub_iter_batch_size_per_gpu, shuffle, sampler, num_workers=num_workers,
                          collate_fn=collate_fn)

    def backward(self, loss_scalar: Tensor, training_sample: Any, model_output: Any) -> None:
        if self.mixed_precision_enabled:
            self.grad_scaler.scale(loss_scalar).backward()
        else:
            loss_scalar.backward()

    def perform_validation_run(self):
        return

    def calculate_optimizer_step_parameters(self, accumulate_gradients: bool, batch_size: int,
                                            max_samples_per_gpu: int) -> Tuple[int, int]:
        if accumulate_gradients:
            # ensure that batch size is larger than the number of available GPUs
            assert batch_size >= self.num_gpus, f"Batch size ({batch_size}) must be >= number of GPUs ({self.num_gpus})"

            if batch_size < (max_samples_per_gpu * self.num_gpus):
                # we have more GPUs than needed
                assert batch_size % self.num_gpus == 0, \
                    f"Batch size ({batch_size}) must be exactly divisible by number of GPUs ({self.num_gpus})"
                optimizer_step_interval = 1
            else:
                assert batch_size % min(batch_size, max_samples_per_gpu) == 0
                optimizer_step_interval = int(batch_size / (min(batch_size, max_samples_per_gpu) * self.num_gpus))

            assert optimizer_step_interval > 0, \
                f"Oops! Something went wrong. Given params: batch_size={batch_size}, " \
                    f"max_samples_per_gpu={max_samples_per_gpu}, num_gpus={self.num_gpus}"

            self.print(f"Optimizer will be run every {optimizer_step_interval} iterations")

        else:
            if batch_size > (self.num_gpus * max_samples_per_gpu):
                raise ValueError(
                    f"A batch size of {batch_size} cannot be achieved because max "
                    f"samples per GPU = {max_samples_per_gpu} and num GPUs = {self.num_gpus} (product of the two is "
                    f"less than batch size)"
                )
            optimizer_step_interval = 1

        sub_iter_batch_size_per_gpu = batch_size // (optimizer_step_interval * self.num_gpus)

        assert 0 < sub_iter_batch_size_per_gpu <= batch_size, \
            f"Oops! Something went wrong. Given params: batch_size={batch_size}, " \
            f"max_samples_per_gpu={max_samples_per_gpu}, num_gpus={self.num_gpus}," \
            f"optimizer_step_interval={optimizer_step_interval}"

        return sub_iter_batch_size_per_gpu, optimizer_step_interval

    def start(self,
              batch_size: int,
              accumulate_gradients: bool,
              clip_gradients: bool,
              max_samples_per_gpu: int,
              display_interval: int,
              summary_interval: int,
              image_summary_interval: int = -1,
              force_end_after_total_iterations_elapsed: bool = True,
              data_loader_cpu_workers: int = -1,
              data_loader_collate_fn: Optional[Callable] = None):

        sub_iter_batch_size_per_gpu, optimizer_step_interval = self.calculate_optimizer_step_parameters(
            accumulate_gradients, batch_size, max_samples_per_gpu
        )

        self.display_interval = display_interval
        self.summary_interval = summary_interval
        self.image_summary_interval = image_summary_interval

        print_list = OrderedDict([
            ("Output directory", self.model_save_dir),
            ("Elapsed iterations", self.elapsed_iterations),
            ("Total iterations", self.total_iterations),
            ("Save interval", self.checkpoint_manager.save_interval),
            ("Batch size", batch_size),
            ("Sub-iteration batch size per GPU", sub_iter_batch_size_per_gpu),
            ("Optimizer step interval", optimizer_step_interval),
            ("Trainable parameters", sum([p.numel() for p in self._model.parameters() if p.requires_grad]))
        ])
        print_list.update(self.get_pretraining_print_list())
        print_list = misc_utils.pretty_parse_dict(print_list)

        self.print(f"Commencing/resuming training with the following settings:\n{print_list}")

        dataset = self.create_training_dataset(batch_size * self.total_iterations)
        data_loader = self.create_training_data_loader(
            dataset=dataset,
            sub_iter_batch_size_per_gpu=sub_iter_batch_size_per_gpu,
            batch_size=batch_size,
            optimizer_step_interval=optimizer_step_interval,
            num_workers=data_loader_cpu_workers,
            collate_fn=data_loader_collate_fn
        )
        data_loader = iter(data_loader)

        subiter_count = 0
        log_vars_cache = defaultdict(lambda: 0.0)

        # Computing logging metrics can be time consuming, so only do it when need (for display, summary etc.)
        logging_metrics_needed = (self.elapsed_iterations + 1) % display_interval == 0 or \
                                 (self.elapsed_iterations + 1) % summary_interval == 0 or \
                                 (self.elapsed_iterations + 1) % image_summary_interval == 0

        dist_utils.synchronize()

        self.eta_estimator.start()
        self.interrupt_detector.start()

        # Instead of using `for training_sample in data_loader`, we use this lengthier code block because it lets
        # us easily time the duration taken by the data_loader to generate the sample.
        while True:
            try:
                if self.elapsed_iterations == self.total_iterations and force_end_after_total_iterations_elapsed:
                    if dist_utils.is_main_process():
                        self.print(
                            f"Total iterations ({self.total_iterations}) have elapsed, but the data loader still "
                            f"contains more samples. These will be ignored because `force_end_after_total_iterations_"
                            f"elapsed` is set to True", level="WARN"
                        )
                    break

                with timer_utils.timed_event(self.print, "Training sample generation duration",
                                             enable=self.log_time_durations):
                    training_sample = next(data_loader)

            except StopIteration as _:
                # data loader has finished generating all samples
                break  

            # -------------------------- FORWARD PASS, LOSS CALCULATION, BACKWARD PASS ---------------------------------
            with torch.autograd.set_detect_anomaly(self.detect_anomaly):
                with autocast(enabled=self.mixed_precision_enabled):
                    # Forward pass
                    model_output, abort_iteration = self._forward_wrapper(training_sample, subiter_count)
                    if abort_iteration:
                        continue

                    # Apply losses/criterion
                    criterion_output = self.apply_criterion(model_output, training_sample)

                    # Compute final, scalar loss value
                    loss = self.compute_loss_scalar(model_output, training_sample, criterion_output)
                    assert loss.numel() == 1, f"Final loss must be a scalar value, but got tensor of shape {list(loss.shape)}"

                # perform backward pass. Note that this should be outside the autocast context but within the
                # detect_anomaly context
                abort_iteration = self._backward_wrapper(
                    training_sample=training_sample, model_output=model_output, criterion_output=criterion_output,
                    loss=loss, optimizer_step_interval=optimizer_step_interval
                )

                if abort_iteration:
                    subiter_count = 0
                    continue
            # ----------------------------------------------------------------------------------------------------------

            # --------------------------- GET LOGGING VARIABLES AND REDUCE ACROSS PROCESSES ----------------------------
            with torch.no_grad():
                if logging_metrics_needed:
                    logging_variables_current_iter = self.get_log_vars(
                        loss, training_sample, model_output, criterion_output
                    )
                    self._update_logging_variables_cache(
                        logging_variables_current_iter, log_vars_cache, optimizer_step_interval
                    )
            # ----------------------------------------------------------------------------------------------------------

            # ---------------------------- END SUB-ITERATION WHEN ACCUMULATING GRADIENTS -------------------------------
            subiter_count += 1
            if subiter_count < optimizer_step_interval:
                self.delete_tensors_in_struct([training_sample, model_output, criterion_output, loss])
                continue
            # ----------------------------------------------------------------------------------------------------------

            # --------------------- STEP OPTIMIZER AND LR SCHEDULERS; RESET SUB-ITERATION COUNTER ----------------------
            subiter_count = 0

            if clip_gradients:
                self._clip_gradients()

            with timer_utils.timed_event(self.print, "Optimizer/LRScheduler step duration",
                                         enable=self.log_time_durations):
                self._step_optimizers_and_lrs()

            self._state_params["elapsed_iterations"] += 1
            self.eta_estimator.tick()
            # ----------------------------------------------------------------------------------------------------------

            # --------------------- ADD METRIC SUMMARIES, BACKUP SESSION, CLEANUP TENSORS---------------------------
            if logging_metrics_needed:
                self._post_step_routine(training_sample, model_output, log_vars_cache)

            log_vars_cache.clear()

            if self.checkpoint_manager.saving_required(self.elapsed_iterations):  
                self.backup_session()

                with torch.no_grad():
                    self.perform_validation_run()

            # determine if metrics need to be logged for the next iteration
            logging_metrics_needed = (self.elapsed_iterations + 1) % display_interval == 0 or \
                                     (self.elapsed_iterations + 1) % summary_interval == 0 or \
                                     (self.elapsed_iterations + 1) % image_summary_interval == 0

            self.delete_tensors_in_struct([training_sample, model_output, criterion_output, loss])
            # ----------------------------------------------------------------------------------------------------------

            # ------------------------------ CHECK IF MAX SESSION RUN-TIME EXCEEDED ------------------------------------
            if (time.time() - self.current_session_start_time) > self.max_runtime:
                self.print("Max runtime for this session has exceeded. Terminating current session.")
                break
            # ----------------------------------------------------------------------------------------------------------

        # Training loop ended. Save final checkpoint if not already saved
        if self.checkpoint_manager.get_last_save_iterations() != self.elapsed_iterations:
            self.backup_session()

        if self.training_complete:
            self.print("Training complete.")

        self.print(
            f"Model(s) saved to: {self.model_save_dir}\n"
            f"Log file(s) saved to: {self.log_dir}\n"
        )

    def _forward_wrapper(self, training_sample: Any, subiter_count: int) -> Tuple[Any, bool]:
        try:
            with timer_utils.timed_event(self.print, label="Forward pass duration",
                                         enable=self.log_time_durations):
                model_output = self.forward(training_sample, subiter_count)

        except RuntimeError as exc:
            if self.is_oom_error(repr(exc)):
                print_str = f"OOM occurred during forward pass in process rank={dist_utils.get_rank()} at " \
                    f"iter={self.elapsed_iterations + 1}, subiter={subiter_count + 1}."

                extra_details = self.get_oom_error_extra_details(training_sample=training_sample)
                if extra_details:
                    print_str += f"\nExtra details: {extra_details}"

                self.print(print_str, level="ERROR")

                if self.ignore_oom_errors and not dist_utils.is_distributed():
                    torch.cuda.empty_cache()
                    return None, True

            raise exc

        if self.interrupt_detector.is_interrupted:
            raise InterruptException()

        return model_output, False

    def _backward_wrapper(self,
                          training_sample,
                          model_output,
                          criterion_output,
                          loss,
                          optimizer_step_interval) -> bool:
        try:
            with timer_utils.timed_event(self.print, label="Backward pass duration",
                                         enable=self.log_time_durations):
                self.backward(loss / float(optimizer_step_interval), training_sample, model_output)

        except RuntimeError as exc:
            if self.ignore_oom_errors and self.is_oom_error(repr(exc)) and not dist_utils.is_distributed():
                self.print("OOM error occurred during backward pass, but will be ignored.", level="ERROR")
                for opt, _ in self.optimizers_and_lr_schedulers:
                    opt.zero_grad()
                # step_idx = 0
                del model_output, criterion_output, loss
                torch.cuda.empty_cache()
                return True
            else:
                raise exc

        return False

    def _step_optimizers_and_lrs(self):
        # Use the same order as in Detectron2:
        # (1) Call step on optimizers
        for opt, _ in self.optimizers_and_lr_schedulers:
            if self.mixed_precision_enabled:
                self.grad_scaler.step(opt)
            else:
                opt.step()

        # (2) Call update on GradScaler if AMP is enabled
        if self.mixed_precision_enabled:
            self.grad_scaler.update()

        # (3) Call step on LR schedulers
        for _, lr_schedulers in self.optimizers_and_lr_schedulers:
            if lr_schedulers is not None:
                for lrs in lr_schedulers:
                    lrs.step()

        # (4) Zero the gradients for the next (sub-)iteration
        for opt, _ in self.optimizers_and_lr_schedulers:
            opt.zero_grad()

    def _update_logging_variables_cache(self,
                                        logging_variables_current_iteration: Dict[str, Tensor],
                                        logging_variables_cache: Dict[str, Tensor],
                                        optimizer_step_interval: int):

        for k, v in logging_variables_current_iteration.items():
            assert torch.is_tensor(v), f"Log variables must be of type torch.Tensor, but got '{type(v)}' for key {k}"
            assert v.ndim == 0, f"Logging variables should be scalar values, but got tensor of " \
                f"shape {list(v.shape)} for key '{k}'"
            logging_variables_cache[k] += (v / float(optimizer_step_interval))

    def _clip_gradients(self):
        gradient_norm = clip_grad_norm_(self._model.parameters(), self.get_gradient_clip_value())

        if gradient_norm != gradient_norm:  # NaN gradients
            self.print("Gradient norm is NaN. Zeroing all gradients", level="WARN")
            for opt, _ in self.optimizers_and_lr_schedulers:
                opt.zero_grad()

    def _post_step_routine(self, training_sample: Any, model_output: Any, log_vars_cache: Dict[str, Tensor]):
        # reduce the logging metrics across all processes. Do this such that the order of keys in `log_vars_cache`
        # is preserved when printing to console.
        log_keys = list(log_vars_cache.keys())
        logging_vars = dist_utils.reduce_dict(dict(log_vars_cache), average=True)
        logging_vars = OrderedDict([
            (key, logging_vars[key]) for key in log_keys
        ])
        
        if not dist_utils.is_main_process():
            return

        # ------------------- log to console ------------------
        if self.elapsed_iterations % self.display_interval == 0:
            print_fn = self.print
        else:
            print_fn = self.console_logger.debug

        eta, avg_time_per_iter = self.eta_estimator.get_eta()
        lr_string = self._get_learning_rate_log_string()

        print_str = self.get_console_logging_string(
            logging_vars=logging_vars, learning_rate=lr_string, estimated_remaining_time=eta,
            avg_iter_duration=avg_time_per_iter
        )

        print_fn(print_str)

        # ------------------ log to tensorboard/wandb -------------------
        metrics_log_dict = dict()

        add_metrics_summary = self.elapsed_iterations % self.summary_interval == 0
        if add_metrics_summary:
            metrics_log_dict.update(logging_vars)

        add_image_summary = self.elapsed_iterations % self.image_summary_interval == 0 and self.image_summary_interval > 0
        if add_image_summary:
            with torch.no_grad():
                image_summaries = self.get_image_summaries(training_sample, model_output)
            metrics_log_dict.update(image_summaries)
            dist_utils.synchronize()

        if len(metrics_log_dict) > 0:
            self.metrics_logger.log(metrics_log_dict, elapsed_iterations=self.elapsed_iterations)
        
        return

    def _get_learning_rate_log_string(self) -> str:
        all_lrs = []

        def reduce_list(x):
            if len(x) == 1:
                return x[0]
            else:
                return x

        for opt, lr_schedulers in self.optimizers_and_lr_schedulers:
            if lr_schedulers is None:
                all_lrs.append(reduce_list([group['lr'] for group in opt.param_groups]))
            else:
                all_lrs.append(reduce_list(lr_schedulers[-1].get_last_lr()))

        def lrs_to_string(x):
            if isinstance(x, list):
                return "[" + ", ".join([lrs_to_string(entry) for entry in x]) + "]"
            elif isinstance(x, float):
                return "{:.2E}".format(x)
            else:
                raise TypeError(f"Invalid type encountered: {type(x)}")

        return lrs_to_string(all_lrs)

    def log_after_iteration(self, key: str, value: Any, allow_overwrite: bool = True):
        """Log an arbitrary variable to console output at the end of the current iteration

        Args:
            key (str): A key for the value
            value (Any): variable to log (must be convertable to string)
        """
        if key in self.logging_buffer and not allow_overwrite:
            raise RuntimeError(f"The variable {key} is already set to be logged.")
        self.logging_buffer[key] = value

    def get_console_logging_string(self, logging_vars: Dict[str, Tensor], learning_rate: str,
                                   estimated_remaining_time: str, avg_iter_duration: float) -> str:
        logging_vars_str = " - ".join([
            f"{Fore.BLUE}{k}: {Fore.GREEN}{v.item():.3f}{Fore.RESET}"
            for k, v in logging_vars.items()
        ])

        for k, v in self.logging_buffer.items():
            logging_vars_str += f" - {Fore.BLUE}{k}: {Fore.GREEN}{str(v)}{Fore.RESET}"
        self.logging_buffer.clear()

        if learning_rate:
            return f"{Fore.CYAN}It: {self.elapsed_iterations:05}{Fore.RESET} - {logging_vars_str} - lr: {learning_rate} -" \
                f" {Fore.YELLOW}ETA: {estimated_remaining_time} - sec/it: {avg_iter_duration:.3f}{Fore.RESET}\n"
        else:
            return f"{Fore.CYAN}It: {self.elapsed_iterations:05}{Fore.RESET} - {logging_vars_str} -" \
                f" {Fore.YELLOW}ETA: {estimated_remaining_time} - sec/it: {avg_iter_duration:.3f}{Fore.RESET}\n"

    def get_extra_checkpoint_state_dict(self) -> Dict[str, Any]:
        return dict()

    def load_extra_checkpoint_state_dict(self):
        pass

    def get_gradient_clip_value(self) -> float:
        if self.elapsed_iterations < 100:
            return 20.0
        elif 50 <= self.elapsed_iterations < 1000:
            return 10.0
        else:
            return 2.0

    def get_oom_error_extra_details(self, training_sample: Any) -> str:
        return ""

    def get_pretraining_print_list(self) -> OrderedDict:
        return OrderedDict([])

    def get_image_summaries(self, training_sample: Any, model_output: Any) -> Dict[str, np.ndarray]:
        """
        The child class implementation of this method should return a dict mapping str to images as NumPy arrays
        :param training_sample:
        :param model_output:
        :return: Dict with keys of type str without white space and special characters. Values should be images
        as NumPay arrays of shape [B, C, H, W]. Dtype should either by uint8 with values in range [0, 255] or float32
        with values in range [0, 1]
        """
        return {}

    def struct_to_local_device(self, x: Any) -> Any:
        if torch.is_tensor(x):
            return x.to(device=self.local_device)
        elif isinstance(x, (list, tuple)):
            return type(x)([self.struct_to_local_device(elem) for elem in x])
        elif isinstance(x, dict):
            return {k: self.struct_to_local_device(v) for k, v in x.items()}
        elif hasattr(x, "to"):
            return x.to(device=self.local_device)
        else:
            return x

    def delete_tensors_in_struct(self, x: Any):
        if torch.is_tensor(x):
            del x
        elif isinstance(x, (list, tuple)):
            for elem in x:
                self.delete_tensors_in_struct(elem)
        elif isinstance(x, dict):
            for v in x.values():
                self.delete_tensors_in_struct(v)
        else:
            return x

    @staticmethod
    def is_oom_error(msg: str) -> bool:
        return msg.startswith("RuntimeError('CUDA out of memory.")

    @staticmethod
    def get_num_available_cpu_cores() -> int:
        # When running under SLURM, we need to check the `SLURM_CPUS_PER_TASK` environment variable to get the
        # correct number of available CPU cores because multiprocessing.cpu_count() just returns the total number of
        # CPU cores on the machine regardless of how many SLURM has allocated for the given job.
        total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        return min(8, max(1, total_cores // local_world_size))


@contextlib.contextmanager
def main_function_wrapper(master_port: Optional[str] = None):
    # When launched with torchrun, the env variables 'WORLD_SIZE' and 'LOCAL_RANK' are set.
    is_multigpu = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_multigpu:
        if master_port is not None:
            os.environ['MASTER_PORT'] = str(master_port)

        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # For multi-node training, it is important that 'rank' be the global rank (i.e. unique process ID across all
        # nodes)
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size)

        try:
            # For assigning tensors/modules, use the local rank within the node
            with torch.cuda.device(local_rank):
                yield None
        except InterruptException as _:
            print("Training session was interrupted")

        dist.destroy_process_group()
    else:
        yield None
