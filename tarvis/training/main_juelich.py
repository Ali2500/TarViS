from argparse import ArgumentParser
from datetime import timedelta
from glob import glob
from typing import List, Union, Any

from tarvis.utils import distributed as dist_utils
from tarvis.training.model_trainer.api import main_function_wrapper

from tarvis.config import cfg
from tarvis.training.tarvis_train_model import TarvisTrainModel
from tarvis.training.main import Trainer, parse_model_dir, parse_config_path, seed_rngs, print_

import cv2
import logging
import os
import os.path as osp
import torch
import torch.distributed as dist
import subprocess
import yaml


def get_job_time_limit():
    if dist_utils.is_main_process():
        job_id = os.getenv("SLURM_JOB_ID")
        out = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True, env=os.environ)
        tokens = out.stdout.split(" ")
        target = [t for t in tokens if "TimeLimit" in t][0]
        target = target.replace("TimeLimit=", "")
        if "-" in target:
            days, rest = target.split("-")
            days = int(days)
        else:
            days = 0
            rest = target

        hrs, mins, secs = [int(x) for x in rest.split(":")]

        duration = timedelta(days=days, hours=hrs, minutes=mins, seconds=secs)
    else:
        duration = timedelta(days=0, hours=0, minutes=0, seconds=0)

    # reduce all
    duration = torch.tensor([duration.days, duration.seconds], dtype=torch.float32, device=f"cuda:{dist_utils.get_local_rank()}")
    dist.all_reduce(duration, op=dist.ReduceOp.SUM)

    days, seconds = duration.int().tolist()
    duration = timedelta(days=days, seconds=seconds)
    return duration - timedelta(minutes=5)


def run(config_path: Union[str, None],
        model_dir: str,
        restore_session: Union[str, None],
        finetune_from: Union[str, None],
        max_checkpoints_to_keep: int,
        log_level: str,
        subprocess_log_level: str,
        detect_anomaly: bool,
        max_samples_per_gpu: int,
        display_interval: int,
        summary_interval: int,
        amp: bool,
        num_cpu_workers: int,
        args_for_cfg: Union[Any, None],
        max_runtime_hours: int = 12
        ):

    try:
        max_runtime = get_job_time_limit()

    except Exception as exc:
        if dist_utils.is_main_process():
            print(f"Failed to fetch SLURM job runtime. Error:\n{str(exc)}")
        dist_utils.synchronize()
        assert max_runtime_hours >= 1
        max_runtime = timedelta(hours=max_runtime_hours - 1, minutes=55)

    if restore_session:
        model_dir = osp.dirname(restore_session)
        config_path = osp.join(model_dir, "config.yaml")
        assert osp.exists(config_path), f"Config file not found at expected path: {config_path}"
        cfg.merge_from_file(config_path)

        model = TarvisTrainModel().cuda()

        trainer = Trainer.restore_from_checkpoint(
            model=model,
            checkpoint_path=restore_session,
            max_runtime=max_runtime
        )
    else:
        assert config_path
        assert model_dir

        # If fine-tuning, first restore config from the pre-training checkpoint directory
        if finetune_from:
            if osp.isfile(finetune_from):
                raise ValueError(
                    "The '--finetune_from' arg should point to the training session directory containing the pretrained"
                    " model checkpoints and config.yaml file. The given path, however, points to a file."
                )

            pretrain_config_path = osp.join(finetune_from, "config.yaml")
            assert osp.exists(pretrain_config_path), \
                f"Config file not found in pre-training directory at expected path: {pretrain_config_path}"

            cfg.merge_from_file(pretrain_config_path)
            restore_weights_path = sorted(glob(osp.join(finetune_from, "*.pth")))[-1]
            print_(f"Loading pre-trained config from: {pretrain_config_path}\n"
                   f"Loading model weights from: {restore_weights_path}")
        else:
            restore_weights_path = None

        config_path = parse_config_path(config_path)
        cfg.merge_from_file(config_path)

        if args_for_cfg is not None and finetune_from is None:
            cfg.update_from_args(args_for_cfg, verbose=dist_utils.is_main_process())

        world_size = dist_utils.get_world_size()
        if cfg.TRAINING.BATCH_SIZE < world_size:
            if dist_utils.is_main_process():
                print(f"Batch size ({cfg.TRAINING.BATCH_SIZE}) is less than world size."
                      f"Updating batch size to {world_size}")
            cfg.TRAINING.update_param("BATCH_SIZE", world_size)

        model = TarvisTrainModel().cuda()
        model_save_dir = parse_model_dir(model_dir)

        trainer = Trainer.new(
            model=model,
            model_save_dir=model_save_dir,
            restore_model_weights=restore_weights_path,
            total_iterations=cfg.TRAINING.NUM_ITERATIONS,
            save_interval=cfg.TRAINING.CHECKPOINT_SAVE_INTERVAL,
            use_mixed_precision=amp,
            find_model_unused_parameters=True,
            convert_sync_batchnorm=False,
            start_saving_checkpoints_after=0,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            wandb_logging=False,
            max_runtime=max_runtime
        )

        # backup config to model directory
        if dist_utils.is_main_process():
            with open(osp.join(model_dir, 'config.yaml'), 'w') as writefile:
                yaml.dump(cfg.d(), writefile)

    if dist_utils.is_main_process():
        log_level = logging.getLevelName(log_level)
    else:
        log_level = logging.getLevelName(subprocess_log_level)

    trainer.console_logger.setLevel(log_level)
    trainer.detect_anomaly = detect_anomaly
    trainer.print(f"Maximum job runtime: {max_runtime}")

    trainer.start(
        batch_size=cfg.TRAINING.BATCH_SIZE,
        accumulate_gradients=cfg.TRAINING.ACCUMULATE_GRADIENTS,
        clip_gradients=False,
        max_samples_per_gpu=max_samples_per_gpu,
        data_loader_cpu_workers=num_cpu_workers,
        display_interval=display_interval,
        summary_interval=summary_interval
    )

    # create empty dummy file to  training completion to future SLURM jobs
    if trainer.training_complete and dist_utils.is_main_process():
        with open(osp.join(model_dir, "training_complete.flag"), 'w') as _:
            pass


def main(args):
    if args.cv2_num_threads:
        cv2.setNumThreads(args.cv2_num_threads)

    job_id = os.environ["SLURM_ARRAY_TASK_ID"]
    job_index = int(job_id) - int(os.environ["SLURM_ARRAY_TASK_MIN"])
    print_(f"SLURM Job index: {job_index}")

    def get_checkpoint(dirpath):
        checkpoint_paths = sorted(glob(osp.join(dirpath, "*.pth")))
        assert len(checkpoint_paths) > 0
        return checkpoint_paths[-1]

    model_dir_pretrain = parse_model_dir(args.model_dir_pretrain)
    pretrain_resumable = len(glob(osp.join(model_dir_pretrain, "*.pth"))) > 0
    pretrain_complete = osp.exists(osp.join(model_dir_pretrain, "training_complete.flag"))

    finetune_disable = args.cfg_finetune == "none"
    model_dir_finetune = parse_model_dir(args.model_dir_finetune)
    finetune_resumable = len(glob(osp.join(model_dir_finetune, "*.pth"))) > 0
    finetune_complete = osp.exists(osp.join(model_dir_finetune, "training_complete.flag"))

    if not pretrain_resumable:  # start pretraining
        # assert job_index == 0
        seed_rngs(args.random_seed)
        # wandb_session_name = osp.split(args.model_dir_pretrain)[-1]

        run(
            model_dir=model_dir_pretrain,
            config_path=args.cfg_pretrain,
            restore_session=None,
            finetune_from=None,
            max_checkpoints_to_keep=args.max_checkpoints_to_keep_pretrain,
            log_level=args.log_level,
            subprocess_log_level=args.subprocess_log_level,
            detect_anomaly=args.detect_anomaly,
            max_samples_per_gpu=args.max_samples_per_gpu_pretrain,
            display_interval=args.display_interval,
            summary_interval=args.summary_interval,
            amp=args.amp,
            num_cpu_workers=args.num_cpu_workers,
            args_for_cfg=args,
            max_runtime_hours=args.max_runtime_hrs
        )

    elif pretrain_resumable and not pretrain_complete:  # resume pretraining
        restore_session_path = get_checkpoint(model_dir_pretrain)
        seed_rngs(args.random_seed + job_index)

        run(
            model_dir=model_dir_pretrain,
            config_path=args.cfg_pretrain,
            restore_session=restore_session_path,
            finetune_from=None,
            max_checkpoints_to_keep=args.max_checkpoints_to_keep_pretrain,
            log_level=args.log_level,
            subprocess_log_level=args.subprocess_log_level,
            detect_anomaly=args.detect_anomaly,
            max_samples_per_gpu=args.max_samples_per_gpu_pretrain,
            display_interval=args.display_interval,
            summary_interval=args.summary_interval,
            amp=args.amp,
            num_cpu_workers=args.num_cpu_workers,
            args_for_cfg=None,
            max_runtime_hours=args.max_runtime_hrs
        )

    elif finetune_disable:
        return

    elif pretrain_complete and not finetune_resumable:  # start finetuning
        assert pretrain_complete
        finetune_from_path = model_dir_pretrain
        seed_rngs(args.random_seed)

        run(
            model_dir=model_dir_finetune,
            config_path=args.cfg_finetune,
            restore_session=None,
            finetune_from=finetune_from_path,
            max_checkpoints_to_keep=args.max_checkpoints_to_keep_finetune,
            log_level=args.log_level,
            subprocess_log_level=args.subprocess_log_level,
            detect_anomaly=args.detect_anomaly,
            max_samples_per_gpu=args.max_samples_per_gpu_finetune,
            display_interval=args.display_interval,
            summary_interval=args.summary_interval,
            amp=args.amp,
            num_cpu_workers=args.num_cpu_workers,
            args_for_cfg=None,
            max_runtime_hours=args.max_runtime_hrs
        )

    elif finetune_resumable and not finetune_complete:  # resume finetuning
        assert pretrain_complete
        restore_session_path = get_checkpoint(model_dir_finetune)
        seed_rngs(args.random_seed)

        run(
            model_dir=model_dir_finetune,
            config_path=args.cfg_finetune,
            restore_session=restore_session_path,
            finetune_from=None,
            max_checkpoints_to_keep=args.max_checkpoints_to_keep_finetune,
            log_level=args.log_level,
            subprocess_log_level=args.subprocess_log_level,
            detect_anomaly=args.detect_anomaly,
            max_samples_per_gpu=args.max_samples_per_gpu_finetune,
            display_interval=args.display_interval,
            summary_interval=args.summary_interval,
            amp=args.amp,
            num_cpu_workers=args.num_cpu_workers,
            args_for_cfg=None,
            max_runtime_hours=args.max_runtime_hrs
        )

    else:
        assert finetune_complete, f"Something isn't right..."
        if dist_utils.is_main_process():
            slurm_array_job_id = os.getenv('SLURM_ARRAY_JOB_ID')
            print(f"Pre-training + fine-tuning is complete. Terminating SLURM job array ID: {slurm_array_job_id}")
            subprocess.run(["scancel", slurm_array_job_id], capture_output=True, text=True, env=os.environ)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_dir_pretrain", required=True)
    parser.add_argument("--model_dir_finetune", required=True)
    parser.add_argument("--cfg_pretrain", required=False)
    parser.add_argument("--cfg_finetune", required=True)

    parser.add_argument("--num_cpu_workers", type=int, default=-1)
    parser.add_argument("--max_samples_per_gpu_pretrain", type=int, default=2)
    parser.add_argument("--max_samples_per_gpu_finetune", type=int, default=1)

    parser.add_argument("--detect_anomaly", action='store_true')
    parser.add_argument("--amp", action='store_true')
    parser.add_argument("--cv2_num_threads", type=int, required=False)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--master_port", required=False)
    parser.add_argument("--max_runtime_hrs", type=int, default=24)

    parser.add_argument("--log_level",            required=False, default="INFO")
    parser.add_argument("--subprocess_log_level", required=False, default="WARN")

    parser.add_argument("--display_interval", type=int, default=10)
    parser.add_argument("--summary_interval", type=int, default=20)

    parser.add_argument("--max_checkpoints_to_keep_pretrain", type=int, default=2)
    parser.add_argument("--max_checkpoints_to_keep_finetune", type=int, default=5)

    cfg_args_group = parser.add_argument_group("Config Arguments")
    cfg.add_args_to_parser(cfg_args_group, suppress_help=True)

    parsed_args = parser.parse_args()
    with main_function_wrapper(master_port=parsed_args.master_port):
        main(parsed_args)
