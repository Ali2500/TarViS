from argparse import ArgumentParser
from collections import OrderedDict
from glob import glob
from einops import rearrange, repeat
from datetime import timedelta
from typing import Dict, List, Any, Set
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tarvis.training.model_trainer import ModelTrainer as TrainerBase, main_function_wrapper

from tarvis.utils import distributed as dist_utils
from tarvis.training import lr_schedulers

from tarvis.config import cfg
from tarvis.utils.paths import Paths
from tarvis.data.collate import collate_fn_train
from tarvis.training.build_dataset import build_concat_dataset
from tarvis.training.batch_sampler import TaskTypeAwareBatchSampler
from tarvis.training.tarvis_train_model import TarvisTrainModel
from tarvis.training.distributed_sampler import DistributedSampler

import cv2
import copy
import logging
import os
import os.path as osp
import random
import numpy as np
import imgaug
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import yaml


warnings.filterwarnings(action='ignore', category=UserWarning, module='torch.optim.lr_scheduler')


class Trainer(TrainerBase):
    def setup(self, *args, **kwargs):
        self.current_iter_task = ""
        # self.current_iter = -1
        self.task_chunking = cfg.TRAINING.SAMPLE_CHUNKING_FACTOR

        self.pretrain_task_weights = {
            k.lower(): v for k, v in cfg.TRAINING.PRETRAIN_TASK_WEIGHTS.as_dict().items()
        }

        self.rng_task_sampling = torch.Generator()
        self.rng_task_sampling.manual_seed(764501)

    def create_optimizers(self, model):
        weight_decay_norm = 0.0
        weight_decay_embed = 0.0
        backbone_multiplier = 0.1

        defaults = {
            "lr": cfg.TRAINING.BASE_LR,
            "weight_decay": 0.05
        }

        norm_module_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)

                if "backbone.backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * backbone_multiplier

                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm

                if isinstance(module, nn.Embedding) or "embedding_container" in module_name:
                    hyperparams["weight_decay"] = weight_decay_embed

                if "w_different" == module_param_name or "w_same" == module_param_name:
                    # print(module_param_name)
                    hyperparams["weight_decay"] = weight_decay_embed

                if module_param_name == "scaling_factors" and value.numel() == 8:
                    hyperparams["weight_decay"] = weight_decay_embed

                if module_param_name in ("scale_encodings_grid", "scale_encodings_block"):
                    hyperparams["weight_decay"] = weight_decay_embed

                if module_param_name in ("mask_object_query_embed", "point_object_query_embed", "bg_query_embed") \
                        and module_name == "vos_query_extractor":
                    hyperparams["weight_decay"] = weight_decay_embed

                if module_param_name in ("level_embed", "level_embed_3d") and list(value.shape) == [3, 256]:
                    hyperparams["weight_decay"] = weight_decay_embed

                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = 0.01  # cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer

        optimizer = maybe_add_full_model_gradient_clipping(optim.AdamW)(
            params, cfg.TRAINING.BASE_LR  # , eps=1e-6
        )

        decay_steps = cfg.TRAINING.LR_DECAY_STEPS
        decay_rates = cfg.TRAINING.LR_DECAY_RATES
        warmup_iters = cfg.TRAINING.LR_WARMUP_ITERATIONS

        if decay_steps:
            lr_scheduler = lr_schedulers.StepLR(optimizer, decay_steps, decay_rates, warmup_iters)
            self.register_optimizer(optimizer, [lr_scheduler])
        else:
            self.register_optimizer(optimizer)

    def load_model_weights(self, state_dict: Dict[str, Tensor]):
        if state_dict["embedding_container._semantic_queries"].shape != self._model.embedding_container._semantic_queries.shape:
            self.load_cityscapes_semantic_queries(state_dict)
            del state_dict["embedding_container._semantic_queries"]

        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        if "embedding_container._semantic_queries" in missing:
            missing.remove("embedding_container._semantic_queries")

        if missing or unexpected:
            raise ValueError(f"Missing/unexpected keys found in restoration checkpoint.\n"
                             f"Missing: {missing}\n"
                             f"Unexpected: {unexpected}")

    @torch.no_grad()
    def load_cityscapes_semantic_queries(self, state_dict: Dict[str, Tensor]):
        cityscapes_vps_kitti_step_to_cityscapes_category_mapping = {
            0: 7,
            1: 8,
            2: 11,
            3: 12,
            4: 13,
            5: 17,
            6: 19,
            7: 20,
            8: 21,
            9: 22,
            10: 23,
            11: 24,
            12: 25,
            13: 26,
            14: 27,
            15: 28,
            16: 31,
            17: 32,
            18: 33
        }

        self._model.embedding_container._semantic_queries[0] = state_dict["embedding_container._semantic_queries"][0]

        # TODO: THIS IS VERY FRAGILE. If the pretrain dataset list changes this will load bogus queries!
        pretrain_dataset_list = ["COCO", "ADE20K", "CITYSCAPES", "MAPILLARY"]
        cityscapes_offset = 1  # first entry in checkpoint is the query embedding
        self.print(f"Restoring Cityscapes semantic queries for Cityscapes VPS and KITTI-STEP. Assuming that pretrain "
                   f"dataset list is: {pretrain_dataset_list}! Make sure that this assumption is valid!")

        for ds in sorted(pretrain_dataset_list):
            if ds == "CITYSCAPES":
                break
            num_classes = cfg.DATASETS.get(ds).NUM_CLASSES
            cityscapes_offset += num_classes

        cityscapes_queries = state_dict["embedding_container._semantic_queries"][cityscapes_offset:cityscapes_offset+34]
        perm = torch.tensor([cityscapes_vps_kitti_step_to_cityscapes_category_mapping[i] for i in range(19)], dtype=torch.int64)
        cityscapes_queries = cityscapes_queries[perm]  # [19, C]

        dataset_list = cfg.TRAINING.DATASET_LIST
        if "KITTI_STEP" in dataset_list:
            self._model.embedding_container.assign_semantic_queries("KITTI_STEP", cityscapes_queries)

        if "CITYSCAPES_VPS" in dataset_list:
            self._model.embedding_container.assign_semantic_queries("CITYSCAPES_VPS", cityscapes_queries)

    def create_training_dataset(self, total_samples: int):
        return None  # make the dataset together with the DataLoader in `create_training_data_loader'

    def create_training_data_loader(
        self, dataset: Dataset, sub_iter_batch_size_per_gpu: int, batch_size: int,
        optimizer_step_interval: int, num_workers: int, collate_fn
    ) -> DataLoader:

        dataset = build_concat_dataset(
            cfg.TRAINING.DATASET_LIST, cfg.TRAINING.DATASET_WEIGHTS, self.total_iterations * batch_size
        )
        sampler = DistributedSampler(dataset, self.num_gpus, self.global_rank, shuffle=True)

        batch_sampler = TaskTypeAwareBatchSampler(
            sampler=sampler, 
            dataset=dataset,
            total_iterations=self.total_iterations,
            batch_size=sub_iter_batch_size_per_gpu * optimizer_step_interval,
            post_shuffle=True,
            sub_batch_size=sub_iter_batch_size_per_gpu,
            chunking_factor=self.task_chunking,
            elapsed_batches=self.elapsed_iterations
        )
        num_workers = self.get_num_available_cpu_cores() if num_workers < 0 else num_workers

        return DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn_train,
            worker_init_fn=dataloder_worker_init
        )

    def forward(self, training_sample: Any, subiter: int) -> Any:
        training_sample = self.struct_to_local_device(training_sample)

        # During pre-training all datasets can be trained with any task. Decide which task to train for here.
        if self.elapsed_iterations % self.task_chunking == 0 and subiter == 0:
            rand = torch.rand(1, generator=self.rng_task_sampling)
            tasks = list(self.pretrain_task_weights.keys())
            weights = torch.tensor([self.pretrain_task_weights[t] for t in tasks])
            weights = weights.cumsum(0)

            for task, w in zip(tasks, weights):
                if rand < w:
                    self.current_iter_task = task
                    break

        self.log_after_iteration("Dataset", training_sample["dataset"][0])
        return self.model(training_sample, target_task=self.current_iter_task)

    def apply_criterion(self, model_output: Any, training_sample: Any) -> Any:
        return None  # losses were applied inside model forward pass

    def compute_loss_scalar(self, model_output, training_sample, criterion_output):
        losses = model_output["losses"]  # one dict per loss type        
        total_loss = sum([loss_type["loss_total"].sum() for loss_type in losses.values()])
        return total_loss

    def get_log_vars(
            self, loss_scalar: Tensor, training_sample: Any, model_output: Any, criterion_output
    ) -> Dict[str, Tensor]:

        log_vars_dict = OrderedDict()
        losses = model_output["losses"]

        num_expected_outputs = self._model.decoder.num_layers + 1

        for losses_per_type in losses.values():
            assert all([x.numel() == num_expected_outputs for x in losses_per_type.values()])

        log_vars_dict["TotalL"] = sum([loss_type["loss_total"].sum() for loss_type in losses.values()])

        vos_losses = losses.get("vos", None)
        if vos_losses:
            log_vars_dict["VSegDiceL"] = vos_losses["loss_dice"][-1]
            log_vars_dict["VSegCeL"] = vos_losses["loss_ce"][-1]
            log_vars_dict["Aux_VSegDiceL"] = vos_losses["loss_dice"][:-1].mean()
            log_vars_dict["Aux_VSegCeL"] = vos_losses["loss_ce"][:-1].mean()

        instance_seg_losses = losses.get("instance_seg", None)
        if instance_seg_losses:
            log_vars_dict["ISegDiceL"] = instance_seg_losses["loss_mask_dice"][-1]
            log_vars_dict["ISegCeL"] = instance_seg_losses["loss_mask_ce"][-1]
            log_vars_dict["ISegClsL"] = instance_seg_losses["loss_cls"][-1]
            log_vars_dict["Aux_ISegDiceL"] = instance_seg_losses["loss_mask_dice"][:-1].mean()
            log_vars_dict["Aux_ISegCeL"] = instance_seg_losses["loss_mask_ce"][:-1].mean()
            log_vars_dict["Aux_ISegClsL"] = instance_seg_losses["loss_cls"][:-1].mean()

            semantic_seg_losses = losses["semantic_seg"]
            if "loss_dice" in semantic_seg_losses:
                log_vars_dict["SSegDiceL"] = semantic_seg_losses["loss_dice"][-1]
            log_vars_dict["SSegCeL"] = semantic_seg_losses["loss_ce"][-1]

            if "loss_dice" in semantic_seg_losses:
                log_vars_dict["Aux_SSegDice"] = semantic_seg_losses["loss_dice"][:-1].mean()
            log_vars_dict["Aux_SSegCe"] = semantic_seg_losses["loss_ce"][:-1].mean()

            if "loss_reg" in semantic_seg_losses:
                log_vars_dict["SSegRegL"] = semantic_seg_losses["loss_reg"].mean()

        panoptic_seg_losses = losses.get("panoptic_seg", None)
        if panoptic_seg_losses:
            log_vars_dict["PSegIDiceL"] = panoptic_seg_losses["loss_instance_mask_dice"][-1]
            log_vars_dict["PSegICeL"] = panoptic_seg_losses["loss_instance_mask_ce"][-1]
            log_vars_dict["PSegIClsL"] = panoptic_seg_losses["loss_instance_cls"][-1]
            log_vars_dict["PSegSSegCeL"] = panoptic_seg_losses["loss_semantic_ce"][-1]
            if "loss_semantic_dice" in panoptic_seg_losses:
                log_vars_dict["PSegSSegDiceL"] = panoptic_seg_losses["loss_semantic_dice"][-1]

            if "loss_semantic_reg" in panoptic_seg_losses:
                log_vars_dict["PSegSSegRegL"] = panoptic_seg_losses["loss_semantic_reg"].mean()

            log_vars_dict["Aux_PSegIDiceL"] = panoptic_seg_losses["loss_instance_mask_dice"][:-1].mean()
            log_vars_dict["Aux_PSegICeL"] = panoptic_seg_losses["loss_instance_mask_ce"][:-1].mean()
            log_vars_dict["Aux_PSegIClsL"] = panoptic_seg_losses["loss_instance_cls"][:-1].mean()
            log_vars_dict["Aux_PSegSSegCeL"] = panoptic_seg_losses["loss_semantic_ce"][:-1].mean()
            if "loss_semantic_dice" in panoptic_seg_losses:
                log_vars_dict["Aux_PSegSSegDiceL"] = panoptic_seg_losses["loss_semantic_dice"][:-1].mean()

        return log_vars_dict

    def _get_learning_rate_log_string(self) -> str:
        all_lrs = []

        for opt, lr_schedulers in self.optimizers_and_lr_schedulers:
            if lr_schedulers is None:
                all_lrs.extend([group['lr'] for group in opt.param_groups])
            else:
                all_lrs.extend(lr_schedulers[-1].get_last_lr())

        all_lrs = sorted(list(set(all_lrs)), reverse=True)

        all_lrs = ["{:.2E}".format(lr) for lr in all_lrs]
        return "[" + ", ".join(all_lrs) + "]"

    def get_pretraining_print_list(self) -> OrderedDict:
        plist = OrderedDict()
        plist["Backbone"] = cfg.MODEL.BACKBONE
        plist["VOS Queries Per Object"] = cfg.MODEL.NUM_VOS_QUERIES_PER_OBJECT
        plist["Cross-attention Type"] = cfg.MODEL.CROSS_ATTENTION_TYPE
        plist["Base LR"] = cfg.TRAINING.BASE_LR
        plist["LR Decay Steps"] = cfg.TRAINING.LR_DECAY_STEPS
        plist["LR Warmup Iterations"] = cfg.TRAINING.LR_WARMUP_ITERATIONS
        plist["No. Background Queries"] = cfg.MODEL.NUM_BACKGROUND_QUERIES
        plist["Datasets"] = ", ".join(cfg.TRAINING.DATASET_LIST)
        plist["Dataset Weights"] = cfg.TRAINING.DATASET_WEIGHTS
        plist["Clip Length"] = cfg.TRAINING.CLIP_LENGTH
        plist["Panoptic Semseg Loss Points"] = cfg.TRAINING.LOSSES.PANOPTIC_SEMANTIC_NUM_POINTS
        plist["Panoptic Instance Loss Points"] = cfg.TRAINING.LOSSES.PANOPTIC_INSTANCE_NUM_POINTS
        plist["PET likelihood"] = cfg.TRAINING.POINT_VOS_TASK_SAMPLE_LIKELIHOOD
        plist["Task Chunking"] = cfg.TRAINING.SAMPLE_CHUNKING_FACTOR

        if cfg.TRAINING.PRETRAINING:
            plist["Task Weights"] = ", ".join([f"{k.lower()}: {v}" for k, v in cfg.TRAINING.PRETRAIN_TASK_WEIGHTS.as_dict().items()])

        return plist


def dataloder_worker_init(worker_id: int) -> None:
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    seed = 220294 + dist_utils.get_rank() + task_id

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    imgaug.seed(seed)


def print_(*args, **kwargs):
    if dist_utils.is_main_process():
        print(*args, **kwargs)


def parse_model_dir(model_dir: str):
    if osp.isabs(model_dir):
        return model_dir
    else:
        return osp.join(Paths.saved_models_dir(), model_dir)


def parse_config_path(cfg_path: str):
    if osp.isabs(cfg_path):
        return cfg_path
    else:
        return osp.join(Paths.configs_dir(), cfg_path)


def seed_rngs(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    imgaug.seed(seed)


def main(args):
    seed_rngs(args.random_seed)

    if args.cv2_num_threads:
        cv2.setNumThreads(args.cv2_num_threads)

    if args.restore_session:
        model_save_dir = osp.dirname(args.restore_session)
        config_path = osp.join(model_save_dir, "config.yaml")
        assert osp.exists(config_path), f"Config file not found at expected path: {config_path}"
        cfg.merge_from_file(config_path)

        model = TarvisTrainModel().cuda()

        trainer = Trainer.restore_from_checkpoint(
            model=model,
            checkpoint_path=args.restore_session
        )
    else:
        assert args.cfg, f"'--cfg' arg is required"
        assert args.model_dir, f"'--model_dir' arg is required"

        # If fine-tuning, first restore config from the pre-training checkpoint directory
        if args.finetune_from:
            assert not args.restore_weights
            if osp.isfile(args.finetune_from):
                raise ValueError(
                    "The '--finetune_from' arg should point to the training session directory containing the pretrained"
                    " model checkpoints and config.yaml file. The given path, however, points to a file."
                )

            pretrain_config_path = osp.join(args.finetune_from, "config.yaml")
            assert osp.exists(pretrain_config_path), \
                f"Config file not found in pre-training directory at expected path: {pretrain_config_path}"

            cfg.merge_from_file(pretrain_config_path)
            restore_weights_path = sorted(glob(osp.join(args.finetune_from, "*.pth")))[-1]
            print_(f"Loading pre-trained config from: {pretrain_config_path}\n"
                   f"Loading model weights from: {restore_weights_path}")
        elif args.restore_weights:
            restore_weights_path = args.restore_weights
        else:
            restore_weights_path = None

        config_path = parse_config_path(args.cfg)
        cfg.merge_from_file(config_path)
        cfg.update_from_args(args, verbose=dist_utils.is_main_process())

        model = TarvisTrainModel().cuda()
        model_save_dir = parse_model_dir(args.model_dir)

        enable_logging = args.wandb_session is not None
        if args.wandb_session is None:
            args.wandb_session = osp.split(args.model_dir)[-1]

        max_runtime = timedelta(days=int(1e6))

        trainer = Trainer.new(
            model=model,
            model_save_dir=model_save_dir,
            total_iterations=cfg.TRAINING.NUM_ITERATIONS,
            save_interval=cfg.TRAINING.CHECKPOINT_SAVE_INTERVAL,
            restore_model_weights=restore_weights_path,
            use_mixed_precision=args.amp,
            find_model_unused_parameters=True,
            convert_sync_batchnorm=False,
            start_saving_checkpoints_after=args.save_ckpts_after,
            max_checkpoints_to_keep=args.max_ckpts_to_keep,
            wandb_logging=enable_logging,
            wandb_project="UVS",
            wandb_run=args.wandb_session,
            wandb_config_dict=cfg.as_dict(),
            max_runtime=max_runtime
        )

    if dist_utils.is_main_process():
        log_level = logging.getLevelName(args.log_level)
    else:
        log_level = logging.getLevelName(args.subprocess_log_level)

    trainer.console_logger.setLevel(log_level)
    trainer.detect_anomaly = args.detect_anomaly
    trainer.ignore_oom_errors = args.ignore_oom_errors

    # backup config to model directory
    if dist_utils.is_main_process():
        with open(osp.join(model_save_dir, 'config.yaml'), 'w') as writefile:
            yaml.dump(cfg.d(), writefile)

    trainer.start(
        batch_size=cfg.TRAINING.BATCH_SIZE,
        accumulate_gradients=cfg.TRAINING.ACCUMULATE_GRADIENTS,
        clip_gradients=False,
        max_samples_per_gpu=args.max_samples_per_gpu,
        data_loader_cpu_workers=args.num_cpu_workers,
        display_interval=args.display_interval,
        summary_interval=args.summary_interval,
    )


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        "--model_dir", required=False, 
        help="Directory where model checkpoints and logs will be stored. "
        "Relative paths are appended to $TARVIS_WORKSPACE_DIR/checkpoints"
    )
    parser.add_argument(
        "--cfg", required=False,
        help="Config file to use. Relative paths are appended to ./configs"
    )

    parser.add_argument(
        "--restore_weights", required=False,
        help="Restore model weights from a pre-existing checkpoint before starting training."
    )
    parser.add_argument(
        "--restore_session", required=False,
        help="Resume training from the given checkpoint file."
    )
    parser.add_argument(
        "--finetune_from", required=False,
        help="Use this to start the finetuning training after pre-training on image datasets. "
        "In addition to loading model weights, this also loads the config from the pre-trained "
        "checkpoint directory before overwriting it with the given '--cfg'. This option should point "
        "to the directory containing the pretrained checkpoint, and not the checkpoint file itself."
    )

    parser.add_argument(
        "--batch_size", type=int, required=False, 
        help="Batch size for training. This is the final batch size summed across all GPUs and nodes. "
        "The per-GPU batch size is automatically set based on the number of available GPUs."
    )
    parser.add_argument(
        "--num_cpu_workers", type=int, default=-1,
        help="Number of CPU workers to use for data-loading (per GPU process)"
    )
    parser.add_argument(
        "--max_samples_per_gpu", type=int, default=1,
        help="Maximum number of batch samples to use per GPU."
    )
    parser.add_argument(
        "--detect_anomaly", action='store_true',
        help="Run the forward and backward passes with 'torch.autograd.detect_anomaly'. "
        "Useful for debugging only."
    )
    parser.add_argument(
        "--amp", action='store_true',
        help="Run training with mixed precision."
    )
    parser.add_argument(
        "--ignore_oom_errors", action='store_true',
        help="If set, OOM errors will be ignored i.e. the training skips to the next iteration. "
        "Only applies to single GPU training."
    )
    parser.add_argument(
        "--cv2_num_threads", type=int, required=False,
        help="Number of threads for OpenCV to use. Should be set to a low number."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Value used to seed all the RNGs for torch, numpy, imguag etc."
    )

    parser.add_argument(
        "--master_port", required=False,
        help="Port number to use for DDP. Recommended to not set this."
    )

    parser.add_argument(
        "--wandb_session", required=False,
        help="To log training metrics using Weights and Biases, provide the session name here"
    )
    parser.add_argument(
        "--log_level", required=False, default="INFO",
        help="Log level for the main process (rank 0)"
    )
    parser.add_argument(
        "--subprocess_log_level", required=False, default="WARN",
        help="Log level for all subprocesses (rank >= 1)"
    )

    parser.add_argument(
        "--display_interval", type=int, default=5,
        help="Specifies the iteration interval for printing training metrics to console."
    )
    parser.add_argument(
        "--summary_interval", type=int, default=20,
        help="Specifies the iteration interval for saving metrics summary to tensorboard or Wandb"
    )
    parser.add_argument(
        "--save_ckpts_after", type=int, default=0,
        help="Specifies the number of iterations after which checkpoints will be regularly saved."
    )
    parser.add_argument(
        "--max_ckpts_to_keep", type=int, default=2,
        help="Maximum number of checkpoints to keep on disk. Older checkpoints are automatically deleted."
    )

    cfg_args_group = parser.add_argument_group("Config Arguments")
    cfg.add_args_to_parser(cfg_args_group, suppress_help=True)

    parsed_args = parser.parse_args()
    with main_function_wrapper(master_port=parsed_args.master_port):
        main(parsed_args)
        