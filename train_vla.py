"""
train_vla.py

Usage:
>>> torchrun --standalone --nnodes 1 --nproc-per-node 1 train_vla.py
"""

import sys
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import torch
import tyro
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import PROFILER_KEY, DummyProfiler
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.metric_logging import DiskLogger

from skillvla.util import CONSOLE
from skillvla.util.training_utils import Checkpointer, TensorBoardLogger

# from torchtune.training.lr_schedulers import get_lr


@dataclass
class OptimizerConfig:
    fused: bool = True  # Use fused optimizer


@dataclass
class CheckpointerConfig:
    checkpoint_dir: str = "checkpoints"  # Directory to save checkpoints
    checkpoint_file: Optional[str] = None  # Checkpoint file to resume from, if None, resume from latest checkpoint
    recipe_checkpoint: Optional[str] = None  # Recipe checkpoint file to resume from.
    resume_from_checkpoint: bool = False  # Resume from checkpoint


@dataclass
class TrainConfig:
    optimizer: OptimizerConfig
    checkpointer: CheckpointerConfig

    device: str = "cuda"  # Device to train on
    dtype: str = "bf16"  # Data type to use for training
    fsdp_cpu_offload: bool = True  # Enable CPU offload for FSDP

    # Training
    seed: Optional[int] = 42  # Random seed
    batch_size: int = 1  # Batch size
    epochs: int = 3  # Number of epochs to train for
    max_steps_per_epoch: Optional[int] = None
    resume_from_checkpoint: bool = False  # Resume from checkpoint
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients over
    optimizer_in_bwd: bool = False  # Run optimizer in backward pass
    clip_grad_norm: Optional[float] = None  # Clip gradient norm

    # Memory Management
    enable_activation_checkpointing: bool = True  # Enable activation checkpointing
    enable_activation_offloading: bool = False  # Enable activation offloading
    custom_sharded_layers: Optional[List[str]] = field(default_factory=lambda: ["tok_embeddings", "output"])  # List of layers to shard
    fsdp_cpu_offload: bool = False  # Enable CPU offload for FSDP
    fsdp_reshard_after_forward: bool = True  # Reshard after forward pass
    compile: False  # pytorch compile, set to true for perf/memory improvement

    # Logging
    output_dir: str = "runs"  # Output directory for logs and checkpoints
    log_every_n_steps: int = 1
    log_peak_memory_stats: bool = True

    def __post_init__(self):
        self.dtype: torch.dtype = training.get_dtype(self.dtype, device=self.device)


class TrainVLARecipe(FTRecipeInterface):
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

        if cfg.device != "cuda":
            raise ValueError("Only CUDA is supported for training VLA model.")

        if cfg.dtype == torch.float16:
            raise ValueError("full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead.")

        if cfg.fsdp_cpu_offload and cfg.optimizer.fused and not utils.torch_version_ge("2.4.0"):
            raise RuntimeError("Using fused optimizer on CPU is only supported in PyTorch nightly.")

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        _, rank = training.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Optimizer in backward is not compatible with gradient accumulation or gradient clipping
        if cfg.optimizer_in_bwd:
            if cfg.clip_grad_norm is not None:
                raise RuntimeError(
                    "Gradient clipping is not supported with optimizer in bwd." "Please set clip_grad_norm=None, or optimizer_in_bwd=False."
                )
            if cfg.gradient_accumulation_steps > 1:
                raise RuntimeError(
                    "Gradient accumulation is not supported with optimizer in bwd."
                    "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
                )

        # activation checkpointing/offloading
        if cfg.enable_activation_offloading:
            if not cfg.enable_activation_checkpointing:
                raise RuntimeError("enable_activation_offloading should only be True when enable_activation_checkpointing is True")
        elif cfg.enable_activation_checkpointing:
            CONSOLE.log(
                "[yellow]Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. Enabling activation offloading should reduce memory further."
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

    def setup(self) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self.metric_logger = TensorBoardLogger(log_dir=self.cfg.output_dir, organize_logs=False)
            self.metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self.model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self.cfg.enable_activation_checkpointing,
            enable_activation_offloading=self.cfg.enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
    ) -> nn.Module:
        if self._is_rank_zero:
            CONSOLE.log("FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...")
            init_start = time.perf_counter()

        with training.set_default_dtype(self.cfg.dtype), torch.device("meta"):
            model = SkillVLA(cfg_model)

    def load_checkpoint(self, cfg_checkpointer: CheckpointerConfig) -> Dict[str, Any]:
        self.checkpointer = Checkpointer(**asdict(cfg_checkpointer))

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self.metric_logger.close()
        destroy_process_group()


def main(cfg: TrainConfig):
    """
    Entry Point for training VLA model
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    if cfg.fsdp_cpu_offload:
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()
    CONSOLE.log(f"Running TrainVLARecipe with config: {cfg}")

    recipe = TrainVLARecipe(cfg=cfg)
    recipe.setup()
    # recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)
