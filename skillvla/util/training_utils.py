"""
util/training_utils.py

This module contains utility functions for training models.
"""

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

import torch
import tyro
from numpy import ndarray
from torchtune.training._distributed import get_world_size_and_rank
from torchtune.training.checkpointing._utils import (
    FormattedCheckpointFiles,
    ModelType,
    get_path,
    safe_torch_load,
    save_config,
)

if TYPE_CHECKING:
    from train_vla import TrainConfig


Scalar = Union[torch.Tensor, ndarray, int, float]


class TensorBoardLogger:
    """
    Adapted from `torchtune.training.metric_logging.TensorBoardLogger`

    Logger for use w/ PyTorch's implementation of TensorBoard (https://pytorch.org/docs/stable/tensorboard.html).

    Args:
        log_dir (str): torch.TensorBoard log directory
        organize_logs (bool): If `True`, this class will create a subdirectory within `log_dir` for the current
            run. Having sub-directories allows you to compare logs across runs. When TensorBoard is
            passed a logdir at startup, it recursively walks the directory tree rooted at logdir looking for
            subdirectories that contain tfevents data. Every time it encounters such a subdirectory,
            it loads it as a new run, and the frontend will organize the data accordingly.
            Recommended value is `True`. Run `tensorboard --logdir my_log_dir` to view the logs.
        **kwargs: additional arguments

    Example:
        >>> from torchtune.training.metric_logging import TensorBoardLogger
        >>> logger = TensorBoardLogger(log_dir="my_log_dir")
        >>> logger.log("my_metric", 1.0, 1)
        >>> logger.log_dict({"my_metric": 1.0}, 1)
        >>> logger.close()

    Note:
        This utility requires the tensorboard package to be installed.
        You can install it with `pip install tensorboard`.
        In order to view TensorBoard logs, you need to run `tensorboard --logdir my_log_dir` in your terminal.
    """

    def __init__(self, log_dir: str, organize_logs: bool = True, **kwargs):
        from torch.utils.tensorboard import SummaryWriter

        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_world_size_and_rank()

        # In case organize_logs is `True`, update log_dir to include a subdirectory for the
        # current run
        self.log_dir = os.path.join(log_dir, f"run_{self._rank}_{time.time()}") if organize_logs else log_dir

        # Initialize the log writer only if we're on rank 0.
        if self._rank == 0:
            self._writer = SummaryWriter(log_dir=self.log_dir)

    def log_config(self, config: "TrainConfig") -> None:
        """Save the config to TensorBoard."""
        if self._writer:
            self._writer.add_text("config", str(config), global_step=0)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        for name, data in payload.items():
            self.log(name, data, step)

    def __del__(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None


class Checkpointer:
    """
    Adapted from `torchtune.training.checkpointer.FullModelTorchTuneCheckpointer`
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_file: Optional[str],
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.resume_from_checkpoint = resume_from_checkpoint
        self.recipe_checkpoint = recipe_checkpoint

        if resume_from_checkpoint:
            assert checkpoint_file is not None, "checkpoint_file must be provided when resume_from_checkpoint is True"

    def load_checkpoint(self, weights_only: bool = True) -> Dict[str, Any]:
        # TODO: Implement this method
        pass

    def save_checkpoint(self, model: ModelType, optimizer: Any, scheduler: Any, epoch: int, **kwargs: Any) -> None:
        # TODO: Implement this method
        pass
