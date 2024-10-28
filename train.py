"""
skillvla/train.py

We support 2 methods of training:

1. Train from pretrained Prismatic VLM
>>> (skillvla-env) bash-4.4$ torchrun --standalone --nnodes 1 --nproc-per-node 1 skillvla/train.py

2. Train from pretrained OpenVLA VLA (trained on top of Prismatic VLM)
>>> (skillvla-env) bash-4.4$ torchrun --standalone --nnodes 1 --nproc-per-node 1 skillvla/train.py --pretrained_checkpoint base_model_ckpts/models--openvla--openvla-7b-prismatic/snapshots/5e44aaf23f992e150f26b257500144225ab6643b/checkpoints/step-295000-epoch-40-loss\=0.2200.pt --no_is_resume
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import tyro

from skillvla.conf.vla_conf import VLAConfig
from skillvla.datasets.materialize import get_vla_dataset_and_collator
from skillvla.models.skillvla import SkillVLA
from skillvla.util import initialize_overwatch
from skillvla.util.torch_utils import set_global_seed

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # VLA Configuration
    vla: VLAConfig  # VLA Configuration

    # Directory Paths
    data_root_dir: str = "/scratch/jqm9ba/datasets"  # Path to Open-X dataset directory
    run_root_dir: str = "runs"  # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[str] = None  # Absolute Path to Pretrained OpenVLA Checkpoint
    is_resume: bool = True  # Whether we are continuing a prior training run
    resume_step: Optional[int] = None  # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None  # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None  # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None  # Extra note for logging, Weights & Biases
    save_interval: int = 2500  # Interval for saving checkpoints (in steps)
    image_aug: bool = False  # Whether to enable image augmentations
    seed: int = 7  # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: str = ".hf_token"  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: List[str] = field(default_factory=lambda: ["jsonl"])  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "openvla"  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"  # Name of entity to log under

    def __post_init__(self) -> None:
        # 1. Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

        # 2. Setup `run_root_dir`
        vla_id = self.vla.vla_id
        self.run_id = (
            f"{vla_id}+n{self.vla.expected_world_size // 8}+b{self.per_device_batch_size}+x{self.seed}"
            if self.run_id is None
            else self.run_id
        )
        if self.run_id_note is not None:
            self.run_id += f"--{self.run_id_note}"
        if self.image_aug:
            self.run_id += "--image_aug"

        os.makedirs(run_dir := (os.path.join(self.run_root_dir, self.run_id)), exist_ok=True)
        os.makedirs(os.path.join(self.run_root_dir, self.run_id, "checkpoints"), exist_ok=True)

        overwatch.info(f"Saving Training Logs & Checkpoints to Directory: {run_dir}")

        # Save configuration
        if overwatch.is_rank_zero():
            yaml_cfg_data = tyro.extras.to_yaml(self)
            overwatch.info("[bold purple]###### Configuration ######")
            overwatch.info(yaml_cfg_data)
            with open(os.path.join(run_dir, "config.yaml"), "w") as yaml_file:
                yaml_file.write(yaml_cfg_data)


def train(cfg: TrainConfig) -> None:
    overwatch.info("[bold purple]###### OpenVLA Training :: Warming Up ######")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Start =>> Build Directories and Set Randomness
    hf_token = Path(cfg.hf_token).read_text().strip()
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch

    vla = SkillVLA.load(
        cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True, pretrained_checkpoint=cfg.pretrained_checkpoint
    )

    # [Validate] Model should be in Full Precision!
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLA parameter not in full precision: {param}"

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info("[bold purple]###### Configuring Training Parameters ######")
    stage = "skill-learning"  # TODO: Allow more stages for freezing
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{cfg.vla.vla_id}` => Stage: `{stage}`")
    vla.freeze_backbones(stage)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )
    for module_key in vla.all_module_keys:
        num_module_params = sum(p.numel() for p in vla.get_module(module_key).parameters())
        num_trainable_module_params = sum(p.numel() for p in vla.get_module(module_key).parameters() if p.requires_grad)
        overwatch.info(
            f"# Params in `{module_key}` (in millions): {num_module_params / 10**6:.3f} Total, {num_trainable_module_params / 10**6:.3f} Trainable"
        )

    # TODO: Implement optimizer and scheduler
    # Get VLA Dataset & Collator
    vla_dataset, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vla.vision_backbone.get_image_transform(),
        tokenizer=vla.llm_backbone.get_tokenizer(),
        action_head=vla.action_head,
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    train(cfg)