"""
skillvla/conf/vla_conf.py

Based on `prismatic/conf/vla.py` with modifications for SkillVLA.
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

from draccus import ChoiceRegistry


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str                                     # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: Union[str, Path]                      # Base VLM as ID/Path to Run Directory (e.g., `prism-dinosiglip+7b`)
    freeze_vision_backbone: bool                    # Freeze Vision Backbone Parameters (akin to pretraining)
    freeze_llm_backbone: bool                       # Freeze LLM Backbone parameters
    unfreeze_last_llm_layer: bool                   # Unfreeze final layer of LLM (only takes effect if LLM is frozen)

    # Data Mixture Parameters
    data_mix: str                                   # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
    shuffle_buffer_size: int                        # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

    # Optimization Parameters
    epochs: int                                     # Epochs to Run (in case `max_steps` is not specified)
    max_steps: Optional[int]                        # [Optional] Max Gradient Steps to Run (overrides `epochs`)

    expected_world_size: int                        # Expected # of GPUs =>> allows us to gate training on hardware
    global_batch_size: int                          # Global Batch Size (divided across processes / world size)
    per_device_batch_size: int                      # Per-Device Batch Size (per-process / individual GPU)
                                                    #   =>> # of accumulation steps is auto-computed

    learning_rate: float                            # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    weight_decay: float                             # Weight Decay for AdamW Optimizer
    max_grad_norm: float                            # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str                          # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float                             # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str                             # Train Strategy (default "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True      # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True    # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True           # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision

    # fmt: on


# === OpenVLA SkillVLA Training Configurations ===
@dataclass
class BaseExp_DinoSigLIP_224px_SkillVLA(VLAConfig):
    # Shared Configuration Parameters
    vla_id: str = ""
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    freeze_vision_backbone: bool = True  # TODO: Curr freeze vision backbone
    freeze_llm_backbone: bool = True  # TODO: Curr freeze llm backbone
    unfreeze_last_llm_layer: bool = True  # TODO: Curr unfreeze last llm layer

    # Data Mixture Parameters
    data_mix: str = ""
    shuffle_buffer_size: int = 256_000

    # Optimization Parameters
    epochs: int = 1000
    max_steps: Optional[int] = None

    expected_world_size: int = 1  # TODO: Currently for single GPU Training
    global_batch_size: int = 1  # TODO: Currently for single GPU Training
    per_device_batch_size: int = 1  # TODO: Currently for single GPU Training

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"


# Subclass for "Droid" configuration
@dataclass
class Exp_DinoSigLIP_224px_SkillVLA_Droid(BaseExp_DinoSigLIP_224px_SkillVLA):
    vla_id: str = "prism-dinosiglip-224px+mx-skillvla-droid"
    data_mix: str = "droid"


# Subclass for "Bridge" configuration
@dataclass
class Exp_DinoSigLIP_224px_SkillVLA_Bridge(BaseExp_DinoSigLIP_224px_SkillVLA):
    vla_id: str = "prism-dinosiglip-224px+mx-skillvla-bridge"
    data_mix: str = "bridge"


@unique
class VLARegistry(Enum):
    # SkillVLA Droid
    DINO_SIGLIP_224PX_SKILLVLA_DROID = Exp_DinoSigLIP_224px_SkillVLA_Droid
    DINO_SIGLIP_224PX_SKILLVLA_BRIDGE = Exp_DinoSigLIP_224px_SkillVLA_Bridge

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
