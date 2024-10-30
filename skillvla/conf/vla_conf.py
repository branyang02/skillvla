"""
skillvla/conf/vla_conf.py

Dataclass Config class for SkillVLA Training Configuration. Default to Bridge Mixture.
Adjust `vla_id`, `base_vlm`, `data_mix` to use another VLA Policy, VLM, or Data Mixture.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLAConfig:
    """VLA Configuration Dataclass for SkillVLA Training. Default to Bridge Mixture."""

    vla_id: str = "prism-dinosiglip-224px+mx-skillvla-droid"  # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: str = "prism-dinosiglip-224px+7b"  # Base VLM from Prismatic VLM (default to `prism-dinosiglip-224px+7b` that is used by OpenVLA)
    # freeze_vision_backbone: bool = (
    #     True  # Freeze Vision Backbone Parameters (akin to pretraining) TODO: Curr freeze vision backbone
    # )
    # freeze_llm_backbone: bool = True  # Freeze LLM Backbone parameters TODO: Curr freeze vision backbone
    # unfreeze_last_llm_layer: bool = (
    #     True  # Unfreeze final layer of LLM (only takes effect if LLM is frozen) TODO: Curr freeze vision backbone
    # )

    # Data Mixture Parameters
    data_mix: str = "droid"  # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
    shuffle_buffer_size: int = 256_000  # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

    # Optimization Parameters
    epochs: int = 1000  # Epochs to Run (in case `max_steps` is not specified)
    max_steps: Optional[int] = None  # [Optional] Max Gradient Steps to Run (overrides `epochs`)

    expected_world_size: int = 1  # Expected # of GPUs =>> allows us to gate training on hardware TODO: Currently for single GPU Training
    global_batch_size: int = 1  # Global Batch Size (divided across processes / world size) TODO: Currently for single GPU Training
    per_device_batch_size: int = 1  # Per-Device Batch Size (per-process / individual GPU) TODO: Currently for single GPU Training

    learning_rate: float = 2e-5  # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    weight_decay: float = 0.0  # Weight Decay for AdamW Optimizer
    max_grad_norm: float = 1.0  # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str = "constant"  # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float = 0.0  # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str = "fsdp-full-shard"  # Train Strategy (default "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True  # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True  # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True  # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision
