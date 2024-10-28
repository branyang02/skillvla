"""
skillvla/models/skillvla.py

PyTorch Module defining SkillVLA model as a wrapper around `prismatic.models.vlms.base_vlm.VLM`.
Similar implementation as `prismatic.models.vlms.prismatic.PrismaticVLM`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import torch

from prismatic.conf.models import ModelConfig
from skillvla.components.action_head.base_action_head import ActionHead
from skillvla.components.llm.base_llm import LLMBackbone
from skillvla.components.materialize import (
    get_action_head,
    get_llm_backbone_and_tokenizer,
    get_skill_selector,
    get_vision_backbone_and_transform,
)
from skillvla.components.skill_selector.vq_skill_selector import SkillSelector
from skillvla.components.vision.base_vision import VisionBackbone
from skillvla.models.base_vla import VLA
from skillvla.util import initialize_overwatch
from skillvla.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
VLA_HF_HUB_REPO = "openvla/openvla-dev"

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class SkillVLA(VLA):
    """SkillVLA Main Class"""

    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        action_head: ActionHead,
        skill_selector: SkillSelector,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        **kwargs,
    ) -> None:
        super().__init__(
            "prismatic",  # TODO: check what model_family does
            model_id,
            vision_backbone,
            llm_backbone,
            action_head,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Skill Selector
        self.skill_selector = skill_selector

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector", "action_head", "skill_selector"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    @staticmethod
    def load_from_openvla(
        ckpt_path: Union[str, Path],
        hf_token: Optional[str] = None,
        load_for_training: bool = False,
    ):
        checkpoint_pt = Path(ckpt_path)
        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        with open(config_json, "r") as f:
            vla_cfg = json.load(f)["vla"]
            model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)

        # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
        #   =>> Print Minimal Config
        overwatch.info(
            f"[bold green]Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
            f"             [bold green]Vision Backbone =>> [bold blue]{model_cfg.vision_backbone_id}[/]\n"
            f"             [bold green]LLM Backbone    =>> [bold blue]{model_cfg.llm_backbone_id}[/]\n"
            f"             [bold green]Arch Specifier  =>> [bold blue]{model_cfg.arch_specifier}[/]\n"
            f"             [bold green]Checkpoint Path =>> [underline]{checkpoint_pt}[/]"
        )

        # Instantiate VLA Components
        overwatch.info("[bold purple]###### Instantiating VLA Components ######")

        # Instantiate Vision Backbone
        overwatch.info(f"[bold yellow]1. Instantiating Vision Backbone [bold blue]{model_cfg.vision_backbone_id}[/]")
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            model_cfg.vision_backbone_id,
            model_cfg.image_resize_strategy,
        )

        # Instantiate LLM Backbone --> note `inference_mode = True` by default when calling `load()`
        overwatch.info(
            f"[bold yellow]2. Instantiating Pretrained LLM [bold blue]{model_cfg.llm_backbone_id}[/] [bold yellow]via HF Transformers"  # noqa: E501
        )
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            model_cfg.llm_backbone_id,
            llm_max_length=model_cfg.llm_max_length,
            hf_token=hf_token,
            inference_mode=not load_for_training,
        )

        # Instantiate Action Head
        overwatch.info("[bold yellow]3. Instantiating Action Head [bold blue]openvla[/]")  # TODO: Move action head param to config
        action_head = get_action_head(
            action_head_id="openvla",
            tokenizer=tokenizer,
            norm_stats=norm_stats,
        )  # tokenizer is needed for OpenVLAActionHead

        # Instantiate Skill Selector
        overwatch.info("[bold yellow]4. Instantiating Skill Selector [bold blue]vq[/]")  # TODO: Move skill selector param to config
        skill_selector = get_skill_selector(skill_selector_id="vq")

        # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
        vla = SkillVLA.from_pretrained(
            checkpoint_pt,
            model_cfg.model_id,
            vision_backbone,
            llm_backbone,
            action_head=action_head,
            skill_selector=skill_selector,
            arch_specifier=model_cfg.arch_specifier,
            freeze_weights=not load_for_training,
        )

        return vla

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        action_head: ActionHead,
        skill_selector: SkillSelector,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        **kwargs,
    ) -> SkillVLA:
        """Initialize a `SkillVLA` from a pretrained checkpoint, freezing all weights, tailored for inference."""
        overwatch.info("[bold purple]###### Loading VLA from Pretrained Checkpoint ######")
        vla = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            action_head,
            skill_selector,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # NOTE: We only have pretrained weights (Prismatic) for vision_backbone, llm_backbone, and projector,
        #      so we need to train the action_head (if applicable) and skill_selector from scratch.
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        overwatch.info(f"[bold yellow]Checkpoint:[/] [underline]{pretrained_checkpoint}")
        overwatch.info(f"[bold yellow]Available Modules:[/] {model_state_dict.keys()}")

        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        # Load Weights
        vla.projector.load_state_dict(model_state_dict["projector"])
        vla.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vla.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vla.requires_grad_(False)
            vla.eval()

        return vla
