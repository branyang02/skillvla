"""
skillvla/models/skillvla.py

PyTorch Module defining SkillVLA model as a wrapper around `prismatic.models.vlms.base_vlm.VLM`.
Similar implementation as `prismatic.models.vlms.prismatic.PrismaticVLM`.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers import LlamaTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.conf.models import ModelConfig
from skillvla.components.action_head.base_action_head import ActionHead
from skillvla.components.llm.base_llm import LLMBackbone
from skillvla.components.llm.prompting.base_prompter import PromptBuilder
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
    def load(
        model_id_or_path: Union[str, Path],
        hf_token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        load_for_training: bool = False,
        pretrained_checkpoint: Optional[Path] = None,
    ) -> SkillVLA:
        """
        If `pretrained_checkpoint` is provided, we load a pretrained **OpenVLA** checkpoint;
        Otherwise, we load a pretrained **Prismatic** checkpoint from HF Hub.
        """

        if pretrained_checkpoint is not None:
            checkpoint_pt = Path(pretrained_checkpoint)
            # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
            assert (checkpoint_pt.suffix == ".pt") and (
                checkpoint_pt.parent.name == "checkpoints"
            ), "Invalid checkpoint!"
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
                f"[bold yellow]2. Instantiating Pretrained LLM [bold blue]{model_cfg.llm_backbone_id}[/] [bold yellow]via HF Transformers"
            )
            llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
                model_cfg.llm_backbone_id,
                llm_max_length=model_cfg.llm_max_length,
                hf_token=hf_token,
                inference_mode=not load_for_training,
            )

            # Instantiate Action Head
            overwatch.info(
                "[bold yellow]3. Instantiating Action Head [bold blue]openvla[/]"
            )  # TODO: Move action head param to config
            action_head = get_action_head(
                action_head_id="openvla",
                tokenizer=tokenizer,
                norm_stats=norm_stats,
            )  # tokenizer is needed for OpenVLAActionHead

            # Instantiate Skill Selector
            overwatch.info(
                "[bold yellow]4. Instantiating Skill Selector [bold blue]vq[/]"
            )  # TODO: Move skill selector param to config
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

        # Load from HF Hub
        overwatch.info(f"Downloading `{(model_id := model_id_or_path)} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
            )
        # Load Model Config from `config.json`
        with open(config_json, "r") as f:
            model_cfg = json.load(f)["model"]

        # = Load Individual Components necessary for Instantiating a VLM =
        #   =>> Print Minimal Config
        overwatch.info(
            f"[bold green]Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
            f"             [bold green]Vision Backbone =>> [bold blue]{model_cfg['vision_backbone_id']}[/]\n"
            f"             [bold green]LLM Backbone    =>> [bold blue]{model_cfg['llm_backbone_id']}[/]\n"
            f"             [bold green]Arch Specifier  =>> [bold blue]{model_cfg['arch_specifier']}[/]\n"
            f"             [bold green]Checkpoint Path =>> [underline]{checkpoint_pt}[/]"
        )

        # Instantiate VLA Components
        overwatch.info("[bold purple]###### Instantiating VLA Components ######")

        # Instantiate Vision Backbone
        overwatch.info(f"[bold yellow]1. Instantiating Vision Backbone [bold blue]{model_cfg.vision_backbone_id}[/]")
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            model_cfg["vision_backbone_id"],
            model_cfg["image_resize_strategy"],
        )

        # Instantiate LLM Backbone --> note `inference_mode = True` by default when calling `load()`
        overwatch.info(
            f"[bold yellow]2. Instantiating Pretrained LLM [bold blue]{model_cfg.llm_backbone_id}[/] [bold yellow]via HF Transformers"
        )
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            model_cfg["llm_backbone_id"],
            llm_max_length=model_cfg.get("llm_max_length", 2048),
            hf_token=hf_token,
            inference_mode=not load_for_training,
        )

        # Instantiate Action Head
        overwatch.info(
            "[bold yellow]3. Instantiating Action Head [bold blue]openvla[/]"
        )  # TODO: Move action head param to config
        action_head = get_action_head(
            action_head_id="openvla",
            tokenizer=tokenizer,
        )  # tokenizer is needed for OpenVLAActionHead

        # Instantiate Skill Selector
        overwatch.info(
            "[bold yellow]4. Instantiating Skill Selector [bold blue]vq[/]"
        )  # TODO: Move skill selector param to config
        skill_selector = get_skill_selector(skill_selector_id="vq")  # TODO: Move skill selector param to config

        # Instantiate SkillVLA, same as `PrismaticVLM.from_pretrained()` method.
        vla = SkillVLA.from_pretrained(
            checkpoint_pt,
            model_cfg["model_id"],
            vision_backbone,
            llm_backbone,
            action_head,
            skill_selector,
            arch_specifier=model_cfg["arch_specifier"],
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

    @torch.inference_mode()
    def predict_action(
        self, image: Image, instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        # FIXME
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(SkillVLA, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_head.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    # FIXME this belonged to OpenVLA class from prismatic
    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        """ONLY USED IN INFERENCE"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    # FIXME this belonged to OpenVLA class from prismatic
    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """ONLY USED IN INFERENCE"""
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.action_head.norm_stats, unnorm_key)

        return len(self.action_head.norm_stats[unnorm_key]["action"]["q01"])

    # FIXME this belonged to OpenVLA class from prismatic
    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """ONLY USED IN INFERENCE"""
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.action_head.norm_stats, unnorm_key)

        return self.action_head.norm_stats[unnorm_key]["action"]

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        # TODO: Implement this method to allow for more stages based on openvla training code
        """
        if stage == "skill-learning":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(False)
            self.action_head.requires_grad_(True)  # no parameters if action head is not trainable
            self.skill_selector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone", "action_head", "skill_selector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Action Head `{self.action_head.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Skill Selector `{self.skill_selector.identifier}`", ctx_level=1)
        else:
            raise NotImplementedError(f"Stage {stage} is not supported!")

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    # ruff: noqa: C901
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Run Visual Feature Extraction
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
            )

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === GenerationMixin Methods ===
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                pixel_values=pixel_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                **kwargs
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text
