"""
skillvla/components/materialize.py

Factory functions for creating components.
These functions are called by `SkillVLA.load()` to instantiate different components
"""

from typing import Literal, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from skillvla.components.action_head.base_action_head import ActionHead
from skillvla.components.action_head.diffusion_action_head import DiffusionActionHead
from skillvla.components.action_head.openvla_action_head import OpenVLAActionHead
from skillvla.components.llm.base_llm import LLMBackbone
from skillvla.components.llm.llama2 import LLaMa2LLMBackbone
from skillvla.components.skill_selector.base_skill_selector import SkillSelector
from skillvla.components.skill_selector.vq_skill_selector import VQSkillSelector
from skillvla.components.vision.base_vision import ImageTransform, VisionBackbone
from skillvla.components.vision.dinosiglip_vit import DinoSigLIPViTBackbone


def get_vision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    # default vision_backbone_id = dinosiglip-vit-so-224px
    vision_backbone: VisionBackbone = DinoSigLIPViTBackbone(
        vision_backbone_id, image_resize_strategy, default_image_size=224
    )
    image_transform = vision_backbone.get_image_transform()
    return vision_backbone, image_transform


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    # default llm_backbone_id = llama2-7b-pure
    llm_backbone: LLMBackbone = LLaMa2LLMBackbone(
        llm_backbone_id,
        llm_max_length=llm_max_length,
        hf_token=hf_token,
        inference_mode=inference_mode,
    )
    tokenizer = llm_backbone.get_tokenizer()
    return llm_backbone, tokenizer


def get_action_head(
    action_head_id: Literal["openvla", "diffusion"] = "openvla",
    **kwargs,
) -> ActionHead:
    if action_head_id == "openvla":
        assert kwargs.get("tokenizer") is not None, "Tokenizer is required for OpenVLA"
        return OpenVLAActionHead(action_head_id, **kwargs)
    elif action_head_id == "diffusion":
        return DiffusionActionHead(action_head_id)
    else:
        raise ValueError(f"Unknown action head {action_head_id}")


def get_skill_selector(
    skill_selector_id: Literal["vq", "kmeans"] = "vq",
) -> SkillSelector:
    if skill_selector_id == "vq":
        return VQSkillSelector(skill_selector_id)
    elif skill_selector_id == "kmeans":
        return SkillSelector(skill_selector_id)
    else:
        raise ValueError(f"Unknown skill selector {skill_selector_id}")
