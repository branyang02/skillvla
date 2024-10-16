"""
skillvla/models/base_vla.py

Base class for VLA model, wrapper around `prismatic.models.vlms.base_vlm.VLM`.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from skillvla.components.action_head import ActionHead


class VLA(VLM, ABC):
    """Base class for VLA model, wrapper around `prismatic.models.vlms.base_vlm.VLM`."""

    def __init__(
        self,
        model_family: str,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        action_head: ActionHead,
        enable_mixed_precision_training: bool = True,
    ) -> None:
        # Initialize parent VLM class
        super().__init__(
            model_family=model_family,
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        self.action_head = action_head

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_family: str,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        action_head: ActionHead,
        **kwargs: str,
    ) -> VLM: ...
