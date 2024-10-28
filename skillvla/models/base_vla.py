"""
skillvla/models/base_vla.py

Base class for VLA model, wrapper around `prismatic.models.vlms.base_vlm.VLM`.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


from skillvla.components.action_head.base_action_head import ActionHead
from skillvla.components.llm.base_llm import LLMBackbone
from skillvla.components.vision.base_vision import VisionBackbone
from skillvla.models.base_vlm import VLM

from transformers import GenerationMixin, PretrainedConfig


class VLA(nn.Module, GenerationMixin, ABC):
    def __init__(
        self, model_id: str, obs_encoder: ObservationEncoder, lang_encoder: LanguageEncoder, llm_backbone: LLMBackbone, action_head: ActionHead
    ):
        pass
