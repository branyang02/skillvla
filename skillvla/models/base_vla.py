"""
skillvla/models/base_vla.py

Base class for VLA models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from skillvla.components.action_head.base_action_head import ActionHead
from skillvla.components.lang_encoder.base_lang_encoder import LanguageEncoder
from skillvla.components.llm_backbone.base_llm_backbone import LLMBackbone
from skillvla.components.obs_encoder.base_obs_encoder import ObservationEncoder


class VLA(nn.Module, GenerationMixin, ABC):
    def __init__(
        self, model_id: str, obs_encoder: ObservationEncoder, lang_encoder: LanguageEncoder, llm_backbone: LLMBackbone, action_head: ActionHead
    ):
        super().__init__()
        self.model_id = model_id  # model_id is a string that uniquely identifies the model
        self.obs_encoder = obs_encoder  # observation encoder
        self.lang_encoder = lang_encoder  # language encoder
        self.llm_backbone = llm_backbone  # LLM backbone
        self.action_head = action_head  # action head

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
