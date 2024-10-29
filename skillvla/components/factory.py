"""
components/component_factory.py

This module contains factory functions for creating VLA components.
"""

from prismatic.models.vlas.openvla import OpenVLA
from skillvla.components.action_head.base_action_head import ActionHead
from skillvla.components.lang_encoder.base_lang_encoder import LanguageEncoder
from skillvla.components.llm_backbone.base_llm_backbone import LLMBackbone
from skillvla.components.obs_encoder.base_obs_encoder import ObservationEncoder


def create_observation_encoder_from_openvla(openvla: OpenVLA) -> ObservationEncoder:
    pass


def create_language_encoder_from_openvla(openvla: OpenVLA) -> LanguageEncoder:
    pass


def create_llm_backbone_from_openvla(openvla: OpenVLA) -> LLMBackbone:
    pass


def create_action_head_from_openvla(openvla: OpenVLA) -> ActionHead:
    pass
