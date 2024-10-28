"""
skillvla/models/skillvla.py

PyTorch Module defining SkillVLA model as a wrapper around `prismatic.models.vlms.base_vlm.VLM`.
Similar implementation as `prismatic.models.vlms.prismatic.PrismaticVLM`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from prismatic.models.load import load_vla
from skillvla.components.action_head.base_action_head import ActionHead
from skillvla.components.lang_encoder.base_lang_encoder import LanguageEncoder
from skillvla.components.llm_backbone.base_llm_backbone import LLMBackbone
from skillvla.components.obs_encoder.base_obs_encoder import ObservationEncoder
from skillvla.components.skill_selector.skill_selector import SkillSelector
from skillvla.models.base_vla import VLA
from skillvla.util import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class SkillVLA(VLA):
    """SkillVLA Main Class"""

    def __init__(
        self,
        model_id: str,
        obs_encoder: ObservationEncoder,
        lang_encoder: LanguageEncoder,
        llm_backbone: LLMBackbone,
        action_head: ActionHead,
        skill_selector: SkillSelector,
    ):
        super().__init__(model_id, obs_encoder, lang_encoder, llm_backbone, action_head)
        self.skill_selector = skill_selector

    @staticmethod
    def load_from_openvla(
        ckpt_path: Union[str, Path],
        hf_token: Optional[str] = None,
        load_for_training: bool = False,
    ) -> SkillVLA:
        """
        Load SkillVLA model from OpenVLA checkpoint.

        Args:
            ckpt_path (Union[str, Path]): Path to OpenVLA checkpoint.
            hf_token (Optional[str], optional): Hugging Face token for downloading model from Hugging Face Hub. Defaults to None.
            load_for_training (bool, optional): Load model for training. Defaults to False.

        Returns:
            SkillVLA: SkillVLA model loaded from OpenVLA checkpoint.
        """
        overwatch.info(f"Loading SkillVLA model from OpenVLA checkpoint: {ckpt_path}")

        # 1. Load base VLA from pretrained OpenVLA checkpoint
        openvla = load_vla(ckpt_path, hf_token=hf_token, load_for_training=load_for_training)

        # 2. Convert OpenVLA to SkillVLA Setup

        # TODO: convert OpenVLA to SkillVLA setup
        return SkillVLA(
            model_id=openvla.model_id,
            obs_encoder=openvla.obs_encoder,
            lang_encoder=openvla.lang_encoder,
            llm_backbone=openvla.llm_backbone,
            action_head=openvla.action_head,
            skill_selector=SkillSelector(),
        )
