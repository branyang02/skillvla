"""
skillvla/components/obs_encoder/openvla_obs_encoder.py

This module contains the OpenVLAObsEncoder class.
"""

import torch

from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone
from skillvla.components.obs_encoder.base_obs_encoder import ObservationEncoder


class OpenVLAObsEncoder(ObservationEncoder):
    def __init__(self, vit_model: DinoSigLIPViTBackbone, obs_encoder_id: str = "openvla_obs_encoder"):
        super().__init__(obs_encoder_id)
        self.vit_model = vit_model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass
