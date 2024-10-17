"""
skillvla/components/skill_predictor/skill_predictor.py

Create SkillPredictor class to predict the skill from codebook
"""

import torch
import torch.nn as nn

from skillvla.components.skill_selector.vq.vector_quantize import VectorQuantize


class SkillPredictor(nn.Module):
    def __init__(
        self,
        num_skills: int = 500,
        skill_dim: int = 64,
        codebook_dim: int = 16,
        decay: float = 0.99,
        commitment_weight: float = 0.25,
        kmeans_init: bool = False,
    ):
        super(SkillPredictor, self).__init__()
        self.num_skills = num_skills

        self.vq = VectorQuantize(
            dim=skill_dim,
            codebook_dim=codebook_dim,
            codebook_size=num_skills,
            decay=decay,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
        )

        # TODO: create CausualTransformer (Maybe GPT-2) to predict the next skill from codebook

    def forward(self, x):
        return self.vq(x)
