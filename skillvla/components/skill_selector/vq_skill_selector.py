"""
skillvla/components/skill_selector/vq_skill_selector.py
"""

from skillvla.components.skill_selector.base_skill_selector import SkillSelector
from skillvla.components.skill_selector.vq.vector_quantize import VectorQuantize


class VQSkillSelector(SkillSelector):
    def __init__(
        self,
        skill_selector_id: str,
        num_skills: int = 500,
        skill_dim: int = 64,
        codebook_dim: int = 16,
        decay: float = 0.99,
        commitment_weight: float = 0.25,
        kmeans_init: bool = False,
    ):
        super().__init__(skill_selector_id=skill_selector_id, num_skills=num_skills)

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
