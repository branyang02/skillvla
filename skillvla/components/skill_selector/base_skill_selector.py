from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SkillSelector(nn.Module, ABC):
    """Abstract class for skill selector models."""

    def __init__(
        self,
        skill_selector_id: str,
        num_skills: int = 100,
    ) -> None:
        super().__init__()
        self.identifier = skill_selector_id
        self.num_skills = num_skills

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
