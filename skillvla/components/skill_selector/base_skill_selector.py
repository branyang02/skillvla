from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SkillSelector(nn.Module, ABC):
    def __init__(self, skill_selector_id: str) -> None:
        super().__init__()
        self.identifier = skill_selector_id

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor: ...
