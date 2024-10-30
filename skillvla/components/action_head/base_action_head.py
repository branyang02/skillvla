from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ActionHead(nn.Module, ABC):
    """Abstract class for action decoder."""

    def __init__(self, action_head_id: str) -> None:
        super().__init__()
        self.identifier = action_head_id

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
