from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ActionHead(nn.Module, ABC):
    """Abstract class for action decoder."""

    def __init__(self, action_head_id: str) -> None:
        super().__init__()
        self.action_head_id = action_head_id

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Predict Action"""
        print(f"Calling ActionHead forward method with input: {input}")
        pass
