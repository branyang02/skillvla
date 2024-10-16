from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ActionDecoder(nn.Module, ABC):
    """Abstract class for action decoder."""

    def __init__(self, action_decoder_id: str) -> None:
        super().__init__()
        self.action_decoder_id = action_decoder_id

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Predict Action"""
        print(f"Calling ActionDecoder forward method with input: {input}")
        pass
