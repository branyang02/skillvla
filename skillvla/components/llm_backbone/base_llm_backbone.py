from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LLMBackbone(nn.Module, ABC):
    def __init__(self, llm_id: str) -> None:
        super().__init__()
        self.identifier = llm_id

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
