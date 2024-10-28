from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module, ABC):
    def __init__(self, obs_encoder_id: str) -> None:
        super().__init__()
        self.identifier = obs_encoder_id

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor: ...
