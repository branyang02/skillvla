from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LanguageEncoder(nn.Module, ABC):
    def __init__(self, lang_encoder_id: str) -> None:
        super().__init__()
        self.identifier = lang_encoder_id

    @abstractmethod
    def forward(self, lang_input: torch.Tensor) -> torch.Tensor: ...
