import torch
import torch.nn as nn

from skillvla.components.action_head.base_action_head import ActionHead


# HACK: This is a dummy implementation of a diffusion policy.
class DiffusionActionHead(ActionHead):
    def __init__(self, action_head_id: str):
        super().__init__(
            action_head_id=action_head_id,
        )

        self.mlp = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input through the MLP
        output = self.mlp(x)
        return output
