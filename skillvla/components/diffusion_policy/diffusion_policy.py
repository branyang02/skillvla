import torch
import torch.nn as nn

from skillvla.components.action_head import ActionHead


# HACK: This is a dummy implementation of a diffusion policy.
class DiffusionPolicy(ActionHead):
    def __init__(self):
        super(DiffusionPolicy, self).__init__(action_head_id="dummy_diffusion_policy")

        self.mlp = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Pass the input through the MLP
        output = self.mlp(input)
        return output
