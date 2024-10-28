import torch.nn as nn


class SkillSelector(nn.Module):
    def __init__(self):
        super(SkillSelector, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.mlp(x)
