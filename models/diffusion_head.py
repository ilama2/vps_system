import torch
import torch.nn as nn

class DiffusionRefiner(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, coords):
        noise = self.net(coords)
        refined = coords - noise
        return refined