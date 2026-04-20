import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, dim=768):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8),
            num_layers=4
        )

        self.coord_head = nn.Linear(dim, 3)
        self.uncertainty_head = nn.Linear(dim, 1)

    def forward(self, features, map_code):
        B, N, D = features.shape
        map_code = map_code.repeat(B, 1, 1)

        x = torch.cat([features, map_code], dim=1)
        x = self.transformer(x)

        x = x[:, :N]

        coords = self.coord_head(x)
        uncertainty = torch.sigmoid(self.uncertainty_head(x))

        return coords, uncertainty