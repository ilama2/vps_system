import torch
from torch import nn


class MultiHypothesisCoordinateHead(nn.Module):
    """
    Predict K coordinate hypotheses + confidence.
    """
    def __init__(self, dim_in: int, num_hypotheses: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.K = num_hypotheses

        self.shared = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.coord_head = nn.Conv2d(hidden_dim, self.K * 3, kernel_size=1)
        self.conf_head = nn.Conv2d(hidden_dim, self.K, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]

        returns:
            coords: [B, K, 3, H, W]
            conf:   [B, K, H, W]  (softmax over K)
        """
        b, _, h, w = x.shape
        feat = self.shared(x)

        coords = self.coord_head(feat).view(b, self.K, 3, h, w)
        conf_logits = self.conf_head(feat).view(b, self.K, h, w)
        conf = torch.softmax(conf_logits, dim=1)

        return coords, conf