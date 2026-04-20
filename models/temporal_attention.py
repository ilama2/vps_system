import math
import torch
from torch import nn


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 16):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C]
        t = x.shape[1]
        return x + self.pe[:, :t].unsqueeze(2)


class TemporalAttentionFusion(nn.Module):
    """
    Input:
        x: [B, T, N, C]
    Output:
        fused: [B, N, C]
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.pos = TemporalPositionalEncoding(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, c = x.shape
        x = self.pos(x)

        # fuse across time for each patch location
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, c)  # [B*N, T, C]

        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        x = x.mean(dim=1)  # [B*N, C]
        x = x.view(b, n, c)

        return x