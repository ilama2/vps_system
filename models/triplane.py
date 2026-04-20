import torch
import torch.nn.functional as F
from torch import nn


class TriPlaneMap(nn.Module):
    """
    Learnable tri-plane scene representation.

    Planes:
      - XY
      - XZ
      - YZ

    Input:
        token features projected to coarse coordinates

    Output:
        tri-plane fused features [B, C, H, W]
    """
    def __init__(self, feat_dim: int, plane_res: int = 64, plane_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.plane_res = plane_res
        self.plane_dim = plane_dim

        # learned scene planes
        self.xy_plane = nn.Parameter(torch.randn(1, plane_dim, plane_res, plane_res) * 0.02)
        self.xz_plane = nn.Parameter(torch.randn(1, plane_dim, plane_res, plane_res) * 0.02)
        self.yz_plane = nn.Parameter(torch.randn(1, plane_dim, plane_res, plane_res) * 0.02)

        # predict coarse latent xyz from feature tokens
        self.coord_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, 3, kernel_size=1),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(plane_dim * 3 + feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
        )

    def sample_plane(self, plane: torch.Tensor, grid: torch.Tensor):
        """
        plane: [1, C, R, R]
        grid:  [B, H, W, 2] in [-1, 1]
        """
        b = grid.shape[0]
        plane = plane.expand(b, -1, -1, -1)
        sampled = F.grid_sample(
            plane,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled

    def forward(self, feat_map: torch.Tensor):
        """
        feat_map: [B, C, H, W]

        Returns:
            fused: [B, C, H, W]
            coarse_xyz: [B, 3, H, W]
        """
        b, c, h, w = feat_map.shape

        coarse_xyz = self.coord_proj(feat_map)  # [B, 3, H, W]

        # normalize coarse xyz to [-1,1] with tanh
        xyz = torch.tanh(coarse_xyz)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        xy_grid = torch.stack([x, y], dim=-1)   # [B,H,W,2]
        xz_grid = torch.stack([x, z], dim=-1)
        yz_grid = torch.stack([y, z], dim=-1)

        xy_feat = self.sample_plane(self.xy_plane, xy_grid)
        xz_feat = self.sample_plane(self.xz_plane, xz_grid)
        yz_feat = self.sample_plane(self.yz_plane, yz_grid)

        fused = torch.cat([feat_map, xy_feat, xz_feat, yz_feat], dim=1)
        fused = self.out_proj(fused)

        return fused, coarse_xyz