import torch
from torch import nn

from models.encoders import DINOv2Encoder
from models.temporal_cross_attention import CurrentQueryTemporalCrossAttention
from models.triplane import TriPlaneMap
from models.multi_hypothesis_head import MultiHypothesisCoordinateHead


class TemporalTriPlaneModel(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vits14_reg",
        use_lora: bool = True,
        lora_rank: int = 8,
        temporal_heads: int = 8,
        plane_res: int = 64,
        plane_dim: int = 64,
        num_hypotheses: int = 4,
    ):
        super().__init__()

        self.encoder = DINOv2Encoder(
            model_name=model_name,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_targets=("attn.qkv", "attn.proj"),
            freeze_backbone_when_lora=True,
            return_tokens=True,
        )

        self.temporal_fusion = CurrentQueryTemporalCrossAttention(
            dim=self.encoder.dim_out,
            num_heads=temporal_heads,
            dropout=0.1,
        )

        self.triplane = TriPlaneMap(
            feat_dim=self.encoder.dim_out,
            plane_res=plane_res,
            plane_dim=plane_dim,
        )

        self.head = MultiHypothesisCoordinateHead(
            dim_in=self.encoder.dim_out,
            num_hypotheses=num_hypotheses,
            hidden_dim=256,
        )

    def forward(self, frames: torch.Tensor):
        """
        frames: [B, T, C, H, W]

        returns:
            coords: [B, K, 3, H', W']
            conf:   [B, K, H', W']
            coarse_xyz: [B, 3, H', W']
        """
        b, t, c, h, w = frames.shape
        frames = frames.view(b * t, c, h, w)

        feat_map, tokens = self.encoder(frames)      # [B*T,C,H',W'], [B*T,N,C]
        _, feat_c, feat_h, feat_w = feat_map.shape

        tokens = tokens.view(b, t, tokens.shape[1], tokens.shape[2])   # [B,T,N,C]
        fused_tokens = self.temporal_fusion(tokens)                     # [B,N,C]

        fused_map = fused_tokens.permute(0, 2, 1).reshape(b, feat_c, feat_h, feat_w)

        tri_feat, coarse_xyz = self.triplane(fused_map)
        coords, conf = self.head(tri_feat)

        return coords, conf, coarse_xyz

    def expected_coords(coords, conf):
        """
        coords: [B,K,3,H,W]
        conf:   [B,K,H,W]
        returns:
            exp_coords: [B,3,H,W]
        """
        return (coords * conf.unsqueeze(2)).sum(dim=1)