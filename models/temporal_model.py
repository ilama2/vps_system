import torch
from torch import nn

from models.encoders import DINOv2Encoder
from models.temporal_attention import TemporalAttentionFusion
from models.coordinate_head import DeterministicCoordinateHead


class TemporalACEGModel(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vits14_reg",
        use_lora: bool = True,
        lora_rank: int = 8,
        temporal_heads: int = 8,
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

        self.temporal_fusion = TemporalAttentionFusion(
            dim=self.encoder.dim_out,
            num_heads=temporal_heads,
            dropout=0.1,
        )

        self.coordinate_head = DeterministicCoordinateHead(dim_in=self.encoder.dim_out)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B, T, C, H, W]
        returns: [B, 3, H', W']
        """
        b, t, c, h, w = frames.shape
        frames = frames.view(b * t, c, h, w)

        feat_map, tokens = self.encoder(frames)   # feat_map [B*T, C, H', W'], tokens [B*T, N, C]
        _, feat_c, feat_h, feat_w = feat_map.shape

        tokens = tokens.view(b, t, tokens.shape[1], tokens.shape[2])  # [B, T, N, C]
        fused_tokens = self.temporal_fusion(tokens)                   # [B, N, C]

        fused_map = fused_tokens.permute(0, 2, 1).reshape(b, feat_c, feat_h, feat_w)
        coords = self.coordinate_head(fused_map)

        return coords