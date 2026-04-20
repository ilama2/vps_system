import torch
from torch import nn

from models.lora import apply_lora_to_named_linears, mark_only_lora_trainable


class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vits14_reg",
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets=("attn.qkv", "attn.proj"),
        freeze_backbone_when_lora: bool = True,
        return_tokens: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora
        self.return_tokens = return_tokens

        self.dinov2 = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            verbose=True,
            trust_repo=True,
            source="github",
            skip_validation=True,
        )

        self.subsample_factor = self.dinov2.patch_embed.patch_size[0]
        self.dim_out = self.dinov2.embed_dim

        if self.use_lora:
            replaced = apply_lora_to_named_linears(
                self.dinov2,
                target_keywords=lora_targets,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            if freeze_backbone_when_lora:
                mark_only_lora_trainable(self.dinov2)

            print(f"LoRA injected into {len(replaced)} layers")

    def forward(self, images: torch.Tensor):
        """
        images: [B, C, H, W]
        returns:
            feat_map: [B, C_out, H', W']
            tokens:   [B, N, C_out]
        """
        b, c, h, w = images.shape

        if c == 1:
            images = images.expand(-1, 3, -1, -1)

        h_crop = self.subsample_factor * (h // self.subsample_factor)
        w_crop = self.subsample_factor * (w // self.subsample_factor)
        images = images[..., :h_crop, :w_crop]

        features = self.dinov2.forward_features(images)
        patch_tokens = features["x_norm_patchtokens"]  # [B, N, C]

        feat_map = (
            patch_tokens.permute(0, 2, 1)
            .reshape(b, -1, h_crop // self.subsample_factor, w_crop // self.subsample_factor)
        )

        if self.return_tokens:
            return feat_map, patch_tokens

        return feat_map