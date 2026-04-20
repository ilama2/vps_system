import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear")

        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + self.scale * lora_out


def _get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora_to_named_linears(
    model: nn.Module,
    target_keywords=("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
):
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            replacements.append(name)

    for name in replacements:
        parent, child_name = _get_parent_module(model, name)
        old_layer = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(old_layer, rank, alpha, dropout))

    return replacements


def mark_only_lora_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True