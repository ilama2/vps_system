import torch


def coordinate_l1_loss(pred_coords: torch.Tensor, gt_coords: torch.Tensor, mask: torch.Tensor = None):
    """
    pred_coords: [B, 3, H, W]
    gt_coords:   [B, 3, H, W]
    mask:        [B, 1, H, W] or None
    """
    loss = torch.abs(pred_coords - gt_coords)

    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    return loss.mean()