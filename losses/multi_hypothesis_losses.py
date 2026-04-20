import torch
import torch.nn.functional as F


def multi_hypothesis_coordinate_loss(pred_coords, pred_conf, gt_coords, mask=None):
    """
    pred_coords: [B, K, 3, H, W]
    pred_conf:   [B, K, H, W]
    gt_coords:   [B, 3, H, W]
    mask:        [B, 1, H, W] or None
    """
    gt = gt_coords.unsqueeze(1)  # [B,1,3,H,W]
    abs_err = torch.abs(pred_coords - gt).mean(dim=2)   # [B,K,H,W]

    weighted = abs_err * pred_conf

    if mask is not None:
        mask = mask.squeeze(1)  # [B,H,W]
        weighted = weighted * mask.unsqueeze(1)
        denom = mask.sum().clamp_min(1.0)
        return weighted.sum() / denom

    return weighted.mean()


def coarse_xyz_regularization(coarse_xyz):
    return 0.001 * coarse_xyz.abs().mean()