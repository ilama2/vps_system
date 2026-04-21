import torch


def expected_coordinates(pred_coords: torch.Tensor, pred_conf: torch.Tensor) -> torch.Tensor:
    """
    pred_coords: [B, K, 3, H, W]
    pred_conf:   [B, K, H, W]

    returns:
        exp_coords: [B, 3, H, W]
    """
    return (pred_coords * pred_conf.unsqueeze(2)).sum(dim=1)


def make_pixel_grid(batch_size: int, h: int, w: int, image_h: int, image_w: int, device):
    """
    Create pixel-center targets in image coordinates for a feature map.

    returns:
        grid: [B, 2, H, W]
    """
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )

    xs = (xs.float() + 0.5) * (image_w / w)
    ys = (ys.float() + 0.5) * (image_h / h)

    xs = xs.unsqueeze(0).expand(batch_size, -1, -1)
    ys = ys.unsqueeze(0).expand(batch_size, -1, -1)

    return torch.stack([xs, ys], dim=1)   # [B,2,H,W]


def world_to_camera(world_points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """
    world_points: [B, 3, H, W]
    pose: [B, 4, 4]   (camera-to-world)

    returns:
        cam_points: [B, 3, H, W]
    """
    B, _, H, W = world_points.shape
    device = world_points.device

    R = pose[:, :3, :3]          # [B,3,3]
    t = pose[:, :3, 3:4]         # [B,3,1]

    world_flat = world_points.view(B, 3, -1)  # [B,3,N]

    # camera-to-world pose -> inverse to world-to-camera
    cam_flat = torch.bmm(R.transpose(1, 2), world_flat - t)

    cam_points = cam_flat.view(B, 3, H, W)
    return cam_points


def project_camera_points(cam_points: torch.Tensor, intrinsics: torch.Tensor, eps: float = 1e-6):
    """
    cam_points: [B, 3, H, W]
    intrinsics: [B, 3, 3]

    returns:
        uv: [B, 2, H, W]
        z:  [B, 1, H, W]
    """
    B, _, H, W = cam_points.shape

    X = cam_points[:, 0]
    Y = cam_points[:, 1]
    Z = cam_points[:, 2].clamp_min(eps)

    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    uv = torch.stack([u, v], dim=1)      # [B,2,H,W]
    z = Z.unsqueeze(1)                    # [B,1,H,W]
    return uv, z


def reprojection_loss(
    pred_coords: torch.Tensor,
    pred_conf: torch.Tensor,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    image_h: int,
    image_w: int,
):
    """
    pred_coords: [B, K, 3, H, W]
    pred_conf:   [B, K, H, W]
    pose:        [B, 4, 4]
    intrinsics:  [B, 3, 3]
    valid_mask:  [B, 1, H, W]
    """
    B, _, _, H, W = pred_coords.shape
    device = pred_coords.device

    exp_coords = expected_coordinates(pred_coords, pred_conf)  # [B,3,H,W]
    cam_points = world_to_camera(exp_coords, pose)             # [B,3,H,W]
    uv_pred, z = project_camera_points(cam_points, intrinsics) # [B,2,H,W], [B,1,H,W]

    uv_gt = make_pixel_grid(B, H, W, image_h, image_w, device)  # [B,2,H,W]

    reproj_err = torch.abs(uv_pred - uv_gt).mean(dim=1, keepdim=True)  # [B,1,H,W]

    # Only valid depth and positive predicted depth
    valid = valid_mask * (z > 1e-6).float()

    denom = valid.sum().clamp_min(1.0)
    return (reproj_err * valid).sum() / denom


def confidence_entropy_regularization(pred_conf: torch.Tensor, weight: float = 0.001):
    """
    Encourage confident but not numerically unstable hypothesis selection.
    pred_conf: [B, K, H, W]
    """
    entropy = -(pred_conf.clamp_min(1e-8) * pred_conf.clamp_min(1e-8).log()).sum(dim=1)
    return weight * entropy.mean()
