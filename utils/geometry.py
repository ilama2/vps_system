import torch


def depth_to_world_coords(depth, pose, intrinsics):
    """
    depth: [B, H, W]
    pose: [B, 4, 4]
    intrinsics: [B, 3, 3]

    returns:
        world_points: [B, 3, H, W]
        valid_mask:   [B, 1, H, W]
    """
    B, H, W = depth.shape
    device = depth.device

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    # [B, H, W]
    x = x.float().unsqueeze(0).expand(B, -1, -1)
    y = y.float().unsqueeze(0).expand(B, -1, -1)

    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)

    Z = depth
    X = (x - cx) / fx * Z
    Y = (y - cy) / fy * Z

    ones = torch.ones_like(Z)

    # [B, 4, H, W]
    cam_points = torch.stack([X, Y, Z, ones], dim=1)

    # [B, 4, N]
    cam_points = cam_points.view(B, 4, -1)

    # batch matrix multiply
    world_points = torch.bmm(pose, cam_points)   # [B, 4, N]

    # [B, 3, H, W]
    world_points = world_points[:, :3, :].view(B, 3, H, W)

    # [B, 1, H, W]
    valid_mask = (depth > 0).float().unsqueeze(1)

    return world_points, valid_mask


def get_7scenes_intrinsics(batch_size, device):
    fx = 585.0
    fy = 585.0
    cx = 320.0
    cy = 240.0

    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)

    return K.unsqueeze(0).repeat(batch_size, 1, 1)