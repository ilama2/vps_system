import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seven_scenes_temporal import SevenScenesTemporalDataset
from models.temporal_model import TemporalACEGModel


def get_scaled_7scenes_intrinsics(
    orig_w=640,
    orig_h=480,
    new_w=224,
    new_h=224,
):
    """
    Scale standard 7-Scenes intrinsics to resized image size.
    """
    fx = 585.0 * (new_w / orig_w)
    fy = 585.0 * (new_h / orig_h)
    cx = 320.0 * (new_w / orig_w)
    cy = 240.0 * (new_h / orig_h)

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return K


def sample_correspondences(pred_coords, num_samples=1024):
    """
    pred_coords: torch.Tensor [3, H, W]

    Returns:
        image_points: [N, 2] float32
        object_points: [N, 3] float32
    """
    c, h, w = pred_coords.shape
    assert c == 3

    pred_coords_np = pred_coords.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    image_points = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    object_points = pred_coords_np.reshape(-1, 3).astype(np.float32)

    # remove bad values
    valid = np.isfinite(object_points).all(axis=1)
    valid &= np.linalg.norm(object_points, axis=1) > 1e-6

    image_points = image_points[valid]
    object_points = object_points[valid]

    if len(image_points) == 0:
        raise ValueError("No valid correspondences found.")

    # random subsample
    if len(image_points) > num_samples:
        idx = np.random.choice(len(image_points), num_samples, replace=False)
        image_points = image_points[idx]
        object_points = object_points[idx]

    return image_points, object_points


def solve_pose_pnp_ransac(image_points, object_points, K):
    """
    Solve camera pose from 2D-3D correspondences.

    Returns:
        success, R, t, inliers
    """
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        iterationsCount=1000,
        reprojectionError=8.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success:
        return False, None, None, None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    return True, R, t, inliers


def pose_matrix_from_rt(R, t):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def rotation_error_deg(R_pred, R_gt):
    """
    Compute geodesic rotation error in degrees.
    """
    R_rel = R_pred @ R_gt.T
    trace = np.trace(R_rel)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def translation_error(t_pred, t_gt):
    return np.linalg.norm(t_pred - t_gt)


def visualize_prediction(image, pred_coords):
    """
    image: [3, H, W]
    pred_coords: [3, H', W']
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))

    axs[0].imshow(image_np)
    axs[0].set_title("Input image")
    axs[0].axis("off")

    axs[1].imshow(pred_coords[0].cpu())
    axs[1].set_title("Pred X")
    axs[1].axis("off")

    axs[2].imshow(pred_coords[1].cpu())
    axs[2].set_title("Pred Y")
    axs[2].axis("off")

    axs[3].imshow(pred_coords[2].cpu())
    axs[3].set_title("Pred Z")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    device = "cpu"

    dataset = SevenScenesTemporalDataset(
        root="/Users/lama/Desktop/aceg_diffusion/7scenes_raw/chess",
        T=4,
        image_size=(224, 224),
        skip_short_history=False,
    )

    model = TemporalACEGModel(
        model_name="dinov2_vits14_reg",
        use_lora=True,
        lora_rank=8,
        temporal_heads=8,
    ).to(device)

    checkpoint_path = "checkpoints/temporal_aceg_epoch_0.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # try a few indices if needed
    for sample_idx in [0, 10, 50, 100, 200]:
        sample = dataset[sample_idx]

    frames = sample["frames"].unsqueeze(0).to(device)   # [1, T+1, 3, H, W]
    image = sample["frames"][-1]
    gt_pose = sample["pose"].numpy()

    with torch.no_grad():
        pred_coords = model(frames)[0]   # [3, H', W']

    visualize_prediction(image, pred_coords)

    # build 2D-3D correspondences
    image_points, object_points = sample_correspondences(pred_coords, num_samples=1024)

    # intrinsics at resized resolution
    K = get_scaled_7scenes_intrinsics(
        orig_w=640,
        orig_h=480,
        new_w=224,
        new_h=224,
    )

    success, R_pred, t_pred, inliers = solve_pose_pnp_ransac(image_points, object_points, K)

    if not success:
        print("PnP failed.")
        return

    pred_pose = pose_matrix_from_rt(R_pred, t_pred)

    R_gt = gt_pose[:3, :3]
    t_gt = gt_pose[:3, 3]

    rot_err = rotation_error_deg(R_pred, R_gt)
    trans_err = translation_error(t_pred, t_gt)

    print("PnP succeeded.")
    print("Predicted pose:\n", pred_pose)
    print("Ground-truth pose:\n", gt_pose)
    print(f"Rotation error (deg): {rot_err:.4f}")
    print(f"Translation error:   {trans_err:.4f}")
    print(f"Num correspondences: {len(image_points)}")
    print(f"Num inliers:         {0 if inliers is None else len(inliers)}")


if __name__ == "__main__":
    main()