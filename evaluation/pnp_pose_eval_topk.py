import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mydatasets.seven_scenes_temporal import SevenScenesTemporalDataset
from models.temporal_triplane_model import TemporalTriPlaneModel


def get_scaled_7scenes_intrinsics(orig_w=640, orig_h=480, new_w=224, new_h=224):
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

def extract_topk_correspondences(coords, conf, image_h=224, image_w=224, topk=512, min_conf=0.2):
    """
    coords: [K, 3, H, W]
    conf:   [K, H, W]
    """
    K, _, Hf, Wf = coords.shape

    best_idx = conf.argmax(dim=0)         # [H,W]
    best_scores = conf.max(dim=0).values  # [H,W]

    coords_perm = coords.permute(2, 3, 0, 1)  # [H,W,K,3]

    row_ids = torch.arange(Hf).unsqueeze(1).expand(Hf, Wf)
    col_ids = torch.arange(Wf).unsqueeze(0).expand(Hf, Wf)

    best_coords = coords_perm[row_ids, col_ids, best_idx]   # [H,W,3]

    ys, xs = np.meshgrid(np.arange(Hf), np.arange(Wf), indexing="ij")
    xs = (xs + 0.5) * (image_w / Wf)
    ys = (ys + 0.5) * (image_h / Hf)

    image_points = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    object_points = best_coords.cpu().numpy().reshape(-1, 3).astype(np.float32)
    scores = best_scores.cpu().numpy().reshape(-1)

    valid = np.isfinite(object_points).all(axis=1)
    valid &= np.linalg.norm(object_points, axis=1) > 1e-6
    valid &= scores > min_conf

    image_points = image_points[valid]
    object_points = object_points[valid]
    scores = scores[valid]

    if len(scores) == 0:
        raise ValueError("No correspondences survived confidence filtering.")

    idx = np.argsort(-scores)[:topk]

    return image_points[idx], object_points[idx], scores[idx]


def solve_pose_pnp_ransac(image_points, object_points, K):
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        iterationsCount=2000,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success:
        return False, None, None, None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return True, R, t, inliers


def rotation_error_deg(R_pred, R_gt):
    R_rel = R_pred @ R_gt.T
    trace = np.trace(R_rel)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def translation_error(t_pred, t_gt):
    return np.linalg.norm(t_pred - t_gt)


def main():
    device = "cpu"

    dataset = SevenScenesTemporalDataset(
        root="/content/drive/MyDrive/chess",
        T=4,
        image_size=(224, 224),
        skip_short_history=False,
    )

    model = TemporalTriPlaneModel(
        model_name="dinov2_vits14_reg",
        use_lora=True,
        lora_rank=8,
        temporal_heads=8,
        plane_res=64,
        plane_dim=64,
        num_hypotheses=4,
    ).to(device)

    ckpt = "checkpoints/temporal_triplane_epoch_5.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    sample = dataset[0]
    frames = sample["frames"].unsqueeze(0).to(device)
    gt_pose = sample["pose"].numpy()

    with torch.no_grad():
        coords, conf, _ = model(frames)
        coords = coords[0]  # [K,3,H,W]
        conf = conf[0]      # [K,H,W]

    image_points, object_points, scores = extract_topk_correspondences(
        coords,
        conf,
        image_h=224,
        image_w=224,
        topk=512,
    )

    Kmat = get_scaled_7scenes_intrinsics(640, 480, 224, 224)
    success, R_pred, t_pred, inliers = solve_pose_pnp_ransac(image_points, object_points, Kmat)

    if not success:
        print("PnP failed.")
        return

    R_gt = gt_pose[:3, :3]
    t_gt = gt_pose[:3, 3]

    print("PnP succeeded.")
    print(f"Rotation error (deg): {rotation_error_deg(R_pred, R_gt):.4f}")
    print(f"Translation error:   {translation_error(t_pred, t_gt):.4f}")
    print(f"Num correspondences: {len(image_points)}")
    print(f"Num inliers:         {0 if inliers is None else len(inliers)}")


if __name__ == "__main__":
    main()
