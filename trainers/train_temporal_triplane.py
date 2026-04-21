import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.geometry import depth_to_world_coords, get_7scenes_intrinsics
from mydatasets.seven_scenes_temporal import SevenScenesTemporalDataset
from models.temporal_triplane_model import TemporalTriPlaneModel
from losses.multi_hypothesis_losses import multi_hypothesis_coordinate_loss, coarse_xyz_regularization
from losses.reprojection_losses import reprojection_loss, confidence_entropy_regularization


# -------------------------
# ARGPARSE
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="/workspace/checkpoints")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--image_h", type=int, default=224)
    parser.add_argument("--image_w", type=int, default=224)
    parser.add_argument("--history", type=int, default=4)

    parser.add_argument("--plane_res", type=int, default=64)
    parser.add_argument("--plane_dim", type=int, default=64)
    parser.add_argument("--num_hypotheses", type=int, default=4)

    parser.add_argument("--reproj_weight", type=float, default=0.01)
    parser.add_argument("--conf_weight", type=float, default=0.001)

    return parser.parse_args()


# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using device:", device)

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = SevenScenesTemporalDataset(
        root=args.data_root,
        T=args.history,
        image_size=(args.image_w, args.image_h),
        skip_short_history=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    model = TemporalTriPlaneModel(
        model_name="dinov2_vits14_reg",
        use_lora=True,
        lora_rank=8,
        temporal_heads=8,
        plane_res=args.plane_res,
        plane_dim=args.plane_dim,
        num_hypotheses=args.num_hypotheses,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )

    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            frames = batch["frames"].to(device)
            pose = batch["pose"].to(device)
            depth = batch["depth"].to(device)

            pred_coords, pred_conf, coarse_xyz = model(frames)
            _, _, _, out_h, out_w = pred_coords.shape

            intrinsics = get_7scenes_intrinsics(depth.shape[0], device)

            gt_coords_full, valid_mask_full = depth_to_world_coords(depth, pose, intrinsics)

            gt_coords = F.interpolate(
                gt_coords_full,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )

            valid_mask = F.interpolate(
                valid_mask_full,
                size=(out_h, out_w),
                mode="nearest",
            )

            loss_coord = multi_hypothesis_coordinate_loss(
                pred_coords, pred_conf, gt_coords, valid_mask
            )

            loss_reproj = reprojection_loss(
                pred_coords=pred_coords,
                pred_conf=pred_conf,
                pose=pose,
                intrinsics=intrinsics,
                valid_mask=valid_mask,
                image_h=args.image_h,
                image_w=args.image_w,
            )

            loss_conf = confidence_entropy_regularization(
                pred_conf, weight=args.conf_weight
            )

            loss_reg = coarse_xyz_regularization(coarse_xyz)

            loss = loss_coord + args.reproj_weight * loss_reproj + loss_conf + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"Loss {loss.item():.6f} | "
                    f"Coord {loss_coord.item():.6f} | "
                    f"Reproj {loss_reproj.item():.6f} | "
                    f"Conf {loss_conf.item():.6f} | "
                    f"Reg {loss_reg.item():.6f}"
                )

        avg_loss = total_loss / (batch_idx + 1)
        print(f"Epoch {epoch} done | Avg Loss {avg_loss:.6f}")

        ckpt_path = os.path.join(
            args.save_dir,
            f"temporal_triplane_epoch_{epoch}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)
        print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()
