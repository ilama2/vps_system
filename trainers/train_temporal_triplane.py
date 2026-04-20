import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.geometry import depth_to_world_coords, get_7scenes_intrinsics
from datasets.seven_scenes_temporal import SevenScenesTemporalDataset
from models.temporal_triplane_model import TemporalTriPlaneModel
from losses.multi_hypothesis_losses import multi_hypothesis_coordinate_loss, coarse_xyz_regularization


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)
    data_root = os.environ.get(
        "DATA_ROOT",
        "/content/drive/MyDrive/chess"
    )

    dataset = SevenScenesTemporalDataset(
        root=data_root,
        T=4,
        image_size=(224, 224),
        skip_short_history=False,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
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

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )

    os.makedirs("checkpoints", exist_ok=True)

    model.train()

    for epoch in range(1):
        total_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            frames = batch["frames"].to(device)
            pose = batch["pose"].to(device)
            depth = batch["depth"].to(device)

            pred_coords, pred_conf, coarse_xyz = model(frames)
            _, _, _, out_h, out_w = pred_coords.shape

            intrinsics = get_7scenes_intrinsics(depth.shape[0], device)

            print("depth shape:", depth.shape)
            print("pose shape:", pose.shape)
            print("pred_coords shape:", pred_coords.shape)
            print("out_h, out_w:", out_h, out_w)

            gt_coords_full, valid_mask_full = depth_to_world_coords(depth, pose, intrinsics)

            print("gt_coords_full shape:", gt_coords_full.shape)
            print("valid_mask_full shape:", valid_mask_full.shape)
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

            loss_main = multi_hypothesis_coordinate_loss(pred_coords, pred_conf, gt_coords, valid_mask)
            loss_reg = coarse_xyz_regularization(coarse_xyz)
            loss = loss_main + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"Loss {loss.item():.6f} | Main {loss_main.item():.6f} | Reg {loss_reg.item():.6f}"
                )

            if batch_idx == 200:
                break

        avg_loss = total_loss / (batch_idx + 1)
        print(f"Epoch {epoch} done | Avg Loss {avg_loss:.6f}")

        torch.save(model.state_dict(), f"checkpoints/temporal_triplane_epoch_{epoch}.pth")
        print(f"Saved checkpoint: checkpoints/temporal_triplane_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()
