import os
import sys
import torch
from torch.utils.data import DataLoader

# allow running from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.geometry import depth_to_world_coords, get_7scenes_intrinsics
from datasets.seven_scenes_temporal import SevenScenesTemporalDataset
from models.temporal_model import TemporalACEGModel
from losses.coordinate_losses import coordinate_l1_loss


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)

    dataset = SevenScenesTemporalDataset(
        root="/Users/lama/Desktop/aceg_diffusion/7scenes_raw/chess",
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

    model = TemporalACEGModel(
        model_name="dinov2_vits14_reg",
        use_lora=True,
        lora_rank=8,
        temporal_heads=8,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    os.makedirs("checkpoints", exist_ok=True)

    model.train()

    for epoch in range(4):
        total_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            frames = batch["frames"].to(device)   # [B, T+1, 3, H, W]
            pose = batch["pose"].to(device)       # [B, 4, 4]
            depth = batch["depth"].to(device)     # [B, H, W]

            pred_coords = model(frames)           # [B, 3, H', W']
            _, _, out_h, out_w = pred_coords.shape

            intrinsics = get_7scenes_intrinsics(depth.shape[0], device)

            gt_coords_full = depth_to_world_coords(depth, pose, intrinsics)

            gt_coords = torch.nn.functional.interpolate(
                gt_coords_full,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )

            loss = coordinate_l1_loss(pred_coords, gt_coords)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.6f}")


        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} done | Avg Loss {avg_loss:.6f}")
        torch.save(model.state_dict(), f"checkpoints/temporal_aceg_epoch_{epoch}.pth")
        print(f"Saved checkpoint: checkpoints/temporal_aceg_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()