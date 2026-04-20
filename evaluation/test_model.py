import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seven_scenes_temporal import SevenScenesTemporalDataset
from models.temporal_model import TemporalACEGModel


def main():
    device = "cpu"

    dataset = SevenScenesTemporalDataset(
        root="/Users/lama/Desktop/aceg_diffusion/7scenes_raw/chess",
        T=4,
        image_size=(224, 224),
    )

    model = TemporalACEGModel().to(device)

    # load your trained model
    model.load_state_dict(torch.load("checkpoints/temporal_aceg_epoch_0.pth", map_location=device))
    model.eval()

    sample = dataset[0]

    frames = sample["frames"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_coords = model(frames)[0]  # [3, H, W]

    # visualize X channel
    plt.imshow(pred_coords[0].cpu())
    plt.title("Predicted X coordinate")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()