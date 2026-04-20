import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SevenScenesTemporalDataset(Dataset):
    def __init__(
        self,
        root,
        T=4,
        image_size=(224, 224),
        skip_short_history=False,
    ):
        self.root = root
        self.T = T
        self.image_size = image_size
        self.skip_short_history = skip_short_history
        self.samples = []

        self.build_samples()

    def build_samples(self):
        seqs = [
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ]

        for seq in sorted(seqs):
            seq_path = os.path.join(self.root, seq)

            frames = sorted([
                f for f in os.listdir(seq_path)
                if f.endswith(".color.png")
            ])

            start_idx = self.T if self.skip_short_history else 0

            for i in range(start_idx, len(frames)):
                current = frames[i]

                # oldest -> newest
                history = []
                for t in range(self.T, 0, -1):
                    history_idx = max(0, i - t)
                    history.append(frames[history_idx])

                self.samples.append({
                    "seq": seq,
                    "frame_idx": i,
                    "image": current,
                    "history": history,
                })

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.image_size is not None:
            img = cv2.resize(img, self.image_size)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img
    def load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)

        depth = depth.astype(np.float32) / 1000.0  # mm -> meters

        # resize depth to match image_size
        if self.image_size is not None:
            depth = cv2.resize(depth, self.image_size, interpolation=cv2.INTER_NEAREST)

        return torch.from_numpy(depth).float()
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = sample["seq"]
        seq_path = os.path.join(self.root, seq)

        # current image
        img_path = os.path.join(seq_path, sample["image"])
        image = self.load_image(img_path)

        # history images
        history = []
        for h in sample["history"]:
            h_path = os.path.join(seq_path, h)
            history.append(self.load_image(h_path))
        history = torch.stack(history, dim=0)  # [T, 3, H, W]

        # all frames = [history..., current]
        frames = torch.cat([history, image.unsqueeze(0)], dim=0)  # [T+1, 3, H, W]

        # current pose
        pose_name = sample["image"].replace(".color.png", ".pose.txt")
        pose_path = os.path.join(seq_path, pose_name)

        if not os.path.exists(pose_path):
            raise FileNotFoundError(pose_path)

        pose = torch.tensor(np.loadtxt(pose_path), dtype=torch.float32)

        depth_name = sample["image"].replace(".color.png", ".depth.png")
        depth_path = os.path.join(seq_path, depth_name)

        depth = self.load_depth(depth_path)

        return {
            "frames": frames,
            "image": image,
            "history": history,
            "pose": pose,
            "depth": depth,
            "seq": seq,
            "frame_idx": sample["frame_idx"],
        }

if __name__ == "__main__":
    dataset = SevenScenesTemporalDataset(
        root="/Users/lama/Desktop/aceg_diffusion/7scenes_raw/chess",
        T=4,
        image_size=(224, 224),
        skip_short_history=False,
    )

    sample = dataset[0]
    print("frames :", sample["frames"].shape)
    print("image  :", sample["image"].shape)
    print("history:", sample["history"].shape)
    print("pose   :", sample["pose"].shape)
