import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SevenScenesDataset(Dataset):
    def __init__(self, root, T=4):

        self.root = root
        self.T = T
        self.samples = []

        self.build_samples()

    def build_samples(self):

        seqs = [
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ]

        for seq in seqs:

            seq_path = os.path.join(self.root, seq)

            frames = sorted([
                f for f in os.listdir(seq_path)
                if f.endswith(".color.png")
            ])

            for i in range(len(frames)):

                current = frames[i]

                history = []
                for t in range(1, self.T + 1):
                    history.append(frames[max(0, i - t)])

                self.samples.append({
                    "seq": seq,
                    "image": current,
                    "history": history
                })

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        return torch.tensor(img).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):

        sample = self.samples[idx]
        seq = sample["seq"]
        seq_path = os.path.join(self.root, seq)

        # image
        img_path = os.path.join(seq_path, sample["image"])
        image = self.load_image(img_path)

        # history
        history = []
        for h in sample["history"]:
            h_path = os.path.join(seq_path, h)
            history.append(self.load_image(h_path))

        history = torch.stack(history)

        # pose
        pose_name = sample["image"].replace(".color.png", ".pose.txt")
        pose_path = os.path.join(seq_path, pose_name)

        if not os.path.exists(pose_path):
            raise FileNotFoundError(pose_path)

        pose = torch.tensor(np.loadtxt(pose_path), dtype=torch.float32)

        # normalize pose (IMPORTANT)
        pose[:3, 3] = pose[:3, 3] / 5.0

        return {
            "image": image,
            "history": history,
            "pose": pose,
            "seq": seq
        }

dataset = SevenScenesDataset("/Users/lama/Desktop/aceg_diffusion/7scenes_raw/chess")

sample = dataset[0]

print(sample["image"].shape)
print(sample["history"].shape)
print(sample["pose"].shape)