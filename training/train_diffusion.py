import torch
from torch.utils.data import DataLoader

from models.diffusion import DiffusionRefiner
from dataset import SceneDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiffusionRefiner().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loader = DataLoader(SceneDataset(), batch_size=4)

for epoch in range(5):
    for _, coords, _ in loader:

        coords = coords.to(device)

        noise = torch.randn_like(coords)
        noisy = coords + noise

        refined = model(noisy)

        loss = ((refined - coords) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Diffusion Epoch:", epoch, loss.item())

torch.save(model.state_dict(), "diffusion.pth")