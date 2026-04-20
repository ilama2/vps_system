import torch
from torch.utils.data import DataLoader

from models.encoder import Encoder
from models.regressor import Regressor
from dataset import SceneDataset
from utils.loss import reprojection_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = Encoder().to(device)
regressor = Regressor().to(device)

map_code = torch.load("map_code.pth").to(device)
map_code.requires_grad = False

optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-4)

loader = DataLoader(SceneDataset(), batch_size=4)

for epoch in range(5):
    for img, coords_gt, pixels in loader:

        img = img.to(device)
        pixels = pixels.to(device)

        features = encoder(img)
        coords_pred, _ = regressor(features, map_code)

        loss = reprojection_loss(coords_pred, pixels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Query Epoch:", epoch, loss.item())