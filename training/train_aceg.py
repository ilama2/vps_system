from models.encoder import DinoEncoder
from  models.regressor import ACEGRegressor
import torch
encoder = DinoEncoder()
model = ACEGRegressor()
map_code = torch.nn.Parameter(torch.randn(1, 64, 768))

optimizer = torch.optim.Adam(
    list(model.parameters()) + [map_code], lr=1e-4
)

for image, gt_coords in dataloader:

    features = encoder(image)
    coords = model(features, map_code)

    loss = ((coords - gt_coords) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()