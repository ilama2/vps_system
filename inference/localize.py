import torch
import numpy as np

from models.encoder import Encoder
from models.regressor import Regressor
from models.diffusion import DiffusionRefiner
from utils.geometry import solve_pnp, create_pixel_grid

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = Encoder().to(device)
regressor = Regressor().to(device)
diffusion = DiffusionRefiner().to(device)

map_code = torch.load("map_code.pth").to(device)
diffusion.load_state_dict(torch.load("diffusion.pth"))

encoder.eval()
regressor.eval()
diffusion.eval()

image = torch.randn(1, 3, 224, 224).to(device)

features = encoder(image)
coords, _ = regressor(features, map_code)

coords = diffusion(coords)

coords = coords.detach().cpu().numpy()
pixels = create_pixel_grid()

pose = solve_pnp(coords, pixels)

print("Pose:", pose)