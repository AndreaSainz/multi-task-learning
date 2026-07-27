import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import tomosipo as ts


def extract_crops(x, centre):
    B = x.shape[0]
    crops = torch.zeros(B, 1, 28, 28, device=x.device, dtype=x.dtype)
    for b in range(B):
        cy = int(centre[b, 0].item())
        cx = int(centre[b, 1].item())
        crops[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
    return crops


def insert_crop_grad_in_image(x, grad_crop, centre, fill_mode="scaled_x"):
    grad_F = torch.zeros_like(x)

    if fill_mode == "scaled_x":
        x_scaled = x.clone()
        scale = grad_crop.norm() / (x_scaled.norm() + 1e-8)
        grad_F = 5 * x_scaled * scale

    for b in range(x.shape[0]):
        cy = int(centre[b, 0].item())
        cx = int(centre[b, 1].item())
        grad_F[b:b+1, :, cy-14:cy+14, cx-14:cx+14] = grad_crop[b:b+1]

    return grad_F
