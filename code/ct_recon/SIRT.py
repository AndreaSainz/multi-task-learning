import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts


def normalization_matrices(A_torch, AT_torch, B, H, W, angles, device, num_extra_dims=2):
    eps = 1e-8

    # create the input tensor for A and AT
    if num_extra_dims == 1:
        image = torch.ones((B, H, W), device=device)
        sinogram = torch.ones((B, angles, H), device=device)
    elif num_extra_dims == 2:
        image = torch.ones((B, 1, H, W), device=device)
        sinogram = torch.ones((B, 1, angles, H), device=device)
    else:
        raise ValueError(f"Unsupported num_extra_dims: {num_extra_dims}")


    row_sums = A_torch(image)
    col_sums = AT_torch(sinogram)

    M = 1.0 / torch.clamp(row_sums, min=eps)  # measurements
    D = 1.0 / torch.clamp(col_sums, min=eps)  # image

    return M , D


def sirt_method(A_torch, AT_torch, x, y, D, M, alpha):
    Ax = A_torch(x)
    r = (y - Ax) * M
    AT_r = AT_torch(r) 
    update = AT_r * D
    step_data = alpha * update
    x_new = x + step_data
    return x_new, step_data


def sirt_step(A_torch, AT_torch, x, y, alpha):
    Ax = A_torch(x)
    r = y - Ax
    AT_r = AT_torch(r)
    step_data = alpha * AT_r
    x_new = x + step_data
    return x_new, step_data