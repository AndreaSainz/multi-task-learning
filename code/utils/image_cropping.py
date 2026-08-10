import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import tomosipo as ts
import os

def extract_crops(x, centre):
    """
    Extract a 28x28 crop centered at the specified coordinates for each image
    in a batch.

    Parameters
    ----------
        x : torch.Tensor
            Batch of input images with shape ``(B, 1, H, W)``.

        centre : torch.Tensor
            Tensor containing the center coordinates of each image with shape
            (B, 2). For each sample, centre[b, 0  corresponds to the
            vertical (y) coordinate and centre[b, 1  corresponds to the
            horizontal (x) coordinate.

    Returns
    -------
    torch.Tensor
        Batch of extracted crops with shape (B, 1, 28, 28).

    Notes
    -----
    The crop is centered at the provided coordinates and extends 14 pixels
    in each direction, resulting in a 28x28 image.
    """

    B = x.shape[0]
    crops = torch.zeros(B, 1, 28, 28, device=x.device, dtype=x.dtype)
    for b in range(B):
        cy = int(centre[b, 0].item())
        cx = int(centre[b, 1].item())
        crops[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
    return crops


def insert_crop(x, crop, centre):
    """
    Insert a batch of image crops into the corresponding positions of a
    batch of full-size images.

    Parameters
    ----------
        x : torch.Tensor
            Batch of full-size images with shape (B, 1, H, W).

        crop : torch.Tensor
            Batch of image crops with shape (B, 1, 28, 28). Each crop is inserted
            into the corresponding image in x.

        centre : torch.Tensor
            Tensor containing the center coordinates of each crop with shape
            (B, 2). For each sample, centre[b, 0] corresponds to the vertical
            (y) coordinate and centre[b, 1] corresponds to the horizontal
            (x) coordinate.

    Returns
    -------
    torch.Tensor
        Batch of images with the corresponding crops inserted at the
        positions specified by centre. The returned tensor has the same
        shape as x.

    Notes
    -----
    For each sample, the crop is inserted into a 28 x 28 region centred at
    the coordinates specified by centre.
    """

    B = x.shape[0]

    for b in range(B):
        cy = int(centre[b, 0].item())
        cx = int(centre[b, 1].item())
        x[b:b+1, :, cy-14:cy+14, cx-14:cx+14] = crop
    
    return x

def insert_crop_grad_in_image(x, grad_crop, centre, scale_param, fill_mode="none"):
    """
    Insert a crop gradient into the corresponding position of each image
    in a batch.

    Parameters
    ----------
    x : torch.Tensor
        Batch of input images with shape (B, 1, H, W).

    grad_crop : torch.Tensor
        Batch of gradients corresponding to the extracted image crops, with
        shape (B, 1, 28, 28).

    centre : torch.Tensor
        Tensor containing the center coordinates of each image with shape
        (B, 2). For each sample, centre[b, 0] corresponds to the
        vertical (y) coordinate and centre[b, 1] corresponds to the
        horizontal (x) coordinate.

    scale_param : float
        Scaling factor applied to the gradient when fill_mode is
        "scaled_x".

    fill_mode : str, optional
        Determines how the gradient outside the crop is initialized.

        - "none" : initialize the gradient to zero outside the crop.
        -  "scaled_x"  : initialize the gradient using a scaled version
          of the input image  x .

        Default is "none" .

    Returns
    -------
    torch.Tensor
        Full-size gradient tensor with the same shape as x. The
        corresponding grad_crop is inserted at the position specified
        by centre for each image.

    Notes
    -----
    When fill_mode="scaled_x", the input image is scaled according to
    the ratio between the norm of grad_crop and the norm of x before
    applying scale_param.
    """

    grad_F = torch.zeros_like(x)

    if fill_mode == "scaled_x":
        x_scaled = x.clone()
        scale = grad_crop.norm() / (x_scaled.norm() + 1e-8)
        grad_F = scale_param * x_scaled * scale

    for b in range(x.shape[0]):
        cy = int(centre[b, 0].item())
        cx = int(centre[b, 1].item())
        grad_F[b:b+1, :, cy-14:cy+14, cx-14:cx+14] = grad_crop[b:b+1]

    return grad_F

