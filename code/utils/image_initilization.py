import random
from code.utils.image_cropping import extract_crops, insert_crop
import torch



def true_images_lines(x_true, centre_true, H, W, noise, noise_range, central_lines, n_lines):
    """
    Generate modified images by extracting the digit crop, optionally adding
    noise, and masking a specified number of vertical lines.

    Parameters
    ----------
        x_true : torch.Tensor
            Batch of ground-truth images with shape (B, 1, H, W).

        centre_true : torch.Tensor
            Tensor containing the center coordinates of each digit with shape
            (B, 2). For each sample, centre_true[b, 0] corresponds to the
            vertical (y) coordinate and centre_true[b, 1] corresponds to the
            horizontal (x) coordinate.

        H : int
            Height of the output images.

        W : int
            Width of the output images.

        noise : bool
            If True, add Gaussian noise to the extracted digit crops.

        noise_range : tuple or list of float
            Minimum and maximum values used to randomly sample the standard
            deviation of the Gaussian noise.

        central_lines : bool
            Determines whether the masked vertical lines are selected from the
            centre of the crop or randomly.

            - True : mask n_lines consecutive central vertical lines.
            - False : mask n_lines randomly selected vertical lines.

        n_lines : int
            Number of vertical lines to mask in the 28 x 28 digit crop.

    Returns
    -------
    torch.Tensor
        Batch of modified images with shape (B, 1, H, W). Each digit crop is
        inserted at its original centre after noise addition and line
        masking.

    Notes
    -----
    The digit crops are extracted using extract_crops and inserted back into
    the full-size images using insert_crop.

    When central_lines is True, the masked lines form a contiguous band
    centred on the middle of the crop. When central_lines is False, the
    masked lines are selected randomly without replacement.
    """


    #batch size
    B = x_true.shape[0]

    #initilizing image black
    x = torch.zeros(B, 1, H, W, device=x_true.device)

    #extracting the MNIST digit
    crop = extract_crops(x_true, centre_true)
    
    # Adding optional noise to the crop
    if noise: 
        #random sigma between noisy range
        sigma = torch.empty(1, device=crop.device).uniform_(noise_range[0], noise_range[1])
        crop = crop + torch.randn_like(crop) * sigma

    # substracting a band from the MNIST digit
    if central_lines:
        # n_lines central lines
        start = (28 - n_lines) // 2
        end = start + n_lines
        crop[:, :, :, start:end] = 0.0
    else:
        # n_lines random lines
        lines = sorted(random.sample(range(28), n_lines))
        crop[:, :, :, lines] = 0.0

    x = insert_crop(x, crop, centre_true)
    return x




def true_images_pixels(x_true, centre_true, H, W, noise, noise_range, percentage_pixels):
    """
    Generate modified images by extracting the digit crop, optionally adding
    Gaussian noise, and randomly masking a specified percentage of pixels
    independently for each sample in the batch.

    Parameters
    ----------
        x_true : torch.Tensor
            Batch of ground-truth images with shape (B, 1, H, W).

        centre_true : torch.Tensor
            Tensor containing the center coordinates of each digit with shape
            (B, 2). For each sample, centre_true[b, 0] corresponds to the
            vertical (y) coordinate and centre_true[b, 1] corresponds to the
            horizontal (x) coordinate.

        H : int
            Height of the output images.

        W : int
            Width of the output images.

        noise : bool
            If True, add Gaussian noise to the extracted digit crops.

        noise_range : tuple or list of float
            Minimum and maximum values used to randomly sample the standard
            deviation of the Gaussian noise.

        percentage_pixels : float
            Percentage of pixels in the 28 x 28 digit crop to randomly mask.
            Values between 0 and 1 are interpreted as fractions, while values
            greater than 1 are interpreted as percentages. For example, 0.1 and
            10 both correspond to masking 10% of the pixels.

    Returns
    -------
    torch.Tensor
        Batch of modified images with shape (B, 1, H, W). Each digit crop
        is inserted at its original centre after optional noise addition
        and random pixel masking.

    Notes
    -----
    For each sample in the batch, a different set of pixels is selected
    randomly without replacement and set to zero. The number of masked
    pixels is the same for all samples and is determined by
    percentage_pixels.

    The digit crops are extracted using extract_crops and inserted back
    into the full-size images using insert_crop.
    """

    if percentage_pixels > 1:
        percentage_pixels = percentage_pixels/100
    
    #batch size
    B = x_true.shape[0]

    #initilizing image black
    x = torch.zeros(B, 1, H, W, device=x_true.device)

    #extracting the MNIST digit
    crop = extract_crops(x_true, centre_true)
    
    # Adding optional noise to the crop
    if noise: 
        #random sigma between noisy range
        sigma = torch.empty(1, device=crop.device).uniform_(noise_range[0], noise_range[1])
        crop = crop + torch.randn_like(crop) * sigma


    # Number of pixels to mask
    true_pixels = 28 * 28
    n_pixels = int(percentage_pixels * true_pixels)

    # Flatten the crop
    crop_flat = crop.reshape(B, 1, -1)

    # Select different random pixels for each sample
    for b in range(B):
        pixels = random.sample(range(true_pixels), n_pixels)
        crop_flat[b, 0, pixels] = 0.0

    # Convert back to 28 x 28
    crop = crop_flat.reshape(B, 1, 28, 28)

    x = insert_crop(x, crop, centre_true)
    return x
