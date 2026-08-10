import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def get_image(data, sample):
    """
    Extract a single image from an input tensor, handling common image
    and batch formats.

    Parameters
    ----------
    data : torch.Tensor
        Input image tensor.

    sample : int
        Index of the sample to extract when a batch dimension is present.

    Returns
    -------
    numpy.ndarray
        Image ready to be passed to ``imshow``.
    """

    # Remove dimensions of size 1 only when they are unambiguous
    if data.ndim == 2:
        # [H, W]
        image = data

    elif data.ndim == 3:
        if data.shape[0] == 1:
            # [1, H, W] -> single-channel image
            image = data[0]

        elif data.shape[-1] == 1:
            # [H, W, 1] -> single-channel image
            image = data[..., 0]

        elif data.shape[0] in (3, 4):
            # [C, H, W] -> RGB/RGBA image
            image = data.permute(1, 2, 0)

        elif data.shape[-1] in (3, 4):
            # [H, W, C] -> RGB/RGBA image
            image = data

        else:
            # [B, H, W] -> batch of grayscale images
            image = data[sample]

    elif data.ndim == 4:
        if data.shape[1] in (1, 3, 4):
            # [B, C, H, W]
            image = data[sample]

            if image.shape[0] == 1:
                image = image[0]
            else:
                image = image.permute(1, 2, 0)

        elif data.shape[-1] in (1, 3, 4):
            # [B, H, W, C]
            image = data[sample]

            if image.shape[-1] == 1:
                image = image[..., 0]

        else:
            raise ValueError(
                f"Unsupported 4D image shape: {tuple(data.shape)}"
            )

    else:
        raise ValueError(
            f"Unsupported image shape: {tuple(data.shape)}"
        )

    return image.detach().cpu().numpy()


def images_plots(results_dir, name, batch_id, sample, images, iteration=-1):
    """
    Generate and save a comparison figure showing the selected sample from
    multiple image collections using a common grayscale color scale.

    Parameters
    ----------
        results_dir : str
            Directory where the generated figure is saved.

        name : str
            Name used to identify the comparison in the output filename.

        batch_id : int or str
            Identifier of the batch, used in the output filename.

        sample : int
            Index of the sample to visualize when a batch dimension is present.

        images : dict
            Dictionary containing the image collections to compare. Values can
            either be image tensors or lists of image tensors.

        iteration : int, optional
            Index of the reconstruction to visualize when an image collection
            is provided as a list. Defaults to -1, corresponding to the last
            reconstruction.

    Saves
    -----
    A PNG file is saved in results_dir with the following format:

    {batch_id}_comparison_{name}.png
    """

    n = len(images)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))

    if n == 1:
        axes = [axes]

    all_imgs = []

    for image_name, image_collection in images.items():

        if isinstance(image_collection, list):
            data = image_collection[iteration]
        else:
            data = image_collection

        #if batch_size is bigger than 1 then I can get sample>1, otherwise there is only one sample in the batch
        if data.shape[0] > 1:
            image = get_image(data, sample)
        else:
            image = get_image(data, 0)
        all_imgs.append(image)

    vmin = min(img.min() for img in all_imgs)
    vmax = max(img.max() for img in all_imgs)

    for i, (image_name, image_collection) in enumerate(images.items()):

        if isinstance(image_collection, list):
            data = image_collection[iteration]
        else:
            data = image_collection

        image = get_image(data, sample)

        im = axes[i].imshow(image,cmap="gray",vmin=vmin,vmax=vmax)

        axes[i].set_title(image_name)
        axes[i].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    plt.savefig(os.path.join(results_dir,f"{batch_id}_comparison_{name}.png"),dpi=300,bbox_inches="tight")

    plt.close()


def psnr_plots(results_dir, name, batch_id, psnr_history):

    """
    Generate and save PSNR curves for SCORE and SIRT reconstructions.

    Two plots are generated:
    1. PSNR of the full images as a function of the iteration parameter I.
    2. PSNR of the cropped images as a function of I.

    The PSNR values are averaged across the batch for each iteration.

    Parameters
    ----------
        results_dir : str
            Directory where the generated plots will be saved.

        name : str
            Name used to identify the experiment in the output filenames.

        batch_id : int or str
            Identifier of the batch, used in the output filenames.

        psnr_history : list of dict
            List containing the PSNR values for each iteration. Each dictionary
            must contain the following keys:

            - I : Iteration value.
            - psnr_x: PSNR values for the SCORE reconstruction.
            - psnr_x_sirt : PSNR values for the SIRT reconstruction.
            - psnr_x_crop : PSNR values for the cropped SCORE reconstruction.
            - psnr_x_sirt_crop: PSNR values for the cropped SIRT reconstruction.

    Saves
    -----
    Two PNG files are saved in results_dir:

        - {batch_id}_psnr_curve_{name}.png
            PSNR curves for the full images.

        - {batch_id}_psnr_crop_curve_{name}.png
            PSNR curves for the cropped images.
    """


    I_vals = [d["I"] for d in psnr_history]
    psnr_x_vals = [d["psnr_x"].mean().item() for d in psnr_history]
    psnr_x_sirt_vals = [d["psnr_x_sirt"].mean().item() for d in psnr_history]

    plt.figure()
    plt.plot(I_vals, psnr_x_vals, marker="o", label="Mean SCORE")
    plt.plot(I_vals, psnr_x_sirt_vals, marker="o", label="Mean SIRT")
    plt.xlabel("I")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_psnr_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()


    psnr_x_crop_vals = [d["psnr_x_crop"].mean().item() for d in psnr_history]
    psnr_x_sirt_crop_vals = [d["psnr_x_sirt_crop"].mean().item() for d in psnr_history]

    plt.figure()
    plt.plot(I_vals, psnr_x_sirt_crop_vals, marker="o", label="Mean SIRT")
    plt.plot(I_vals, psnr_x_crop_vals, marker="o", label="Mean SCORE")
    plt.xlabel("I")
    plt.ylabel("PSNR crop (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_psnr_crop_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()





def ssim_plots(results_dir, name, batch_id, ssim_history):
    """
    Generate and save SSIM curves for SCORE and SIRT reconstructions.

    Two plots are generated:
    1. SSIM of the full images as a function of the iteration parameter I.
    2. SSIM of the cropped images as a function of I.

    The SSIM values are averaged across the batch for each iteration.

    Parameters
    ----------
        results_dir : str
            Directory where the generated plots will be saved.

        name : str
            Name used to identify the experiment in the output filenames.

        batch_id : int or str
            Identifier of the batch, used in the output filenames.

        ssim_history : list of dict
            List containing the SSIM values for each iteration. Each dictionary
            must contain the following keys:

            - "I" : Iteration value.
            - "ssim_x" : SSIM values for the SCORE reconstruction.
            - "ssim_x_sirt" : SSIM values for the SIRT reconstruction.
            - "ssim_x_crop" : SSIM values for the cropped SCORE reconstruction.
            - "ssim_x_sirt_crop" : SSIM values for the cropped SIRT reconstruction.

    Saves
    -----
    Two PNG files are saved in results_dir:

    - {batch_id}_ssim_curve_{name}.png
        SSIM curves for the full images.

    - {batch_id}_ssim_crop_curve_{name}.png
        SSIM curves for the cropped images.
    """


    I_vals = [d["I"] for d in ssim_history]
    ssim_x_vals = [d["ssim_x"].mean().item() for d in ssim_history]
    ssim_x_sirt_vals = [d["ssim_x_sirt"].mean().item() for d in ssim_history]

    plt.figure()
    plt.plot(I_vals, ssim_x_vals, marker="o", label="Mean SCORE")
    plt.plot(I_vals, ssim_x_sirt_vals, marker="o", label="Mean SIRT")
    plt.xlabel("I")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_ssim_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

   
    ssim_x_crop_vals = [d["ssim_x_crop"].mean().item() for d in ssim_history]
    ssim_x_sirt_crop_vals = [d["ssim_x_sirt_crop"].mean().item() for d in ssim_history]

    plt.figure()
    plt.plot(I_vals, ssim_x_sirt_crop_vals, marker="o", label="Mean SIRT")
    plt.plot(I_vals, ssim_x_crop_vals, marker="o", label="Mean SCORE")
    plt.xlabel("I")
    plt.ylabel("SSIM crop ")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_ssim_crop_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()



def digit_plots(results_dir, name, batch_id, digit_acc_history):
    """
    Generate and save digit classification metrics for SCORE and SIRT
    reconstructions.

    Three plots are generated:
    1. Probability assigned to the true digit as a function of I.
    2. Negative log-likelihood (NLL) of the true digit as a function of I.
    3. Entropy of the digit prediction as a function of I.

    The metric values are averaged across the batch for each iteration.

    Parameters
    ----------
        results_dir : str
            Directory where the generated plots will be saved.

        name : str
            Name used to identify the experiment in the output filenames.

        batch_id : int or str
            Identifier of the batch, used in the output filenames.

        digit_acc_history : list of dict
            List containing the digit classification metrics for each iteration.
            Each dictionary must contain the following keys:

            - "I" : Iteration value.
            - "digit_prob_true" : Probability assigned to the true digit for
            the SCORE reconstruction.
            - "digit_prob_true_sirt" : Probability assigned to the true digit
            for the SIRT reconstruction.
            - "digit_nll" : Negative log-likelihood of the true digit for
            the SCORE reconstruction.
            - "digit_nll_sirt" : Negative log-likelihood of the true digit
            for the SIRT reconstruction.
            - "digit_entropy" : Prediction entropy for the SCORE reconstruction.
            - "digit_entropy_sirt" : Prediction entropy for the SIRT
            reconstruction.

    Saves
    -----
    Three PNG files are saved in results_dir:

    - {batch_id}_digit_prob_curve_{name}.png
        True-digit probability curves for SCORE and SIRT.

    - {batch_id}_digit_nll_curve_{name}.png
        True-digit negative log-likelihood curves for SCORE and SIRT.

    - {batch_id}_digit_entropy_curve_{name}.png
        Digit prediction entropy curves for SCORE and SIRT.
    """

    I_vals = [d["I"] for d in digit_acc_history]
    digit_prob_vals = [d["digit_prob_true"].mean().item() for d in digit_acc_history]
    digit_prob_vals_sirt = [d["digit_prob_true_sirt"].mean().item() for d in digit_acc_history]

    plt.figure()
    plt.plot(I_vals, digit_prob_vals, marker="o", label="Mean Digit prob (SCORE)")
    plt.plot(I_vals, digit_prob_vals_sirt, marker="o", label="Mean Digit prob (SIRT)")
    plt.xlabel("I")
    plt.ylabel("Probability")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_digit_prob_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    digit_nll_vals = [d["digit_nll"].mean().item() for d in digit_acc_history]
    digit_nll_vals_sirt = [d["digit_nll_sirt"].mean().item() for d in digit_acc_history]

    plt.figure()
    plt.plot(I_vals, digit_nll_vals, marker="o", label="Mean Digit NLL (SCORE)")
    plt.plot(I_vals, digit_nll_vals_sirt, marker="o", label="Mean Digit NLL (SIRT)")
    plt.xlabel("I")
    plt.ylabel("-log p(true)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_digit_nll_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    digit_margin_vals = [d["digit_entropy"].mean().item() for d in digit_acc_history]
    digit_margin_vals_sirt = [d["digit_entropy_sirt"].mean().item() for d in digit_acc_history]

    plt.figure()
    plt.plot(I_vals, digit_margin_vals, marker="o", label="Mean Digit entropy (SCORE)")
    plt.plot(I_vals, digit_margin_vals_sirt, marker="o", label="Mean Digit entropy (SIRT)")
    plt.xlabel("I")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_digit_entropy_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()



def centre_plots(results_dir, name, batch_id, centre_mse_history):
    """
    Generate and save Wasserstein-1 distance curves for SCORE and SIRT
    reconstructions.

    The plot shows the mean Centre W1 distance as a function of the iteration
    parameter I. The distance is measured in pixels and averaged across
    the batch for each iteration.

    Parameters
    ----------
        results_dir : str
            Directory where the generated plot will be saved.

        name : str
            Name used to identify the experiment in the output filename.

        batch_id : int or str
            Identifier of the batch, used in the output filename.

        centre_mse_history : list of dict
            List containing the Centre W1 values for each iteration. Each
            dictionary must contain the following keys:

            - "I" : Iteration value.
            - "centre_w1" : Centre W1 distance for the SCORE reconstruction.
            - "centre_w1_sirt" : Centre W1 distance for the SIRT reconstruction.

    Saves
    -----
    A PNG file is saved in results_dir:

    {batch_id}_centre_w1_curve_{name}.png
        Centre W1 distance curves for SCORE and SIRT.
    """


    I_vals = [d["I"] for d in centre_mse_history]
    centre_w1_vals = [d["centre_w1"].mean().item() for d in centre_mse_history]
    centre_w1_vals_sirt = [d["centre_w1_sirt"].mean().item() for d in centre_mse_history]

    plt.figure()
    plt.plot(I_vals, centre_w1_vals, marker="o", label="Mean Centre W1 (SCORE)")
    plt.plot(I_vals, centre_w1_vals_sirt, marker="o", label="Mean Centre W1 (SIRT)")
    plt.xlabel("I")
    plt.ylabel("Wasserstein-1 (pixels)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_centre_w1_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()