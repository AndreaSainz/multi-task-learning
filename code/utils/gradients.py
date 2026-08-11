import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import tomosipo as ts
import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
from code.ct_recon.SIRT import sirt_step
from code.utils.image_cropping import extract_crops, insert_crop_grad_in_image



def nfs_grad(crop_x, digit, mu_model):
    """
    Compute the gradient of the log-density predicted by a normalizing flow
    model with respect to the input crop.

    Parameters
    ----------
        crop_x : torch.Tensor
            Batch of input image crops with shape (B, 1, H, W).

        digit : torch.Tensor
            Tensor containing the digit labels corresponding to each crop.

        mu_model : torch.nn.Module
            Normalizing flow model that predicts the log-density of the input
            crop conditioned on the corresponding digit.

    Returns
    -------
    torch.Tensor
        Gradient of the summed log-density with respect to crop_x, with the
        same shape as crop_x.

    Notes
    -----
    The input crop is detached from the original computation graph before
    computing the gradient. The resulting gradient is also detached from
    the graph before being returned.
    """


    B = crop_x.shape[0]
    
    crop_req = crop_x.detach().clone().requires_grad_(True)

    log_mu = mu_model(crop_req, digit).view(B, 1) 

    F =  log_mu.sum()

    grad_F = torch.autograd.grad(F, crop_req, create_graph=False)[0]

    step_F = grad_F.detach()

    return step_F


def cdm_grad(crop_x, digit, t_input, context_mask, fixed_noise, sqrtab, sqrtmab):
    """
    Compute the score-based gradient of a conditional diffusion model
    with respect to the input crop.

    Parameters
    ----------
        crop_x : torch.Tensor
            Batch of input image crops with shape (B, 1, H, W).

        digit : torch.Tensor
            Tensor containing the digit labels corresponding to each crop.

        t_input : torch.Tensor
            Diffusion timestep input used to condition the score model.

        context_mask : torch.Tensor
            Mask controlling the conditioning information provided to the score
            model.

        fixed_noise : torch.Tensor
            Fixed Gaussian noise used to construct the noisy diffusion sample.

        sqrtab : torch.Tensor
            Square root of the cumulative product of diffusion coefficients,
            indexed by the diffusion timestep.

        sqrtmab : torch.Tensor
            Square root of one minus the cumulative product of diffusion
            coefficients, indexed by the diffusion timestep.

    Returns
    -------
    torch.Tensor
        Score-based gradient tensor with the same shape as crop_x.

    Notes
    -----
    The noisy sample x_t is constructed from crop_x and fixed_noise using
    the diffusion coefficients. The predicted noise is obtained from the
    score model without tracking gradients. The score is then converted
    into a gradient with respect to the crop.
    """
    
    B = crop_x.shape[0]
    x_t = (
        sqrtab[t].view(B, 1, 1, 1) * crop_x
        + sqrtmab[t].view(B, 1, 1, 1) * fixed_noise
    )
    with torch.no_grad():
        eps_hat = score_model(crop_x, digit, t_input, context_mask)
    score = - eps_hat / sqrtmab[t].view(B, 1, 1, 1)

    grad_crop = - score

    return grad_crop


def bayes_grad(crop_x, digit, mode, i, batch_id, gamma, class_model, den_model):
    """
    Compute a Bayesian gradient based on classification and/or denoising
    objectives.

    Parameters
    ----------
        crop_x : torch.Tensor
            Batch of input image crops with shape (B, 1, H, W).

        digit : torch.Tensor
            Tensor containing the target digit labels corresponding to each crop.

        mode : str
            Determines which gradient components are included.

            - "both" : use both the classification and denoising gradients.
            - "class" : use only the classification gradient.
            - "deno" : use only the denoising gradient.

        i : int
            Current iteration index.

        batch_id : int
            Index identifying the current batch.

        gamma : torch.Tensor or array-like
            Weighting coefficients applied to the classification and denoising
            gradients. gamma[0] weights the classification component and
            gamma[1] weights the denoising component.

        class_model : torch.nn.Module
            Classification model used to compute the gradient of the log
            probability assigned to the target digit.

        den_model : torch.nn.Module
            Denoising model used to compute the denoising gradient.

    Returns
    -------
    torch.Tensor
        Combined gradient tensor with the same shape as crop_x.

    Notes
    -----
    The classification gradient is computed from the log-probability of the
    target digit after applying a log-softmax to the classifier logits.

    The denoising gradient is defined as the difference between the
    denoising model output and the input crop.

    The selected gradient components are combined using the coefficients
    specified by gamma and negated before being returned.
    """


    grad_classi = 0.0
    grad_deno = 0.0

    if mode in ["both", "class"]:
        logits = class_model(crop_x)
        log_probs = torch.log_softmax(logits, dim=1)
        classi = log_probs.gather(1, digit.unsqueeze(1)).squeeze(1)
        grad_classi = torch.autograd.grad(classi.sum(), crop_x, create_graph=False)[0]

    if mode in ["both", "deno"]:
        deno = den_model(crop_x)
        grad_deno = deno - crop_x


    if mode == "both":
        grad_crop = - (gamma[0] * grad_classi + gamma[1] * grad_deno)
    elif mode == "class":
        grad_crop = - (gamma[0] * grad_classi)
    elif mode == "deno":
        grad_crop = - (gamma[1] * grad_deno)

    return grad_crop



def update_case(x, y, A_torch, AT_torch, centre, digit, alpha, gamma, sigma_data, mode, if_sirt, i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters):
    """
    Perform one reconstruction update consisting of an optional SIRT step
    followed by a regularization gradient step.

    Parameters
    ----------
        x : torch.Tensor
            Batch of current image reconstructions with shape (B, 1, H, W).

        y : torch.Tensor
            Batch of measured projection data used by the SIRT reconstruction
            step.

        A_torch : torch.Tensor
            Forward projection operator used to compute the simulated projection
            data from the current reconstruction.

        AT_torch : torch.Tensor
            Adjoint or backprojection operator used in the SIRT update.

        centre : torch.Tensor
            Tensor containing the center coordinates of each image with shape
            (B, 2). For each sample, centre[b, 0] corresponds to the vertical
            (y) coordinate and centre[b, 1] corresponds to the horizontal
            (x) coordinate.

        digit : torch.Tensor
            Tensor containing the digit labels corresponding to each image crop.

        alpha : float
            Step size used for the SIRT update.

        gamma : float or torch.Tensor
            Scaling factor applied to the regularization gradient. For the
            Bayesian regularizer, gamma contains the weights of the classification
            and denoising gradient components.

        sigma_data : float
            Data noise scale used to scale the regularization gradient.

        mode : str
            Specifies which reconstruction update is performed.

            - "none" : perform only the SIRT update when if_sirt is True.
            - "both" : use the complete regularization gradient.
            - "class" : use only the classification gradient for the Bayesian
            regularizer.
            - "deno" : use only the denoising gradient for the Bayesian
            regularizer.

        if_sirt : bool
            If True, perform a SIRT update before computing the regularization
            gradient. If False, only the regularization gradient is applied.

        i : int
            Current outer iteration index.

        r : int
            Current inner reconstruction or regularization step index.

        batch_id : int
            Index identifying the current batch.

        regularizer : str
            Type of regularization applied after the SIRT step.

            - "nf" : use the normalizing flow gradient.
            - "cdm" : use the conditional diffusion model gradient.
            - "bayes" : use the Bayesian classification and/or denoising
            gradient.

        results_dir : str
            Directory where gradient visualizations are saved when verbose
            gradient logging is enabled.

        name : str
            Name used to identify the saved gradient visualization files.

        verbose_gradient : bool
            If True, log the gradient norm and periodically save a visualization
            of the crop gradient.

        **parameters : dict
            Additional parameters required by the selected regularizer.

            For "nf", the dictionary must contain:

            - "model" : normalizing flow model used to compute the gradient.

            For "cdm", the dictionary must contain:

            - "score_model" : conditional diffusion score model.
            - "t_input" : diffusion timestep input.
            - "context_mask" : mask controlling the conditioning information.
            - "fixed_noise" : fixed noise used to construct the noisy diffusion
            sample.
            - "sqrtab" : square root of the cumulative product of diffusion
            coefficients.
            - "sqrtmab" : square root of one minus the cumulative product of
            diffusion coefficients.

            For "bayes", the dictionary must contain:

            - "class_model" : classification model used to compute the
            classification gradient.
            - "den_model" : denoising model used to compute the denoising
            gradient.

    Returns
    -------
    torch.Tensor
        Updated batch of image reconstructions with the same shape as x.
        The values are clamped to the interval [0, 1].

    torch.Tensor or str
        Information returned by the SIRT step. If if_sirt is True, this is
        the step_data returned by sirt_step. If if_sirt is False, the value
        is the string "None".

    Notes
    -----
    If if_sirt is True, the function first performs a SIRT reconstruction
    step. If mode is "none", the SIRT-updated reconstruction is returned
    without applying a regularization gradient.

    Otherwise, image crops centered at centre are extracted from the current
    reconstruction. The corresponding regularization gradient is computed
    on the crops and inserted back into the full-size image before applying
    the gradient descent update.

    The regularization gradient is scaled by sigma_data**2 and, where
    applicable, by gamma. For the Bayesian regularizer, mode determines
    whether the classification gradient, denoising gradient, or both are
    used.

    Gradient logging and visualization are optionally performed when
    verbose_gradient is True.
    """

    # SIRT STEP
    if if_sirt:
        x, step_data = sirt_step(A_torch, AT_torch, x, y, alpha)
        # case where only SIRT
        if mode == "none":
            return x.clamp(0, 1), step_data


    # extracting crops
    crop_x = extract_crops(x, centre).detach().requires_grad_(True)

    # calculating the gradient of the crop image
    if regularizer == "nf":
        model = parameters["model"]
        grad_crop = gamma * (sigma_data**2) * nfs_grad(crop_x, digit, model)
    
    elif regularizer == "cdm":
        score_model = parameters["score_model"]
        t_input = parameters["t_input"]
        context_mask = parameters["context_mask"]
        fixed_noise = parameters["fixed_noise"]
        sqrtab = parameters["sqrtab"]
        sqrtmab = parameters["sqrtmab"]
        grad_crop = gamma * (sigma_data**2) * cdm_grad(crop_x, digit, t_input, context_mask, fixed_noise, sqrtab, sqrtmab)

    elif regularizer == "bayes":
        class_model = parameters["class_model"]
        den_model = parameters["den_model"]
        grad_crop = (sigma_data**2) * bayes_grad(crop_x, digit, mode, gamma, class_model, den_model)

    

    # debugging gradient
    if verbose_gradient and accelerator.is_main_process:
        logger.info(f"Gradient descent step {batch_id}_{i}_{r}: {-grad_crop.flatten(1).norm(dim=1).mean().item()}")

    if  verbose_gradient and batch_id == 0 and i % 2 == 0 and r==0: 
        plt.imshow(grad_crop[0,0].detach().cpu().numpy(), cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_grad_crop_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()


    # inserting the crop into the complete gradient
    grad_F =  insert_crop_grad_in_image(x, grad_crop, centre, None)
    
    # gradient step
    x = (x - grad_F).clamp(0, 1)

    if if_sirt:
        return x, step_data
    else:
        return x, "None"








def score_centers_difussion_model(
    model,
    img,
    *,
    denoiser=None,
    H=100,
    W=100,
    patch=28,
    ymin=14,
    ymax=85,
    xmin=14,
    xmax=85,
    chunk_size=512,
    return_digit_map=False,
):
    """
    Para cada centro candidato (y,x), extrae el crop 28x28 y calcula:

        score(y,x) = max_c p(c | crop_{y,x})

    donde p(c|crop) viene del clasificador.

    Args:
        model: clasificador que devuelve logits [B,10]
        img: [B,1,H,W]
        denoiser: opcional. Si se da, se aplica al crop antes del clasificador
        H, W: tamaño de la imagen
        patch: tamaño del crop (28)
        ymin, ymax, xmin, xmax: rango de centros válidos
        chunk_size: número de crops procesados por chunk
        return_digit_map: si True, devuelve también el mapa del dígito predicho

    Returns:
        map_x: [B,H,W] con score = max softmax prob en centros válidos y -inf fuera
        digit_map (opcional): [B,H,W] con dígito predicho en centros válidos y -1 fuera
    """

    device = img.device
    B, C, Hx, Wx = img.shape
    assert C == 1
    assert Hx == H and Wx == W
    assert patch % 2 == 0

    rad = patch // 2

    # Todos los patches posibles
    patches = img.unfold(2, patch, 1).unfold(3, patch, 1)
    # [B,1,H-patch+1,W-patch+1,patch,patch]

    ty_min, ty_max = ymin - rad, ymax - rad
    tx_min, tx_max = xmin - rad, xmax - rad

    # Solo la zona válida de centros
    patches = patches[:, :, ty_min:ty_max+1, tx_min:tx_max+1, :, :]
    # [B,1,gh,gw,patch,patch]

    gh, gw = patches.shape[2], patches.shape[3]
    N = gh * gw

    # [B*N,1,patch,patch]
    patches_flat = patches.contiguous().view(B * N, 1, patch, patch)

    scores_flat = torch.empty(B * N, device=device, dtype=torch.float32)

    if return_digit_map:
        digits_flat = torch.empty(B * N, device=device, dtype=torch.long)

    for start in range(0, B * N, chunk_size):
        end = min(start + chunk_size, B * N)
        p = patches_flat[start:end]

        if denoiser is not None:
            p = denoiser(p).clamp(0.0, 1.0)

        logits = model(p)                 # [chunk, 10]
        probs = torch.softmax(logits, dim=1)

        conf, pred = probs.max(dim=1)     # [chunk], [chunk]

        scores_flat[start:end] = conf.float()

        if return_digit_map:
            digits_flat[start:end] = pred

    scores_grid = scores_flat.view(B, gh, gw)

    map_x = torch.full((B, H, W), -torch.inf, device=device, dtype=torch.float32)
    map_x[:, ymin:ymax+1, xmin:xmax+1] = scores_grid

    if return_digit_map:
        digits_grid = digits_flat.view(B, gh, gw)
        digit_map = torch.full((B, H, W), -1, device=device, dtype=torch.long)
        digit_map[:, ymin:ymax+1, xmin:xmax+1] = digits_grid
        return map_x, digit_map

    return map_x


