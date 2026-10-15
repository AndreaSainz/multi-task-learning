import torch
import torch.nn.functional as F
import numpy as np
from code.utils.gradients import update_case
from code.utils.image_cropping import extract_crops
from code.utils.metrics import compute_psnr_results, compute_ssim, compute_mse
from code.ct_recon.SIRT import sirt_step
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import os



    
def inverse_gradient_only_pipeline(y, x, A_torch, AT_torch, alpha, gamma, sigma_data, R_ini, R_min, R_max, I, decay_iters, H, W, x_true, centre_true, digit_true, regularizer, batch_id, results_dir, name, accelerator, verbose_images, verbose_metrics, verbose_convergence, verbose_gradient, logger, **parameters):
    """
    Perform iterative image reconstruction using SIRT initialization followed
    by gradient-based regularization.

    The pipeline first initializes the reconstruction with ``R_ini`` SIRT
    iterations. It then performs ``I`` outer iterations, each containing a
    number of gradient updates controlled by ``R``. Depending on the selected
    regularizer, the reconstruction can be updated using a normalizing flow,
    conditional diffusion model, or Bayesian classification and denoising
    gradients.

    Parameters
    ----------
        y : torch.Tensor
            Measured projection data.

        x : torch.Tensor
            Initial batch of image reconstructions with shape
            (B, 1, H, W).

        A_torch : torch.Tensor
            Forward projection operator used by the SIRT reconstruction.

        AT_torch : torch.Tensor
            Adjoint or backprojection operator used by the SIRT reconstruction.

        alpha : float
            Step size used in the SIRT updates.

        gamma : float or torch.Tensor
            Scaling factor for the regularization gradient. For the Bayesian
            regularizer, gamma contains the weights for the classification and
            denoising gradient components.

        sigma_data : float
            Data noise scale used to scale the regularization gradient.

        R_ini : int
            Number of SIRT iterations used to initialize the reconstruction.

        R_min : int
            Minimum number of inner gradient iterations per outer iteration.

        R_max : int
            Maximum number of inner gradient iterations per outer iteration.

        I : int
            Number of outer reconstruction iterations.

        decay_iters : int
            Number of outer iterations over which the number of inner steps
            decreases linearly from R_max to R_min.

        H : int
            Height of the full reconstruction image.

        W : int
            Width of the full reconstruction image.

        x_true : torch.Tensor
            Ground-truth full-size images used only for evaluation metrics.

        centre_true : torch.Tensor
            Ground-truth centre coordinates of the digit in each image, with
            shape (B, 2). These coordinates are used to extract the image crops
            for evaluation and to define the reconstruction region.

        digit_true : torch.Tensor
            Ground-truth digit labels used to condition the regularizers and
            evaluate the reconstruction.

        regularizer : str
            Type of regularization used in the gradient updates.

            - ``"nf"`` : normalizing flow regularizer.
            - ``"cdm"`` : conditional diffusion model regularizer.
            - ``"bayes"`` : Bayesian classification and denoising regularizer.

        batch_id : int
            Index identifying the current batch.

        results_dir : str
            Directory where diagnostic plots and other results are saved.

        name : str
            Name used to identify the experiment in saved result files.

        accelerator : Accelerator
            Accelerator object used for distributed execution.

        verbose_images : bool
            If True, periodically save reconstruction images.

        verbose_metrics : bool
            If True, log reconstruction metrics during the pipeline.

        verbose_convergence : bool
            If True, compute and save convergence diagnostics for the initial
            SIRT reconstruction.

        verbose_gradient : bool
            If True, log gradient norms and optionally save gradient
            visualizations.

        logger : logging.Logger
            Logger used to record metrics and diagnostic information.

        **parameters : dict
            Additional parameters required by the selected regularizer.

            For ``"nf"``, expected parameters include:

                - ``"model"`` : normalizing flow model.

            For ``"cdm"``, expected parameters include:

                - ``"score_model"`` : conditional diffusion score model.
                - ``"t"`` : diffusion timestep used to construct the noisy sample
                and to index the diffusion coefficients.
                - ``"t_input"`` : diffusion timestep input provided to the score
                model.
                - ``"context_mask"`` : mask controlling the conditioning
                information provided to the score model.
                - ``"fixed_noise"`` : fixed Gaussian noise used to construct the
                noisy diffusion sample.
                - ``"sqrtab"`` : square root of the cumulative product of the
                diffusion coefficients.
                - ``"sqrtmab"`` : square root of one minus the cumulative product
                of the diffusion coefficients.

            For ``"bayes"``, expected parameters include:

                - ``"class_model"`` : classification model.
                - ``"den_model"`` : denoising model.

    Returns
    -------
        If ``regularizer != "bayes"``:
            images : list of torch.Tensor
                Stored reconstructions produced by the selected regularizer.

            images_SIRT : list of torch.Tensor
                Stored SIRT reconstructions.

            psnr_history : list of dict
                PSNR values and associated iteration information.

            ssim_history : list of dict
                SSIM values and associated iteration information.

        If ``regularizer == "bayes"``:
            images : list of torch.Tensor
                Stored reconstructions using both Bayesian gradient components.

            images_SIRT : list of torch.Tensor
                Stored SIRT reconstructions.

            images_class : list of torch.Tensor
                Stored reconstructions using only the classification gradient.

            images_deno : list of torch.Tensor
                Stored reconstructions using only the denoising gradient.

            psnr_history : list of dict
                PSNR values for all reconstruction variants.

            ssim_history : list of dict
                SSIM values for all reconstruction variants.

    Notes
    -----
    This pipeline is designed to study the inverse problem in isolation.
    Therefore, the localization and digit-classification tasks are not
    inferred during reconstruction. Instead, the ground-truth centre
    ``centre_true`` and ground-truth digit label ``digit_true`` are provided
    to the reconstruction procedure.

    Consequently, the localization and classification outputs should be
    interpreted as known estimators or conditioning information rather than
    as probabilities or jointly inferred task variables. In particular, the
    pipeline does not model uncertainty in the centre or digit prediction.
    This setup isolates the effect of the inverse-problem regularization
    conditioned on the correct auxiliary-task information.

    The reconstruction is initialized independently using SIRT. At each outer
    iteration, the number of inner gradient steps is decreased from ``R_max``
    to ``R_min`` over ``decay_iters`` iterations and remains equal to
    ``R_min`` thereafter.

    For the Bayesian regularizer, three reconstruction trajectories are
    maintained: one using both classification and denoising gradients, one
    using only the classification gradient, and one using only the denoising
    gradient. The SIRT trajectory is maintained separately as a baseline.

    The ground-truth images are used only to compute evaluation metrics and
    are not used as inputs to the reconstruction updates.
    """

    regularizer = regularizer.strip().lower()


    # initilizing pipeline variables
    images_SIRT = []
    images = []
    psnr_history = []
    ssim_history = []   

    if regularizer == "bayes":
        images_class = []
        images_deno = []
    
    
    # extracting batch size
    B = y.shape[0]


    ############################################ Initilizing tasks ############################################


    # RECONSTRUCTION
    x = x.clone()
    x_sirt = torch.zeros(B, 1, H, W, device=y.device)

    if verbose_convergence and accelerator.is_main_process and batch_id == 0:
        dx_norm_mean = torch.zeros(R_ini, device=y.device) 
        dx_norm_b0 = torch.zeros(R_ini, device=y.device)  


    for r in range(R_ini):
        x_prev = x_sirt.clone()
        x_sirt , step_data = sirt_step(A_torch, AT_torch, x_sirt, y, alpha)

        # looking into the gradient size
        if verbose_gradient and accelerator.is_main_process and batch_id ==0 and r % 10 == 0:
            logger.info(f"Gradient descent step -1_{r}: {step_data.flatten(1).norm(dim=1).mean().item()}")

        # debugging || x^{k+1} -x^k|| --> convergence 
        if verbose_convergence and accelerator.is_main_process and batch_id == 0:
            dx = x_sirt - x_prev 
            dx_norm_b0[r]   = dx[0].flatten().norm()
            dx_norm_mean[r] = dx.flatten(1).norm(dim=1).mean().item()

    
    #plotting the rate convergence
    if verbose_convergence and accelerator.is_main_process and batch_id == 0:
        plt.figure() 
        plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch")
        plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("$||x_{t+1}-x_t||$")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_initial_dx_norm_batch0_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()


    # Metrics after the initialization with SIRT (equi to I = -1, with R_ini steps)
    crop_x_true = extract_crops(x_true, centre_true )
    crop_x = extract_crops(x, centre_true )
    crop_x_sirt = extract_crops(x_sirt, centre_true )

        
    psnr_x_init = compute_psnr_results(x.detach(), x_true.detach())
    psnr_sirt_init = compute_psnr_results(x_sirt.detach(), x_true.detach())

    ssim_x_init = compute_ssim(x.detach(), x_true.detach(), data_range=1.0)
    ssim_sirt_init = compute_ssim(x_sirt.detach(), x_true.detach(), data_range=1.0)
    
    psnr_x_init_crop = compute_psnr_results(crop_x.detach(), crop_x_true.detach())
    psnr_sirt_init_crop = compute_psnr_results(crop_x_sirt.detach(), crop_x_true.detach())
    
    ssim_x_init_crop = compute_ssim(crop_x.detach(), crop_x_true.detach(), data_range=1.0)
    ssim_sirt_init_crop = compute_ssim(crop_x_sirt.detach(), crop_x_true.detach(), data_range=1.0)


    ssim_entry = {
    "I": -1,
    "num_inner_steps": R_ini,
    "ssim_x": ssim_x_init.detach().cpu(),
    "ssim_x_sirt": ssim_sirt_init.detach().cpu(),
    "ssim_x_crop": ssim_x_init_crop.detach().cpu(),
    "ssim_x_sirt_crop": ssim_sirt_init_crop.detach().cpu(),
    }

    if regularizer == "bayes":
        ssim_entry["ssim_x_class"] = ssim_sirt_init.detach().cpu()
        ssim_entry["ssim_x_deno"] = ssim_sirt_init.detach().cpu()
        ssim_entry["ssim_x_class_crop"] = ssim_sirt_init_crop.detach().cpu()
        ssim_entry["ssim_x_deno_crop"] = ssim_sirt_init_crop.detach().cpu()

    ssim_history.append(ssim_entry)


    if verbose_metrics and accelerator.is_main_process:
        logger.info(
            f"[PSNR] I=-1 | steps={R_ini} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_sirt_init.mean().item():.4f} dB"
        )
        logger.info(
            f"[SSIM] I=-1 | steps={R_ini} | x={ssim_x_init.mean().item():.4f} | x_sirt={ssim_sirt_init.mean().item():.4f}"
        )


    psnr_entry = {
    "I": -1,
    "num_inner_steps": R_ini,
    "psnr_x": psnr_x_init.detach().cpu(),
    "psnr_x_sirt": psnr_sirt_init.detach().cpu(),
    "psnr_x_crop": psnr_x_init_crop.detach().cpu(),
    "psnr_x_sirt_crop": psnr_sirt_init_crop.detach().cpu(),
    }

    if regularizer == "bayes":
        psnr_entry["psnr_x_class"] = psnr_x_init.detach().cpu()
        psnr_entry["psnr_x_deno"] = psnr_x_init.detach().cpu()
        psnr_entry["psnr_x_class_crop"] = psnr_x_init_crop.detach().cpu()
        psnr_entry["psnr_x_deno_crop"] = psnr_x_init_crop.detach().cpu()

    psnr_history.append(psnr_entry)

    


    # copying x for all the cases (based in regularizer)
    x = x.clamp(0,1)
    x_both = x.clone()


    if regularizer == "bayes":
        x_class = x.clone()
        x_deno = x.clone()



    ###### DIGIT 
    # taking the true digit 
    digit = torch.nn.functional.one_hot(digit_true, num_classes=10).float()
    digit = digit.to(x.device)
    digit_true = digit_true.to(x.device)

    ####### CENTRE 
    # taking the true centre 
    centre_true = centre_true.to(x.device)
    centre = centre_true.clone()
    

    

    images.append(x.detach().cpu())
    images_SIRT.append(x_sirt.detach().cpu())
    if regularizer == "bayes":
        images_class.append(x_class.detach().cpu())
        images_deno.append(x_deno.detach().cpu())
    

    ####################################### Conditional Loop  #######################################

    for i in range(I):
        if i < decay_iters:
            if decay_iters > 1:
                R = int(R_max - (R_max - R_min) * i / (decay_iters - 1))
            else:
                R = R_min
        else:
            R = R_min
        
        # reconstruction step using only the gradient
        for r in range(R):

            # BOTH image, meaning the one using the complete gradient
            x_both , _ = update_case(x_both, y, A_torch, AT_torch, centre, digit, alpha, gamma, sigma_data, "both", False, i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)

            # SIRT image
            x_sirt , _ = update_case(x_sirt, y, A_torch, AT_torch, centre, digit, alpha, gamma, sigma_data, "none", True, i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)

            if regularizer == "bayes":
                #only classifier
                x_class , _ = update_case(x_class, y, A_torch, AT_torch, centre, digit, alpha, gamma, sigma_data, "class", False, i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)

                #only denoiser
                x_deno , _ = update_case(x_deno, y, A_torch, AT_torch, centre, digit, alpha, gamma, sigma_data, "deno", False, i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)


        # taking crops
        crop_x = extract_crops(x_true, centre_true )
        crop_x_both = extract_crops(x_both, centre_true )
        crop_x_sirt = extract_crops(x_sirt, centre_true )
        
        if regularizer == "bayes":
            crop_x_class = extract_crops(x_class, centre_true )
            crop_x_deno = extract_crops(x_deno, centre_true)
            


        # computing metrics 
        psnr_x_init = compute_psnr_results(x_both.detach(), x_true.detach())
        psnr_sirt_init = compute_psnr_results(x_sirt.detach(), x_true.detach())

        ssim_x_init = compute_ssim(x_both.detach(), x_true.detach(), data_range=1.0)
        ssim_sirt_init = compute_ssim(x_sirt.detach(), x_true.detach(), data_range=1.0)
        
        psnr_x_init_crop = compute_psnr_results(crop_x_both.detach(), crop_x.detach())
        psnr_sirt_init_crop = compute_psnr_results(crop_x_sirt.detach(), crop_x.detach())
        
        ssim_x_init_crop = compute_ssim(crop_x_both.detach(), crop_x.detach(), data_range=1.0)
        ssim_sirt_init_crop = compute_ssim(crop_x_sirt.detach(), crop_x.detach(), data_range=1.0)
        

        if regularizer == "bayes":
            psnr_x_clas = compute_psnr_results(x_class.detach(), x_true.detach())
            psnr_x_deno = compute_psnr_results(x_deno.detach(), x_true.detach())
            ssim_x_clas = compute_ssim(x_class.detach(), x_true.detach(), data_range=1.0)
            ssim_x_deno = compute_ssim(x_deno.detach(), x_true.detach(), data_range=1.0)
            psnr_x_clas_crop = compute_psnr_results(crop_x_class.detach(), crop_x.detach())
            psnr_x_deno_crop = compute_psnr_results(crop_x_deno.detach(), crop_x.detach())
            ssim_x_clas_crop = compute_ssim(crop_x_class.detach(), crop_x.detach(), data_range=1.0)
            ssim_x_deno_crop = compute_ssim(crop_x_deno.detach(), crop_x.detach(), data_range=1.0)


        #saving metrics
        ssim_entry = {
        "I": i,
        "num_inner_steps": R,
        "ssim_x": ssim_x_init.detach().cpu(),
        "ssim_x_sirt": ssim_sirt_init.detach().cpu(),
        "ssim_x_crop": ssim_x_init_crop.detach().cpu(),
        "ssim_x_sirt_crop": ssim_sirt_init_crop.detach().cpu()
        }

        if regularizer == "bayes":
            ssim_entry["ssim_x_class"] = ssim_x_clas.detach().cpu()
            ssim_entry["ssim_x_deno"] = ssim_x_deno.detach().cpu()
            ssim_entry["ssim_x_class_crop"] = ssim_x_clas_crop.detach().cpu()
            ssim_entry["ssim_x_deno_crop"] = ssim_x_deno_crop.detach().cpu()

        ssim_history.append(ssim_entry)
        

        psnr_entry = {
        "I": i,
        "num_inner_steps": R,
        "psnr_x": psnr_x_init.detach().cpu(),
        "psnr_x_sirt": psnr_sirt_init.detach().cpu(),
        "psnr_x_crop": psnr_x_init_crop.detach().cpu(),
        "psnr_x_sirt_crop": psnr_sirt_init_crop.detach().cpu(),
        }

        if regularizer == "bayes":
            psnr_entry["psnr_x_class"] = psnr_x_clas.detach().cpu()
            psnr_entry["psnr_x_deno"] = psnr_x_deno.detach().cpu()
            psnr_entry["psnr_x_class_crop"] = psnr_x_clas_crop.detach().cpu()
            psnr_entry["psnr_x_deno_crop"] = psnr_x_deno_crop.detach().cpu()

        psnr_history.append(psnr_entry)

    

        if verbose_metrics and batch_id == 0 and accelerator.is_main_process:
            logger.info(
                f"[PSNR] I={i} | steps={R} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_sirt_init.mean().item():.4f} dB"
            )
            logger.info(
                f"[SSIM] I={i} | steps={R} | x={ssim_x_init.mean().item():.4f} | x_sirt={ssim_sirt_init.mean().item():.4f}"
            )


       
        if verbose_images and accelerator.is_main_process and batch_id == 0 and i % 2 == 0: 
            for b in range(B):

                # --- SIRT ---
                fig, ax = plt.subplots()
                im = ax.imshow(x_sirt[b,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax)
                fig.savefig(
                    os.path.join(results_dir, f"{batch_id}_{i}_sample{b}_sirt_{name}.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close(fig)

                # --- BOTH ---
                fig, ax = plt.subplots()
                im = ax.imshow(x_both[b,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax)
                fig.savefig(
                    os.path.join(results_dir, f"{batch_id}_{i}_sample{b}_flow_{name}.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close(fig)

        

        

        #save information
        if (I>20 and i % 10 == 0) or i == I-1:
    
            images.append(x_both.detach().cpu())
            images_SIRT.append(x_sirt.detach().cpu())

        if ((I>20 and i % 10 == 0) or i == I-1) and regularizer == "bayes" :
            images_class.append(x_class.detach().cpu())
            images_deno.append(x_deno.detach().cpu())
            del crop_x_class, crop_x_deno


        del crop_x, crop_x_both, crop_x_sirt
        torch.cuda.empty_cache()

    if regularizer == "bayes":
        return images, images_SIRT, images_class, images_deno, psnr_history, ssim_history
    
    return images, images_SIRT, psnr_history, ssim_history








def inverse_only_pipeline(y, A_torch, AT_torch, alpha, gamma, sigma_data, R_ini, R_min, R_max, I, decay_iters, H, W, x_true, centre_true, digit_true, regularizer, batch_id, results_dir, name, accelerator, verbose_images, verbose_metrics, verbose_convergence, verbose_gradient, logger, **parameters):
    """
    Perform the complete inverse problem reconstruction pipeline using SIRT
    followed by weighted gradient descent with a task-based regularizer.

    This pipeline is intended for the inverse problem only. The task models
    (e.g. classification, denoising, normalizing flow, or conditional
    diffusion models) are therefore used as estimators that provide
    reconstruction gradients, rather than as probabilistic models. In
    particular, the true digit label and true image centre are provided
    throughout the reconstruction. Consequently, this pipeline does not
    currently model or infer uncertainty in the task variables.

    Parameters
    ----------
        y : torch.Tensor
            Batch of measured projection data.

        A_torch : torch.Tensor
            Forward projection operator.

        AT_torch : torch.Tensor
            Adjoint/backprojection operator used by SIRT.

        alpha : float
            Step size used for the SIRT updates.

        gamma : float or torch.Tensor
            Weighting coefficients applied to the regularization gradient.
            For the Bayesian regularizer, gamma[0] and gamma[1] weight the
            classification and denoising components, respectively.

        sigma_data : float
            Data noise scale used to scale the regularization gradient.

        R_ini : int
            Number of SIRT steps used to initialise the reconstruction.

        R_min : int
            Minimum number of inner gradient-descent steps performed per
            outer iteration.

        R_max : int
            Maximum number of inner gradient-descent steps performed per
            outer iteration.

        I : int
            Number of outer reconstruction iterations.

        decay_iters : int
            Number of outer iterations over which the number of inner steps
            decreases linearly from R_max to R_min.

        H : int
            Height of the full reconstruction image.

        W : int
            Width of the full reconstruction image.

        x_true : torch.Tensor
            Ground-truth batch of images, used only for evaluation.

        centre_true : torch.Tensor
            Ground-truth centre coordinates of the digit in each image.
            These coordinates are used during reconstruction to extract and
            insert the task-relevant image crops.

        digit_true : torch.Tensor
            Ground-truth digit labels. These labels are used as conditioning
            information for the task models during reconstruction.

        regularizer : str
            Regularizer used to compute the reconstruction gradient.
            Supported options include:

            - "nf" : normalizing flow gradient.
            - "cdm" : conditional diffusion model gradient.
            - "bayes" : weighted combination of classification and denoising
              gradients.

        batch_id : int
            Index identifying the current batch.

        results_dir : str
            Directory where diagnostic plots are saved.

        name : str
            Name used to identify saved output files.

        accelerator :
            Accelerator object used for distributed execution.

        verbose_images : bool
            If True, periodically save reconstruction visualisations.

        verbose_metrics : bool
            If True, log reconstruction metrics.

        verbose_convergence : bool
            If True, compute and save convergence diagnostics for the initial
            SIRT reconstruction.

        verbose_gradient : bool
            If True, log gradient norms and save gradient visualisations.

        logger :
            Logger used for diagnostic messages.

        **parameters : dict
            Additional parameters required by the selected regularizer.

    Returns
    -------
        If regularizer == "bayes":
            tuple
                ``images`` : list of reconstructed images using the complete
                weighted gradient.

                ``images_SIRT`` : list of reconstructions obtained using SIRT
                only.

                ``images_class`` : list of reconstructions using only the
                classification gradient.

                ``images_deno`` : list of reconstructions using only the
                denoising gradient.

                ``psnr_history`` : list of dictionaries containing PSNR
                measurements at each outer iteration.

                ``ssim_history`` : list of dictionaries containing SSIM
                measurements at each outer iteration.

        Otherwise:
            tuple
                ``images`` : list of reconstructed images using the selected
                regularizer.

                ``images_SIRT`` : list of reconstructions obtained using SIRT
                only.

                ``psnr_history`` : list of dictionaries containing PSNR
                measurements.

                ``ssim_history`` : list of dictionaries containing SSIM
                measurements.

    Notes
    -----
    The reconstruction proceeds in two stages.

    First, an initial reconstruction is obtained by applying R_ini SIRT
    updates. This reconstruction is then used to initialise the different
    reconstruction cases.

    Second, each outer iteration performs R weighted gradient-descent steps.
    Each step consists of a SIRT update followed by a task-based regularization
    gradient. The number of inner steps R decreases from R_max to R_min over
    ``decay_iters`` outer iterations.

    The regularization gradient is computed on a crop centred at the known
    ground-truth centre and then inserted back into the full image. The
    resulting update can be interpreted as a weighted gradient descent step,
    where the task-derived gradient acts as a regularizer of the inverse
    problem.

    The true centre and true digit are intentionally used as conditioning
    variables. Therefore, the task models are treated as estimators providing
    useful reconstruction directions rather than as probability distributions
    over the task variables. A probabilistic treatment in which the centre
    and digit are inferred or marginalised is not performed in this pipeline.

    The SIRT-only reconstruction is maintained as a baseline. For the Bayesian
    regularizer, separate reconstructions using only the classification
    gradient and only the denoising gradient are also maintained to isolate
    the contribution of each task.
    """

    regularizer = regularizer.strip().lower()


    # initilizing pipeline variables
    images_SIRT = []
    images = []
    psnr_history = []
    ssim_history = []   

    if regularizer == "bayes":
        images_class = []
        images_deno = []
    
    
    # extracting batch size
    B = y.shape[0]


    ############################################ Initilizing tasks ############################################


    # RECONSTRUCTION
    x = torch.zeros(B, 1, H, W, device=x_true.device)

    if verbose_convergence and accelerator.is_main_process and batch_id == 0:
        dx_norm_mean = torch.zeros(R_ini, device=y.device) 
        dx_norm_b0 = torch.zeros(R_ini, device=y.device)  


    for r in range(R_ini):
        x_prev = x.clone()
        x , step_data = sirt_step(A_torch, AT_torch, x, y, alpha)

        # looking into the gradient size
        if verbose_gradient and accelerator.is_main_process and batch_id ==0 and r % 10 == 0:
            logger.info(f"Gradient descent step -1_{r}: {step_data.flatten(1).norm(dim=1).mean().item()}")

        # debugging || x^{k+1} -x^k|| --> convergence 
        if verbose_convergence and accelerator.is_main_process and batch_id == 0:
            dx = x - x_prev 
            dx_norm_b0[r]   = dx[0].flatten().norm()
            dx_norm_mean[r] = dx.flatten(1).norm(dim=1).mean().item()

    
    #plotting the rate convergence
    if verbose_convergence and accelerator.is_main_process and batch_id == 0:
        plt.figure() 
        plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch")
        plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("$||x_{t+1}-x_t||$")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_initial_dx_norm_batch0_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()


    # Metrics after the initialization with SIRT (equi to I = -1, with R_ini steps)
    crop_x_true = extract_crops(x_true, centre_true )
    crop_x = extract_crops(x, centre_true )
    

        
    psnr_x_init = compute_psnr_results(x.detach(), x_true.detach())

    ssim_x_init = compute_ssim(x.detach(), x_true.detach(), data_range=1.0)
    
    psnr_x_init_crop = compute_psnr_results(crop_x.detach(), crop_x_true.detach())
    
    ssim_x_init_crop = compute_ssim(crop_x.detach(), crop_x_true.detach(), data_range=1.0)
    


    ssim_entry = {
    "I": -1,
    "num_inner_steps": R_ini,
    "ssim_x": ssim_x_init.detach().cpu(),
    "ssim_x_sirt": ssim_x_init.detach().cpu(),
    "ssim_x_crop": ssim_x_init_crop.detach().cpu(),
    "ssim_x_sirt_crop": ssim_x_init_crop.detach().cpu(),
    }

    if regularizer == "bayes":
        ssim_entry["ssim_x_class"] = ssim_x_init.detach().cpu()
        ssim_entry["ssim_x_deno"] = ssim_x_init.detach().cpu()
        ssim_entry["ssim_x_class_crop"] = ssim_x_init_crop.detach().cpu()
        ssim_entry["ssim_x_deno_crop"] = ssim_x_init_crop.detach().cpu()

    ssim_history.append(ssim_entry)


    if verbose_metrics and accelerator.is_main_process:
        logger.info(
            f"[PSNR] I=-1 | steps={R_ini} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_x_init.mean().item():.4f} dB"
        )
        logger.info(
            f"[SSIM] I=-1 | steps={R_ini} | x={ssim_x_init.mean().item():.4f} | x_sirt={ssim_x_init.mean().item():.4f}"
        )


    psnr_entry = {
    "I": -1,
    "num_inner_steps": R_ini,
    "psnr_x": psnr_x_init.detach().cpu(),
    "psnr_x_sirt": psnr_x_init.detach().cpu(),
    "psnr_x_crop": psnr_x_init_crop.detach().cpu(),
    "psnr_x_sirt_crop": psnr_x_init_crop.detach().cpu(),
    }

    if regularizer == "bayes":
        psnr_entry["psnr_x_class"] = psnr_x_init.detach().cpu()
        psnr_entry["psnr_x_deno"] = psnr_x_init.detach().cpu()
        psnr_entry["psnr_x_class_crop"] = psnr_x_init_crop.detach().cpu()
        psnr_entry["psnr_x_deno_crop"] = psnr_x_init_crop.detach().cpu()

    psnr_history.append(psnr_entry)

    


    # copying x for all the cases (based in regularizer)
    x = x.clamp(0,1)
    x_sirt = x.clone()
    x_both = x.clone()


    if regularizer == "bayes":
        x_class = x.clone()
        x_deno = x.clone()



    ###### DIGIT 
    # taking the true digit 
    digit = torch.nn.functional.one_hot(digit_true, num_classes=10).float()
    digit = digit.to(x.device)
    digit_true = digit_true.to(x.device)

    ####### CENTRE 
    # taking the true centre 
    centre_true = centre_true.to(x.device)
    centre = centre_true.clone()
    

    

    images.append(x.detach().cpu())
    images_SIRT.append(x_sirt.detach().cpu())
    if regularizer == "bayes":
        images_class.append(x_class.detach().cpu())
        images_deno.append(x_deno.detach().cpu())
    

    ####################################### Conditional Loop  #######################################

    for i in range(I):
        if i < decay_iters:
            if decay_iters > 1:
                R = int(R_max - (R_max - R_min) * i / (decay_iters - 1))
            else:
                R = R_min
        else:
            R = R_min
        
        # reconstruction step using only the gradient
        for r in range(R):

            # BOTH image, meaning the one using the complete gradient
            x_both , _ = update_case(x_both, y, A_torch, AT_torch, centre, digit, alpha, gamma, sigma_data, "both", True, i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)

            # SIRT image
            x_sirt , _ = update_case(x_sirt, y, A_torch, AT_torch, centre, digit, alpha, gamma, sigma_data, "none", True, i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)

            if regularizer == "bayes":
                #only classifier
                x_class , _ = update_case(x_class, y, A_torch, AT_torch, centre, digit, alpha, gamma, sigma_data, "class", True, i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)

                #only denoiser
                x_deno , _ = update_case(x_deno, y, A_torch, AT_torch, centre, digit, alpha, gamma, sigma_data, "deno", True, i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)


        # taking crops
        crop_x = extract_crops(x_true, centre_true )
        crop_x_both = extract_crops(x_both, centre_true )
        crop_x_sirt = extract_crops(x_sirt, centre_true )
        
        if regularizer == "bayes":
            crop_x_class = extract_crops(x_class, centre_true )
            crop_x_deno = extract_crops(x_deno, centre_true)
            


        # computing metrics 
        psnr_x_init = compute_psnr_results(x_both.detach(), x_true.detach())
        psnr_sirt_init = compute_psnr_results(x_sirt.detach(), x_true.detach())

        ssim_x_init = compute_ssim(x_both.detach(), x_true.detach(), data_range=1.0)
        ssim_sirt_init = compute_ssim(x_sirt.detach(), x_true.detach(), data_range=1.0)
        
        psnr_x_init_crop = compute_psnr_results(crop_x_both.detach(), crop_x.detach())
        psnr_sirt_init_crop = compute_psnr_results(crop_x_sirt.detach(), crop_x.detach())
        
        ssim_x_init_crop = compute_ssim(crop_x_both.detach(), crop_x.detach(), data_range=1.0)
        ssim_sirt_init_crop = compute_ssim(crop_x_sirt.detach(), crop_x.detach(), data_range=1.0)
        

        if regularizer == "bayes":
            psnr_x_clas = compute_psnr_results(x_class.detach(), x_true.detach())
            psnr_x_deno = compute_psnr_results(x_deno.detach(), x_true.detach())
            ssim_x_clas = compute_ssim(x_class.detach(), x_true.detach(), data_range=1.0)
            ssim_x_deno = compute_ssim(x_deno.detach(), x_true.detach(), data_range=1.0)
            psnr_x_clas_crop = compute_psnr_results(crop_x_class.detach(), crop_x.detach())
            psnr_x_deno_crop = compute_psnr_results(crop_x_deno.detach(), crop_x.detach())
            ssim_x_clas_crop = compute_ssim(crop_x_class.detach(), crop_x.detach(), data_range=1.0)
            ssim_x_deno_crop = compute_ssim(crop_x_deno.detach(), crop_x.detach(), data_range=1.0)


        #saving metrics
        ssim_entry = {
        "I": i,
        "num_inner_steps": R,
        "ssim_x": ssim_x_init.detach().cpu(),
        "ssim_x_sirt": ssim_sirt_init.detach().cpu(),
        "ssim_x_crop": ssim_x_init_crop.detach().cpu(),
        "ssim_x_sirt_crop": ssim_sirt_init_crop.detach().cpu()
        }

        if regularizer == "bayes":
            ssim_entry["ssim_x_class"] = ssim_x_clas.detach().cpu()
            ssim_entry["ssim_x_deno"] = ssim_x_deno.detach().cpu()
            ssim_entry["ssim_x_class_crop"] = ssim_x_clas_crop.detach().cpu()
            ssim_entry["ssim_x_deno_crop"] = ssim_x_deno_crop.detach().cpu()

        ssim_history.append(ssim_entry)
        

        psnr_entry = {
        "I": i,
        "num_inner_steps": R,
        "psnr_x": psnr_x_init.detach().cpu(),
        "psnr_x_sirt": psnr_sirt_init.detach().cpu(),
        "psnr_x_crop": psnr_x_init_crop.detach().cpu(),
        "psnr_x_sirt_crop": psnr_sirt_init_crop.detach().cpu(),
        }

        if regularizer == "bayes":
            psnr_entry["psnr_x_class"] = psnr_x_clas.detach().cpu()
            psnr_entry["psnr_x_deno"] = psnr_x_deno.detach().cpu()
            psnr_entry["psnr_x_class_crop"] = psnr_x_clas_crop.detach().cpu()
            psnr_entry["psnr_x_deno_crop"] = psnr_x_deno_crop.detach().cpu()

        psnr_history.append(psnr_entry)

    

        if verbose_metrics and batch_id == 0 and accelerator.is_main_process:
            logger.info(
                f"[PSNR] I={i} | steps={R} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_sirt_init.mean().item():.4f} dB"
            )
            logger.info(
                f"[SSIM] I={i} | steps={R} | x={ssim_x_init.mean().item():.4f} | x_sirt={ssim_sirt_init.mean().item():.4f}"
            )


       
        if verbose_images and accelerator.is_main_process and batch_id == 0 and i % 2 == 0: 
            for b in range(B):

                # --- SIRT ---
                fig, ax = plt.subplots()
                im = ax.imshow(x_sirt[b,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax)
                fig.savefig(
                    os.path.join(results_dir, f"{batch_id}_{i}_sample{b}_sirt_{name}.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close(fig)

                # --- BOTH ---
                fig, ax = plt.subplots()
                im = ax.imshow(x_both[b,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax)
                fig.savefig(
                    os.path.join(results_dir, f"{batch_id}_{i}_sample{b}_flow_{name}.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close(fig)

        #save information
        if (I>20 and i % 10 == 0) or i == I-1:
    
            images.append(x_both.detach().cpu())
            images_SIRT.append(x_sirt.detach().cpu())

        if ((I>20 and i % 10 == 0) or i == I-1) and regularizer == "bayes" :
            images_class.append(x_class.detach().cpu())
            images_deno.append(x_deno.detach().cpu())
            del crop_x_class, crop_x_deno


        del crop_x, crop_x_both, crop_x_sirt
        torch.cuda.empty_cache()

    if regularizer == "bayes":
        return images, images_SIRT, images_class, images_deno, psnr_history, ssim_history
    
    return images, images_SIRT, psnr_history, ssim_history



