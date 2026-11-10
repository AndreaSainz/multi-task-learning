# code from: https://github.com/DPS2022/diffusion-posterior-sampling/tree/main

import torch
import torch.nn.functional as F
import numpy as np
import tomosipo as ts
import math
import logging
import os
import random
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
from code.ct_recon.operators import ct_operator
from code.utils.logger import setup_logger
from code.dataloader.data_import import import_data
from code.models.score_unet import ContextUnet, ddpm_schedules
from code.ct_recon.SIRT import normalization_matrices, sirt_method
from code.utils.metrics import compute_psnr_results, compute_ssim, compute_mse, update_confusion_matrix, metrics_from_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd

torch.backends.cudnn.enabled = False
seed = 29072000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


###################################################################### Parameters #####################################################################

### data
accelerator = Accelerator()
device = accelerator.device
n_train = 100
n_val = 30
batch_size = 10
B = batch_size
seed = 29072000
H = 28
W = 28
num_angles = 60 #30
angle_range = [0, np.pi]
i_0 = 10**5
sigma_data = 0.3





#saving results
results_dir = "experiments/Results/DPS"


###################################################################### Logger ##################################################################
# create logger for checkpoints
log_dir = "logs"
name = "DPS"
logger = setup_logger(log_dir, name, accelerator = accelerator)


###################################################################### Data #####################################################################

# define CT geometry
A = ct_operator(H, W, angle_range, num_angles, "parallel")
A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)


# create pytorch training dataloader
train_dataloader, val_dataloader = import_data(True, True, n_train, n_val, A, num_angles, sigma_data, i_0, H, W, False,False,False,False,False,
                     batch_size , seed, logger, accelerator , device)



################################################################## Trained score model ############################################################
score_model = ContextUnet(in_channels=1, n_feat=128, n_classes=10)


checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/diffusion_model/model_39.pth", map_location="cpu")
alpha_t = checkpoint["alpha_t"].to(device)
beta_t = 1 - alpha_t
T = len(beta_t) - 1
I = T
alphabar_t = checkpoint["alphabar_t"].to(device) # \bar{\alpha_t}
sqrtab = checkpoint["sqrtab"].to(device) # \sqrt{\bar{\alpha_t}}
sqrtmab = checkpoint["sqrtmab"].to(device) # \sqrt{1-\bar{\alpha_t}}

model_state = {
    k.replace("nn_model.", ""): v
    for k, v in checkpoint.items()
    if k.startswith("nn_model.")
}
score_model.load_state_dict(model_state)
score_model = score_model.to(device)

# model in evaluation mode, we don't want to train it
score_model.eval() 
for p in score_model.parameters():
   p.requires_grad_(False)

# for using the score
context_mask_cond = torch.zeros(batch_size, device=device) #we want to use the condition
context_mask = torch.ones(batch_size, device=device)



###################################################################### DPS Algo #####################################################################

# normlizing matrices
M , D = normalization_matrices(A, A.T, 1, H, W, num_angles, device, num_extra_dims=1)
zeta_scale = 0.09295959900229107
sigma_scale = 0.08684175776720753

def DPS(y, digit_class, I, batch_id): #, zeta_scale, sigma_scale):
    #hyperparameters
    zeta_schedule = torch.linspace(0.0, zeta_scale, I)
    sigma_schedule = torch.linspace(0.0, sigma_scale, I)

    B = y.shape[0]
    device = y.device
    
    # initilize the image
    x = torch.randn((B, 1, 28, 28), device=device)

    #DPS 
    for i in reversed(range(0, I)):

        if i > 0:
            alpha_bar_prev = alphabar_t[i - 1]
        else:
            alpha_bar_prev = torch.ones_like(alphabar_t[i])

        t_input = torch.full((B,), (i+1)/T, device=device, dtype=torch.float32)

        # score value for x_i in time i
        x = x.detach().requires_grad_(True)
        eps_hat = score_model(x, digit_class, t_input, context_mask)
        score_value = - eps_hat / sqrtmab[i]

        # new image based on the diffusion process (clean image)
        x_0 = (x + (1-alphabar_t[i])*score_value)/sqrtab[i]

        # random noise (same size as images)
        z = torch.randn_like(x)

        # posterior mean (last value of sigma should be 0)
        x_1 = (((alpha_t[i])**(0.5))*(1-alpha_bar_prev)/(1-alphabar_t[i])) * x + (torch.sqrt(alpha_bar_prev)*beta_t[i]/(1-alphabar_t[i])) * x_0 + sigma_schedule[i]*z
        

        # Reconstruction step (make sure the image looks similar to the data)
        ## y_pred = A_torch(x_0)
        ## loss = (0.5 / sigma_data**2)*torch.sum((y - y_pred)**2)
        ## grad = torch.autograd.grad(outputs=loss, inputs=x)[0]

        _ , grad = sirt_method(A_torch, AT_torch, x_0, y, D, M, zeta_schedule[i])
        x = x_1 + grad
        

        if batch_id <2 and i%50 == 0:
            # Save batch reconstructions as a grid
            B = len(x)

            ncols = int(np.ceil(np.sqrt(B)))
            nrows = int(np.ceil(B / ncols))

            fig = plt.figure(figsize=(ncols * 1.5 + 0.4, nrows * 1.5),constrained_layout=True)
            gs = fig.add_gridspec( nrows, ncols + 1, width_ratios=[1] * ncols + [0.04])

            # Create image axes
            axes = [ fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]

            # Common colorbar axis
            cax = fig.add_subplot(gs[:, -1])

            # Plot images
            for j in range(B):
                img = x[j].detach().cpu().squeeze().numpy()
                im = axes[j].imshow(img,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
                axes[j].axis("off")

            # Hide unused axes
            for j in range(B, len(axes)):
                axes[j].axis("off")

            # Shared colorbar
            fig.colorbar(im, cax=cax)
            fig.savefig(os.path.join( results_dir, f"unconditional_reconstruction_batch_{batch_id}_iter_{i}.png"),dpi=200,bbox_inches="tight",pad_inches=0.05)
            plt.close(fig)

    return x.detach()


# for loop for data in dataloader 
recon_images = []
gt_images = []

for batch_id, batch in enumerate(train_dataloader):

    ground_truth_1 = batch["ground_truth_image"].to(device)
    img_true = ground_truth_1.detach().cpu()
    input_data = batch["noisy_sinogram"].to(device)
    labels = batch["label"].to(device)

    x = DPS(input_data, labels, I, batch_id)

    # Save batch true_images as a grid
    B = len(x)

    ncols = int(np.ceil(np.sqrt(B)))
    nrows = int(np.ceil(B / ncols))

    fig = plt.figure(figsize=(ncols * 1.5 + 0.4, nrows * 1.5),constrained_layout=True)

    gs = fig.add_gridspec( nrows, ncols + 1, width_ratios=[1] * ncols + [0.04])

    # Create image axes
    axes = [ fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]

    # Common colorbar axis
    cax = fig.add_subplot(gs[:, -1])

    # Plot images
    for j in range(B):
        img = img_true[j].detach().cpu().squeeze().numpy()

        im = axes[j].imshow(img,cmap="gray",vmin=0,vmax=1,interpolation="nearest")

        axes[j].axis("off")

    # Hide unused axes
    for j in range(B, len(axes)):
        axes[j].axis("off")

    # Shared colorbar
    fig.colorbar(im, cax=cax)

    fig.savefig(os.path.join( results_dir, f"true_images_batch_{batch_id}.png"),dpi=200,bbox_inches="tight",pad_inches=0.05)

    plt.close(fig)
    

    # if accelerator.is_main_process and batch_id<2: 
    #     for b in range(B):
    #         recon = x[b].detach().cpu()
    #         img_true = ground_truth_1[b].detach().cpu()
            
    #         # Save reconstruction as image
    #         fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    #         im0 = ax[0].imshow(recon.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    #         ax[0].set_title("recon DPS")
    #         ax[0].axis("off")

    #         im1 = ax[1].imshow(img_true.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    #         ax[1].set_title("Ground Truth")
    #         ax[1].axis("off")

    #         # Shared colorbar
    #         fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    
    #         plt.savefig(os.path.join(results_dir, f"reconstruction_DPS_{b}.png"),bbox_inches="tight",pad_inches=0)
    #         plt.close()

    #         # Metrics
    #         mse = compute_mse(recon, img_true)
    #         psnr = compute_psnr_results(recon, img_true)
    #         ssim = compute_ssim(recon, img_true, 1.0)

    #         print(f"mse {b}: {mse}")
    #         print(f"psnr {b}: {psnr}")
    #         print(f"ssim {b}: {ssim}")
    #         gt_images.append(img_true)
    #         recon_images.append(recon)
    # else: 
    #     break





# def objective(trial):

#     zeta_scale = trial.suggest_float("zeta_scale",1e-7,1e-1,log=True)

#     sigma_scale = trial.suggest_float("sigma_scale",1e-3,1.0,log=True)

#     ssim_values = []

#     # Use the same validation samples in every trial
#     for batch_id, batch in enumerate(val_dataloader):

#         input_data = batch["noisy_sinogram"].to(device)
#         ground_truth = batch["ground_truth_image"].to(device)
#         labels = batch["label"].to(device)

#         # Use the same random seed for every trial
#         torch.manual_seed(seed + batch_id)

#         recon = DPS(input_data,labels,I,zeta_scale,sigma_scale)

#         ssim = compute_ssim(recon.detach().cpu(),ground_truth.detach().cpu(),1.0)

#         # Store individual SSIM values
#         ssim_values.extend(torch.as_tensor(ssim).detach().cpu().reshape(-1).tolist())

#     # Mean SSIM over the entire validation dataset
#     mean_ssim = np.mean(ssim_values)

#     return mean_ssim

# study = optuna.create_study(
#     direction="maximize",
#     study_name="DPS_hyperparameter_search"
# )

# study.optimize(
#     objective,
#     n_trials=30
# )

# print("Best parameters:")
# print(study.best_params)

# print("Best SSIM:")
# print(study.best_value)

# results_df = study.trials_dataframe()

# results_df.to_csv(
#     os.path.join(results_dir, "optuna_results.csv"),
#     index=False
# )

# print("Best parameters:", study.best_params)

