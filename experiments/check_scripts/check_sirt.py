import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_import import import_data
from code.ct_recon.operators import ct_operator
from code.ct_recon.SIRT import normalization_matrices, sirt_method
from code.utils.image_cropping import extract_crops, insert_crop
from code.utils.plots import images_plots, psnr_plots, ssim_plots
from code.utils.logger import setup_logger
import matplotlib
import logging
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

## seeds
seed = 29072000
torch.manual_seed(0)

### data
accelerator = Accelerator()
device = accelerator.device
n_train = 10
batch_size = 1
H = 100
W = 100
num_angles = 60
angle_range = [0, np.pi]
i_0 = 10**5
sigma_data = 0.3

# logger and file names
log_dir = "logs"
name = "sirt_check"
os.makedirs(log_dir, exist_ok=True)

# saving directory
results_dir = "experiments/Results/Checks/SIRT"
os.makedirs(results_dir, exist_ok=True)

## logger
logger = setup_logger(log_dir, name, accelerator = accelerator)

# SIRT 
alpha = 10**4/(75.77664947509766)**2
sirt_iter = 20


# define CT geometry
A = ct_operator(H, W, angle_range, num_angles, "parallel")
A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)



# create pytorch test dataloader
test_dataloader = import_data(False, True, 10, None, A, num_angles, sigma_data, i_0, H, W, batch_size, seed, logger, accelerator, device)


#normalization matrices
M , D = normalization_matrices(A_torch, AT_torch, batch_size, H, W, num_angles, device)

sirt_recon = []

#for loop
for batch_id, batch in enumerate(test_dataloader):

    ground_truth_1 = batch["ground_truth_image"]
    ground_truth_2 = batch["label"]
    ground_truth_3 = batch["centre"]
    input_data = batch["noisy_sinogram"]

    x = torch.zeros(batch_size, 1, H, W, device=device)
    sirt_recon.append(x)

    for i in range(sirt_iter):
        x, _ = sirt_method(A_torch, AT_torch, x, input_data, D, M, alpha)
        sirt_recon.append(x)
    
#create figure
fig, axes = plt.subplots(5, 6, figsize=(12, 12))

# Same color scale for all images
vmin = min(img.min().item() for img in sirt_recon)
vmax = max(img.max().item() for img in sirt_recon)


for i, ax in enumerate(axes.flat):
    im = ax.imshow(
        sirt_recon[i].detach().cpu().numpy().squeeze(),
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )
    ax.set_title(f"Batch {i + 1}")
    ax.axis("off")


# One shared colorbar
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Intensity")

fig.suptitle("SIRT reconstructions", fontsize=16)
plt.savefig(os.path.join(results_dir, "sirt_reconstructions_2.png"), dpi=300, bbox_inches="tight")
plt.close()