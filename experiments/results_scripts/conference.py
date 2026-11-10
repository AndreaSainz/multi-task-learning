import torch
import torch.nn.functional as F
import numpy as np
import tomosipo as ts
import math
import logging
import os
import random
from pathlib import Path
from skimage.transform import iradon
import json
from LION.classical_algorithms.fdk import fdk
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
from code.ct_recon.operators import ct_operator
from code.utils.logger import setup_logger
from code.models.classifier import CNNClassifier
from code.dataloader.data_import import import_data
from code.utils.metrics import compute_psnr_results
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



seed = 29072000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


###################################################################### Parameters #####################################################################

### data
accelerator = Accelerator()
device = accelerator.device
n_train = 10
batch_size = 1
B = batch_size
seed = 29072000
H = 28
W = 28
num_angles = 30 #60
angle_range = [0, np.pi]
i_0 = 10**5
sigma_data = 0.3

noisy_images=False
sigma_image=False 
sirt_images=True 
sirt_iterations=10
sirt_alpha=2e-1/(28.349531173706055)**2            #1/(liptchitz constant)^2: 2e-1/(40.08146667480469)**2 



# Model name
model_name = "cnn_classifier_mnist_noisy_best"

#saving results
results_dir = "experiments/Results/Checks/"

###################################################################### Logger #####################################################################
# create logger for checkpoints
log_dir = "logs"
name = "conference"
logger = setup_logger(log_dir, name, accelerator = accelerator)


###################################################################### Data #####################################################################

# define CT geometry
A = ct_operator(H, W, angle_range, num_angles, "parallel")
A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)


# create pytorch training dataloader
train_dataloader = import_data(True, False, n_train, None, A, num_angles, sigma_data, i_0, H, W, 
                    noisy_images, sigma_image,  sirt_images , sirt_iterations,  sirt_alpha,
                    batch_size , seed, logger, accelerator , device)



################################################################## Classifier ################################################################

# Directories
checkpoint_dir = Path("checkpoints/classifier")
class_model = CNNClassifier(noise_aware=False)

# Paths
checkpoint_path = checkpoint_dir / f"{model_name}.pt"

# Load checkpoint
checkpoint = torch.load(checkpoint_path,map_location="cpu")

if isinstance(checkpoint, dict) and "model" in checkpoint:
    class_model.load_state_dict(checkpoint["model"])
else:
    class_model.load_state_dict(checkpoint)

class_model.eval() 
for p in class_model.parameters():
   p.requires_grad_(False)

class_model = class_model.to("cpu")



######################################################################### Loop #################################################################
for batch_id, batch in enumerate(train_dataloader):
    if batch_id == 0:
        mnist = batch['mnist_image']
        img_true = batch['ground_truth_image']
        img_mnist = batch['ground_truth_small']
        sinogram = batch['noisy_sinogram']
        sirt_recon = batch['sirt_reconstructions'][-1]
        label = batch['label']

        
        

        #checking it's working
        psnr_true_true = compute_psnr_results(mnist,img_true)
        psnr_true_true2 = compute_psnr_results(mnist,img_mnist)


        logger.info(f"PSNR of mnist images from dataloader: {psnr_true_true} | {psnr_true_true2}")

        # Reconstruction using FDK or FBP 
        sino = sinogram[0, 0].cpu().numpy().T
        theta = np.linspace(angle_range[0], angle_range[1], num_angles, endpoint=False)
        fbp = iradon(sino, theta=theta, filter_name='ramp', circle=True)
        fbp_tensor = torch.as_tensor(fbp, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        fbp_cpu = fbp_tensor.cpu()
        sirt_tensor = sirt_recon.unsqueeze(1).cpu()
        mnist = mnist.cpu()



        logits_fbp = class_model(fbp_cpu)
        digit_fbp = torch.softmax(logits_fbp, dim=1).argmax(dim=1)
        logger.info(f"Classification FBP: {digit_fbp} ")

        logits_sirt = class_model(sirt_tensor)
        digit_sirt = torch.softmax(logits_sirt, dim=1).argmax(dim=1)
        logger.info(f"Classification SIRT: {digit_sirt} ")

        logits_true = class_model(mnist)
        digit_true = torch.softmax(logits_true, dim=1).argmax(dim=1)
        logger.info(f"Classification TRUE: {digit_true} ")

        #create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # Images
        images = [mnist.cpu().numpy(),fbp,sirt_tensor.cpu().numpy()]

        titles = ["Ground Truth (MNIST)","FBP","SIRT"]

        # Same color scale for all images
        vmin = min(img.min().item() for img in images)
        vmax = max(img.max().item() for img in images)

        # Plot images
        for ax, img, title in zip(axes, images, titles):
            im = ax.imshow(img.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
        # One shared colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Intensity")
        fig.suptitle("Image Reconstructions", fontsize=16)
        plt.savefig( os.path.join(results_dir, "reconstructions.png"), dpi=300, bbox_inches="tight")
        plt.close()


