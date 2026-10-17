import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_import import import_data
from code.pipelines.pipelines import inverse_only_pipeline
from code.models.score_unet import ContextUnet, ddpm_schedules
from code.utils.image_cropping import extract_crops, insert_crop
from code.utils.image_initilization import true_images_lines
from code.utils.plots import images_plots, psnr_plots, ssim_plots
from code.utils.logger import setup_logger
from code.ct_recon.operators import ct_operator
import matplotlib
import logging
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os



###################################################################### Parameters #####################################################################

### data
accelerator = Accelerator()
device = accelerator.device
n_train = 10
batch_size = 1
save_batch_id = 0
H = 100
W = 100
num_angles = 60
angle_range = [0, np.pi]
i_0 = 10**5
sigma_data = 0.3


### training parameters
R_ini = 80
R_min = 70
R_max = 70
decay_iters = 70
I = 10
alpha = 2e-1/(75.77664947509766)**2            #1/(liptchitz constant)^2: 2e-1/(53.595951080322266)**2 
gamma =  5e-2 #1e-3


# saving directory
results_dir = "simpler_pipeline/Results/ExperimentsScore/bands"
os.makedirs(results_dir, exist_ok=True)

# logger and file names
log_dir = "logs"
name = "cdm_central_bands"
os.makedirs(log_dir, exist_ok=True)

## seeds
seed = 29072000
torch.manual_seed(0)


###################################################################### Logger ##################################################################
logger = setup_logger(log_dir, name, accelerator = accelerator)

###################################################################### Data #####################################################################

# define CT geometry
A = ct_operator(H, W, angle_range, num_angles, "parallel")
A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)


# create pytorch training dataloader
train_dataloader = import_data(True, False, n_train, None, A, num_angles, sigma_data, i_0, H, W, batch_size, seed, logger, accelerator, device)

###################################################################### Trained Score Models #####################################################################

score_model = ContextUnet(in_channels=1, n_feat=128, n_classes=10)

checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/diffusion_model/model_39.pth", map_location="cpu")
t = torch.full((batch_size,), 5, device=device) #time step in the denoiser
alpha_t = checkpoint["alpha_t"]
beta_t = 1 - alpha_t
beta1 = beta_t[0].item()
beta2 = beta_t[-1].item()
T_max =  len(alpha_t) - 1
model_state = {
    k.replace("nn_model.", ""): v
    for k, v in checkpoint.items()
    if k.startswith("nn_model.")
}
score_model.load_state_dict(model_state)
sqrtab = checkpoint["sqrtab"].to(device)
sqrtmab = checkpoint["sqrtmab"].to(device)

# model in evaluation mode, we don't want to train it
score_model.eval() 
for p in score_model.parameters():
   p.requires_grad_(False)

# for using the score
t_input = t.float().view(batch_size, 1) / T_max
fixed_noise = torch.randn((batch_size, 1, 28, 28), device=device)
context_mask = torch.zeros(batch_size, device=device) #we want to use the condition

score_model,train_dataloader = accelerator.prepare(score_model, train_dataloader)

#parameters for the main loop function
regularizer = "cdm"
parameters = {
    "score_model": score_model,
    "t" : t,
    "t_input" : t_input,
    "context_mask" : context_mask,
    "fixed_noise" : fixed_noise,
    "sqrtab" : sqrtab,
    "sqrtmab" : sqrtmab
}


###################################################################### Multi-task function ###############################################################

# initilizing empty lists
gt_images = []
gt_digits = []
gt_centers = []
model_images = []
sirt_images = []
model_psnr = []
sirt_psnr = []
model_ssim = []
sirt_ssim = []
model_psnr_crop = []
sirt_psnr_crop = []
model_ssim_crop = []
sirt_ssim_crop = []



for batch_id, batch in enumerate(train_dataloader):

    ground_truth_1 = batch["ground_truth_image"]
    ground_truth_2 = batch["label"]
    ground_truth_3 = batch["centre"]
    input_data = batch["noisy_sinogram"]

    B = ground_truth_1.shape[0]

    #check batch_size
    if B != batch_size:
        raise ValueError(f"Batch Size is not weel defined, B:{B} is not {batch_size}")


    # Initializing the image with a white band on the digit
    x = true_images_lines(ground_truth_1, ground_truth_3, H, W, False, None, True, 5)

    # # Initilizing with black image
    # x = torch.zeros(B, 1, H, W, device=x_true.device)

    # # Initlizing the image with random black pixels in the digit
    # x = true_images_pixels(ground_truth_1, ground_truth_3, H, W, False, None, 0.2)



    

    if accelerator.is_main_process and batch_id == 0 :
        logger.info(f"Original digit: {ground_truth_2}")
        logger.info(f"original centre: {ground_truth_3}")

    images, images_SIRT, psnr_history, ssim_history = inverse_only_pipeline(input_data, A_torch, AT_torch, alpha, 
                                                        gamma, sigma_data, R_ini, R_min, R_max, I, decay_iters, H, W, ground_truth_1, 
                                                        ground_truth_3, ground_truth_2, regularizer, batch_id, results_dir, 
                                                        name, accelerator, True, True, False, False, logger, **parameters)


    if accelerator.is_main_process and batch_id == 0:

        #defining a dictorionary with the images
        images_dict = {"SIRT only": images_SIRT, "SIRT + CDM": images, "Ground Truth": ground_truth_1}

        # images plot
        images_plots(results_dir, name, batch_id, save_batch_id, images_dict)

        # PSNR plots 
        psnr_plots(results_dir, name, batch_id, psnr_history)

        # SSIM plots 
        ssim_plots(results_dir, name, batch_id, ssim_history)

    
        for b in range(B):
            gt_images.append(ground_truth_1.detach().cpu())
            gt_digits.append(ground_truth_2.detach().cpu())
            gt_centers.append(ground_truth_3.detach().cpu())
            model_images.append(images)
            sirt_images.append(images_SIRT)
            sirt_psnr.append([d["psnr_x_sirt"] for d in psnr_history])
            model_psnr.append([d["psnr_x"] for d in psnr_history])
            sirt_ssim.append([d["ssim_x_sirt"] for d in ssim_history])
            model_ssim.append([d["ssim_x"] for d in ssim_history])
            sirt_psnr_crop.append([d["psnr_x_sirt_crop"] for d in psnr_history])
            model_psnr_crop.append([d["psnr_x_crop"] for d in psnr_history])
            sirt_ssim_crop.append([d["ssim_x_sirt_crop"] for d in ssim_history])
            model_ssim_crop.append([d["ssim_x_crop"] for d in ssim_history])


        results = {
            "R_ini" : R_ini,
            "R_min" : R_min,
            "R_max" : R_max,
            "decay_iters" : decay_iters,
            "I" : I,
            "sigma_data": sigma_data,
            "i_0": i_0,
            "num_angles" : num_angles,
            "angle_range" : angle_range,
            "gt_images": gt_images,
            "gt_digits": gt_digits,
            "gt_centers": gt_centers,
            "model_images": model_images,      # [batch][iter][B,1,H,W]
            "sirt_images": sirt_images,
            "sirt_psnr": sirt_psnr,
            "model_psnr": model_psnr,
            "sirt_ssim": sirt_ssim,
            "model_ssim": model_ssim,
            "sirt_psnr_crop": sirt_psnr_crop,
            "model_psnr_crop": model_psnr_crop,
            "sirt_ssim_crop": sirt_ssim_crop,
            "model_ssim_crop": model_ssim_crop
            }


        rank = accelerator.process_index
        torch.save( results, os.path.join(results_dir, f"results_mnist_ct_{name}_rank{rank}.pt" ))

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            for h in list(logger.handlers):
                try:
                    h.flush()
                    h.close()
                finally:
                    logger.removeHandler(h)

            logging.shutdown()

        accelerator.wait_for_everyone()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        
        break