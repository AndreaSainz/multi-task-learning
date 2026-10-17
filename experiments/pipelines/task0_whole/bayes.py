import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_import import import_data
from code.models.classifier import CNNClassifier
from code.models.denoiser import ResUNetDenoiser
from code.pipelines.pipelines import inverse_only_pipeline
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
R_ini = 90
R_min = 90
R_max = 90
decay_iters = 90
I = 40
alpha = 2e-1/(75.77664947509766)**2            #1/(liptchitz constant)^2: 2e-1/(53.595951080322266)**2 
gamma  = [0.52, 0.12]   #[8e-2, 5e-2] 


# saving directory
results_dir = "simpler_pipeline/Results/ExperimentsClassDeno/bands"
os.makedirs(results_dir, exist_ok=True)

# logger and file names
log_dir = "logs"
name = "bayes_central_bands"
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

###################################################################### Trained Classifier and Denoiser Models #####################################################################

class_model = CNNClassifier()
checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/classifier/cnn_classifier_clean1yrp_best.pt", map_location="cpu")
class_model.load_state_dict(checkpoint) #["model"])


den_model = ResUNetDenoiser()
checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/denoiser/dncnn_best.pt", map_location="cpu")
den_model.load_state_dict(checkpoint)


# model in evaluation mode, we don't want to train it
class_model.eval() 
for p in class_model.parameters():
   p.requires_grad_(False)

den_model.eval() 
for p in den_model.parameters():
   p.requires_grad_(False)

class_model, den_model, train_dataloader = accelerator.prepare(class_model, den_model, train_dataloader)

#parameters for the main loop function
regularizer = "bayes"
parameters = {
    "class_model" : class_model,
    "den_model" : den_model
}
###################################################################### Multi-task function ###############################################################

# initilizing empty lists
gt_images = []
gt_digits = []
gt_centers = []
model_images = []
sirt_images = []
class_images = []
deno_images = []
model_psnr = []
sirt_psnr = []
class_psnr = []
deno_psnr = []
model_ssim = []
sirt_ssim = []
class_ssim = []
deno_ssim = []
model_psnr_crop = []
sirt_psnr_crop = []
class_psnr_crop = []
deno_psnr_crop = []
model_ssim_crop = []
sirt_ssim_crop = []
class_ssim_crop = []
deno_ssim_crop = []



for batch_id, batch in enumerate(train_dataloader):

    ground_truth_1 = batch["ground_truth_image"]
    ground_truth_2 = batch["label"]
    ground_truth_3 = batch["centre"]
    input_data = batch["noisy_sinogram"]

    B = ground_truth_1.shape[0]

    #check batch_size
    if B != batch_size:
        raise ValueError(f"Batch Size is not weel defined, B:{B} is not {batch_size}")



    

    if accelerator.is_main_process and batch_id == 0 :
        logger.info(f"Original digit: {ground_truth_2}")
        logger.info(f"original centre: {ground_truth_3}")

    images, images_SIRT, images_class, images_deno, psnr_history, ssim_history = inverse_only_pipeline(input_data, A_torch, AT_torch, alpha, 
                                                        gamma, sigma_data, R_ini, R_min, R_max, I, decay_iters, H, W, ground_truth_1, 
                                                        ground_truth_3, ground_truth_2, regularizer, batch_id, results_dir, 
                                                        name, accelerator, True, True, False, False, logger, **parameters)


    if accelerator.is_main_process and batch_id == 0:

        #defining a dictorionary with the images
        images_dict = {"SIRT only": images_SIRT, "SIRT + bayes": images, "SIRT + Class": images_class, "SIRT + Deno": images_deno, "Ground Truth": ground_truth_1}

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
            class_images.append(images_class)
            deno_images.append(images_deno)
            sirt_psnr.append([d["psnr_x_sirt"] for d in psnr_history])
            model_psnr.append([d["psnr_x"] for d in psnr_history])
            class_psnr.append([d["psnr_x_class"] for d in psnr_history])
            deno_psnr.append([d["psnr_x_deno"] for d in psnr_history])
            sirt_ssim.append([d["ssim_x_sirt"] for d in ssim_history])
            model_ssim.append([d["ssim_x"] for d in ssim_history])
            class_ssim.append([d["ssim_x_class"] for d in ssim_history])
            deno_ssim.append([d["ssim_x_deno"] for d in ssim_history])
            sirt_psnr_crop.append([d["psnr_x_sirt_crop"] for d in psnr_history])
            model_psnr_crop.append([d["psnr_x_crop"] for d in psnr_history])
            class_psnr_crop.append([d["psnr_x_class_crop"] for d in psnr_history])
            deno_psnr_crop.append([d["psnr_x_deno_crop"] for d in psnr_history])
            sirt_ssim_crop.append([d["ssim_x_sirt_crop"] for d in ssim_history])
            model_ssim_crop.append([d["ssim_x_crop"] for d in ssim_history])    
            class_ssim_crop.append([d["ssim_x_class_crop"] for d in ssim_history])
            deno_ssim_crop.append([d["ssim_x_deno_crop"] for d in ssim_history])




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
            "class_images": class_images,      
            "deno_images": deno_images,
            "sirt_psnr": sirt_psnr,
            "model_psnr": model_psnr,
            "class_psnr": class_psnr,
            "deno_psnr": deno_psnr,
            "sirt_ssim": sirt_ssim,
            "model_ssim": model_ssim,
            "class_ssim": class_ssim,
            "deno_ssim": deno_ssim,
            "sirt_psnr_crop": sirt_psnr_crop,
            "model_psnr_crop": model_psnr_crop,
            "class_psnr_crop": class_psnr_crop,
            "deno_psnr_crop": deno_psnr_crop,
            "sirt_ssim_crop": sirt_ssim_crop,
            "model_ssim_crop": model_ssim_crop,
            "class_ssim_crop": class_ssim_crop,
            "deno_ssim_crop": deno_ssim_crop
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