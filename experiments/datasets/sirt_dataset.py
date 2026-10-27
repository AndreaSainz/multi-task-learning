import torch
import os
import json
import math
import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from tomosipo.torch_support import to_autograd
from code.dataloader.data_import import import_data
from code.dataloader.dataset_creator import save_sirt_dataset
from code.utils.logger import setup_logger
from code.ct_recon.operators import ct_operator
from accelerate import Accelerator
from pathlib import Path



# SEED
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


# PARAMETERS
batch_size = 128
accelerator = Accelerator()
num_workers = 0
device= "cuda" 

### data
n_train = 48000
H = 100
W = 100
num_angles = 60
angle_range = [0, np.pi]
i_0 = 10**5
sigma_data = 0.3

# model training parameters 
noisy_images= False
sigma_image= [0.1,0.4]
sirt_images= True
sirt_iterations = (1, 20)
sirt_alpha= 10**4/(75.77664947509766)**2


# logger and file names
log_dir = "logs"
name = "sirt_dataset_creation"
os.makedirs(log_dir, exist_ok=True)

# saving directory
results_dir = "multi-task-learning/data/MNIST_sirt/"
os.makedirs(results_dir, exist_ok=True)

## logger
logger = setup_logger(log_dir, name, accelerator = accelerator)

# define CT geometry
A = ct_operator(H, W, angle_range, num_angles, "parallel")
A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)


# dataloader import
train_dataloader, val_dataloader = import_data(True, True, n_train, None, A, num_angles, sigma_data, i_0, H, W, noisy_images,
                sigma_image, sirt_images, sirt_iterations, sirt_alpha, batch_size, seed, logger, accelerator, device)

#save dataset configuration
data_config = {
    "n_train": n_train,
    "H": H,
    "W": W,
    "num_angles": num_angles,
    "i_0": i_0,
    "sigma_data": sigma_data,
    "noisy_images": noisy_images,
    "sigma_image": sigma_image,
    "sirt_images": sirt_images,
    "sirt_iterations": sirt_iterations,
    "sirt_alpha": sirt_alpha,
    "batch_size": batch_size,
    "num_workers": num_workers,
    "device": device
}

dataset_dir = Path(results_dir)
dataset_dir.mkdir(parents=True, exist_ok=True)
config_path = dataset_dir / "config.json"

with open(config_path, "w") as f:
    json.dump(data_config, f, indent=4)


# create datasets
save_sirt_dataset( train_dataloader, os.path.join(results_dir, "mnist_sirt_train.pt"))
save_sirt_dataset( val_dataloader, os.path.join(results_dir, "mnist_sirt_val.pt"))