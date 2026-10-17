# code from: https://github.com/DPS2022/diffusion-posterior-sampling/tree/main

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_loader import FashionMnistDataset
from code.models.score import score_model
from code.utils.metrics import compute_psnr_results, compute_ssim, compute_mse, update_confusion_matrix, metrics_from_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import logging
import os
from torch.utils.data import Subset
import normflows as nf
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


seed = 29072000
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

###################################################################### Parameters #####################################################################

### data
seed = 29072000
# I don't want a bigger image right now
H = 28
W = 28
num_angles = 60 #30
i_0 = 10**5
sigma_data = 0.3
patch = 28
rad = patch // 2
ymin, ymax = rad, H - rad - 1
xmin, xmax = rad, W - rad - 1


### training parameters
T_ini = 80
T_max = 70
T_min = 70
decay_iters = 70
I = 10
tol_dx = 1e-3
patience_dx = 3
alpha = 2e-1/(75.776649475097666)**2 #1/(liptchitz constant)^2. .  53.595951080322266
beta1 = [8e-2, 8e-2] #[1e-2, 1e-2]
beta2 = [5e-2, 5e-2] #[5e-2, 5e-2] #[5e-4, 5e-3] 
alpha_img = 0.01
accelerator = Accelerator()
device = accelerator.device
n_train = 10
batch_size = 1
save_batch_id = 0
topk_digits = 5184
results_dir = "simpler_pipeline/Results/ExperimentsClassDeno/Experiments4/60_angles_sigma_0.3"


###################################################################### Data #####################################################################

# define CT geometry
vg = ts.volume(shape=(1,H,W))                                 # Volume
angles = np.linspace(0, np.pi, num_angles, endpoint=True)     # Angles
pg = ts.parallel(angles=angles, shape=(1, H))                 # Parallel beam structure
A = ts.operator(vg,pg)


A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)


# Import data
data = FashionMnistDataset(training = True,
                testing= False, 
                A = A, 
                dimension = (H,W), 
                num_angles = num_angles, 
                sigma = sigma_data,
                i_0 = i_0,
                cut = False,
                seed = seed,
                max_len = None,
                device = device)

#for reproducibility
g_split = torch.Generator()
g_split.manual_seed(seed)
labels = np.array([data[i]["label"] for i in range(len(data))])

# Split estratificado
train_idx, val_idx = train_test_split(
    np.arange(len(data)),
    train_size=n_train,
    stratify=labels,
    random_state=seed
)

train_data = Subset(data, train_idx)


# create dataloader
train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    generator=g_split)


###################################################################### Logger #####################################################################
# create logger for checkpoints
log_dir = "logs"
name = "60_angles_sigma_0.3_lines"

os.makedirs(log_dir, exist_ok=True)
handlers = [logging.StreamHandler()]
if accelerator.is_main_process:
    handlers.insert(0, logging.FileHandler(os.path.join(log_dir, f"{name}.log")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=handlers
)

logger = logging.getLogger(__name__)




#Check if the dataset is balanced
labels_bal = torch.tensor([train_data[i]["label"] for i in range(len(train_data))])
counts = torch.bincount(labels_bal, minlength=10)

if accelerator.is_main_process:
    logger.info("Balanced dataset label counts:")
    for d in range(10):
        logger.info(f"Digit {d}: {counts[d].item()}")


###################################################################### Trained score model ######################################################################
score = score_model(in_channels=1, n_feat=128)


checkpoint = torch.load(".pth", map_location="cpu")
alpha_t = checkpoint["alpha_t"]
beta_t = 1 - alpha_t

model_state = {
    k.replace("nn_model.", ""): v
    for k, v in checkpoint.items()
    if k.startswith("nn_model.")
}
score.load_state_dict(model_state)

# model in evaluation mode, we don't want to train it
score.eval() 
for p in score.parameters():
   p.requires_grad_(False)


# for using the score
t_input = t.float().view(batch_size, 1) / T_max
fixed_noise = torch.randn((batch_size, 1 , 28, 28), device=device)
context_mask = torch.zeros(batch_size, device=device) #we want to use the condition

score_model, train_dataloader = accelerator.prepare(score, train_dataloader)




def DPS(y):
    B = y.shape[0]
    device = y.device
    
    # initilize the image
    x = torch.randn((B, 1, 28, 28), device=device)

    # diffusion coefficients
    alpha_bar = torch.cumprod(alpha_t, dim=0)


    #DPS 
    for i in reversed(range(N)):

        if i > 0:
            alpha_bar_prev = alpha_bar[i - 1]
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar[i])

        # score value for x_i in time i
        x = x.detach().requires_grad_(True)
        score_value = score_model(x, i)

        # new image based on the diffusion process (clean image)
        x_0 = (alpha_bar[i])**(-0.5)*(x + (1-alpha_bar[i])*score_value)

        # random noise (same size as images)
        z = torch.randn((B, 1, 28, 28), device=device)

        # posterior mean
        x_1 = (((alpha_t[i])**(0.5))*(1-alpha_bar_prev)/(1-alpha_bar[i])) * x + (((alpha_bar_prev)**(0.5))*beta_t[i]/(1-alpha_bar[i])) * x_0 + sigmas[i]*z

        # Reconstruction step (make sure the image looks similar to the data)
        y_pred = A_torch(x_0)
        loss = torch.sum((y - y_pred)**2)
        grad = torch.autograd.grad(outputs=loss,inputs=x)[0]

        x = x_1 - zetas[i]*grad

    return x.detach()





# for loop for data in dataloader

recon_images = []
gt_images = []

for batch_id, batch in enumerate(train_dataloader):

    ground_truth_1 = batch["ground_truth_image"].to(device)
    input_data = batch["noisy_sinogram"].to(device)

    x = DPS(input_data)

    if accelerator.is_main_process and batch_id ==0: 
        for b in range(B):
            gt_images.append(ground_truth_1[b].detach().cpu())
            recon_images(x[b].detach().cpu())
    else: 
        break