import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_loader import MnistDataset
from code.models.score_unet import ContextUnet, ddpm_schedules
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
import torch, gc




###################################################################### Parameters #####################################################################

### data
accelerator = Accelerator()
device = accelerator.device
seed = 29072000
H = 100
W = 100
num_angles = 60
i_0 = 10**5
sigma_data = 0.3


### training parameters
R_ini = 80
R = 70
I = 10
tol_dx = 1e-3
patience_dx = 3
alpha = 2e-1/(75.77664947509766)**2 #1/(liptchitz constant)^2 2e-1/(53.595951080322266)**2 #
gamma = [1e-3, 1e-3]
n_train = 10
batch_size = 1
t = torch.full((batch_size,), 5, device=device) #time step in the denoiser
save_batch_id = 0
topk_digits = 5184
results_dir = "simpler_pipeline/results/ExperimentsScore/Experiments4/60_angles_sigma_0.3"


###################################################################### Data #####################################################################

# define CT geometry
vg = ts.volume(shape=(1,H,W))                                 # Volume
angles = np.linspace(0, np.pi, num_angles, endpoint=True)     # Angles
pg = ts.parallel(angles=angles, shape=(1, H))                 # Parallel beam structure
A = ts.operator(vg,pg)


A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)


# Import data
data = MnistDataset(training = True,
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
name = "conditional_score_60_angles_sigma_0.3"
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


###################################################################### Trained NF Model #####################################################################

# Import trained mu(x_cropped|digit)
# Define flows
L = 2               #number of latent variables (z_l)
K = 32              #how complex the invertible function is
torch.manual_seed(0)

input_shape = (1, 28, 28)
n_dims = np.prod(input_shape)
channels = 1
hidden_channels = 256           #how large are CNN inside GlowBlocks, so s() and t()
split_mode = 'channel'
scale = True
num_classes = 10

# Set up flows, distributions and merge operations
q0 = []
merges = []
flows = []
for i in range(L):
    flows_ = []
    for j in range(K):
        flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                     split_mode=split_mode, scale=scale)]
    flows_ += [nf.flows.Squeeze()]
    flows += [flows_]
    if i > 0:
        merges += [nf.flows.Merge()]
        latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                        input_shape[2] // 2 ** (L - i))
    else:
        latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                        input_shape[2] // 2 ** L)
    q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

mu_model = nf.MultiscaleFlow(q0, flows, merges)
mu_model = mu_model.to(device)

checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/normalizing flow/glow_mnist_checkpoint_clean.pt", map_location=device)
mu_model.load_state_dict(checkpoint["model_state_dict"])

mu_model.eval() 
for p in mu_model.parameters():
   p.requires_grad_(False)

