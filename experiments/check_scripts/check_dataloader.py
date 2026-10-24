from code.dataloader.data_loader import MnistDataset
from code.ct_recon.SIRT import normalization_matrices, sirt_method
import matplotlib.pyplot as plt
import os
import tomosipo as ts
import torch
import torch.nn.functional as F
from tomosipo.torch_support import to_autograd
import numpy as np
from torch.utils.data import DataLoader, Subset


# Parameters
seed = 29072000
H = 100
W = 100
num_angles = 30 #60
i_0 = 10**5
sigma_data = 0.3
results_dir = "simpler_pipeline/Results/Dataloader"
os.makedirs(results_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 1/(53.595951080322266)**2


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


train_data = Subset(data, [0,1,2])


# create dataloader
train_dataloader = DataLoader(
    train_data,
    batch_size=1)


# creating a log file
log_file = os.path.join(results_dir, "dataset_info.txt")
with open(log_file, "w") as f:

    f.write("batch_id\tcentre\tdigit\n")


# Normalization matrices for SIRT
M , D = normalization_matrices(A_torch, AT_torch, 1, H, W, num_angles, device)


for batch_id, batch in enumerate(train_dataloader):

    mnist_image = batch["mnist_image"].cpu().squeeze()
    full_image = batch["ground_truth_image"].cpu().squeeze()
    label = batch["label"].cpu().item()
    centre = batch["centre"].squeeze().cpu().tolist()
    noisy_sinogram = batch["noisy_sinogram"].cpu().squeeze()

    x = torch.zeros(1, 1, H, W, device=device)
    x_new, _ = sirt_method(A_torch, AT_torch, x, noisy_sinogram.to(device) , D, M, alpha)


    fig, ax = plt.subplots()
    im = ax.imshow(mnist_image.squeeze(), cmap="gray")
    ax.set_title("MNIST")
    ax.axis("off")
    fig.colorbar(im)
    plt.savefig(os.path.join(results_dir, f"MNIST_{batch_id}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()

    im = ax.imshow(full_image, cmap="gray")
    ax.set_title("Full Image")
    ax.axis("off")
    fig.colorbar(im)
    plt.savefig(os.path.join(results_dir, f"full_image_{batch_id}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()

    im = ax.imshow(noisy_sinogram, cmap="gray")
    ax.set_title("Noisy sinogram (30 angles)")
    ax.axis("off")
    fig.colorbar(im)
    plt.savefig(os.path.join(results_dir, f"sinogram_30_{batch_id}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.imshow(x_new.cpu().squeeze(), cmap="gray")
    ax.set_title("Reconstruction")
    ax.axis("off")
    fig.colorbar(im)
    plt.savefig(os.path.join(results_dir, f"reconstruction_{batch_id}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(log_file, "a") as f:
        f.write(f"{batch_id}\t{centre}\t{label}\n")