from data_loader import MnistDataset
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import tomosipo as ts
import matplotlib.pyplot as plt
import torch

# global seed for reproducibility
seed = 29072000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#parameter values
batch_size = 1
training = True
testing = False
H = 100
W = 100
dimension = (H, W)
num_angles = 30
sigma = 0.01
standard_dose = True
cut = False
max_len = 2
device = torch.device("cuda:0")


# Create tomosipo volume and projection geometry
vg = ts.volume(shape=(1,H,W))                                                       # Volumen
angles = np.linspace(0, np.pi, num_angles, endpoint=True)                           # Angles
pg = ts.parallel(angles=angles, shape=(1, H))                                       # Parallel beam structure
A = ts.operator(vg,pg)     

# Call dataloader for training
train_data = MnistDataset(training, testing, A, dimension, num_angles, sigma, standard_dose, cut, seed , max_len ,  device)
g = torch.Generator() 
g.manual_seed(seed)
train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    generator=g)

for epoch in range(3):
    print(f"\n=== EPOCH {epoch} ===")
    for batch_id, batch in enumerate(train_dataloader):
        plt.imshow(batch["noisy_sinogram"][0].squeeze().cpu().numpy(), cmap="gray")
        plt.title(f"Sinogram {batch_id} epoch {epoch} Try 1")
        plt.savefig(f"sinogram_batch_{batch_id}_epoch_{epoch}_try1_30.png")

        plt.imshow(A.T(batch["noisy_sinogram"][0])[0].squeeze().cpu().numpy())
        plt.title(f" reconstructed Image {batch_id} epoch {epoch} Try 1")
        plt.savefig(f"reconstructed_image_batch_{batch_id}_epoch_{epoch}_try1_30.png")

        plt.imshow(batch["ground_truth_image"][0].squeeze().cpu().numpy())
        plt.title(f"Image {batch_id} epoch {epoch} try 1")
        plt.savefig(f"Image_batch_{batch_id}_epoch_{epoch}_try1_30.png")

        noisy = batch["noisy_sinogram"][0]          # quitamos dim de batch
        center = batch["center"][0]                 # primer (y único) centro

        print(
            f"Call 1: center={center}, "
            f"first_pixel_sino={noisy.view(-1)[0].item():.6f}"
        )