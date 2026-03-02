# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_loader import MnistDataset
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
import struct
import gzip
import torch
from torch.utils.data import Dataset
from PIL import Image


# class AddGaussianNoise:
#     def __init__(self, sigma):
#         self.sigma = sigma

#     def __call__(self, x):
#         x =  x + torch.randn_like(x) * self.sigma
#         return torch.clamp(x, 0.0, 1.0)


#     def __repr__(self):
#         return f"AddGaussianNoise(sigma={self.sigma})"

# Set up model

# Define flows
L = 2  #2 gaus
K = 32  #24 gaus
torch.manual_seed(0)

input_shape = (1, 28, 28)
n_dims = np.prod(input_shape)
channels = 1
hidden_channels = 256
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


# Construct flow model with the multiscale architecture
model = nf.MultiscaleFlow(q0, flows, merges)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)

#################################### For CT
def _open_idx(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


class CTMNIST(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.transform = transform

        # --- load images ---
        with _open_idx(images_path) as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051
            data = f.read()
            images = np.frombuffer(data, dtype=np.uint8)
            self.images = images.reshape(num, rows, cols)

        # --- load labels ---
        with _open_idx(labels_path) as f:
            magic, num = struct.unpack(">II", f.read(8))
            assert magic == 2049
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            self.labels = labels

        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx], mode="L")
        label = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, label


batch_size = 128

transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    nf.utils.Scale(255. / 256.),
    nf.utils.Jitter(1 / 256.)
])

train_data = CTMNIST(
    images_path="data/MNIST_CT/train-images-ct-idx3-ubyte",
    labels_path="data/MNIST_CT/train-labels-ct-idx1-ubyte",
    transform=transform,
)

test_data = CTMNIST(
    images_path="data/MNIST_CT/test-images-ct-idx3-ubyte",
    labels_path="data/MNIST_CT/test-labels-ct-idx1-ubyte",
    transform=transform,
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size
)






# # Prepare training data
# batch_size = 128

# transform = tv.transforms.Compose([
#     tv.transforms.ToTensor(),
#     # AddGaussianNoise(sigma=0.2),    
#     nf.utils.Scale(255. / 256.),
#     # nf.utils.Jitter(1 / 256.)
# ])
# train_data = tv.datasets.MNIST('datasets/', train=True,
#                                  download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
#                                            drop_last=True)

# test_data = tv.datasets.MNIST('datasets/', train=False,
#                                 download=True, transform=transform) 
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

train_iter = iter(train_loader)


# def batch_to_crop_ct(batch):
#     input_data = batch["noisy_sinogram"].to(device)
#     y = batch["label"].to(device).long()
#     centre = batch["centre"].to(device)

#     B = input_data.shape[0]   # NO uses B=20 fijo
#     x = torch.zeros(B, 1, H, W, device=device)

#     # SIRT sin grad (ahorra memoria y es lo que quieres)
#     with torch.no_grad():
#         for _ in range(20):
#             r = input_data - A_torch(x)
#             x = x + alpha * AT_torch(r)
#         x = x.clamp(0, 1)

#         crop_ct = torch.zeros(B, 1, 28, 28, device=device, dtype=x.dtype)
#         for b in range(B):
#             cy = int(centre[b, 0].item())
#             cx = int(centre[b, 1].item())
#             crop_ct[b] = x[b, :, cy-14:cy+14, cx-14:cx+14]

#     return crop_ct, y



# H = 100
# W = 100
# num_angles = 60
# i_0 = 10**5
# sigma_data = 0.5
# n_train = None
# alpha = 1/(40.08146667480469)**2 #2e-1

# # define CT geometry
# vg = ts.volume(shape=(1,H,W))                                 # Volume
# angles = np.linspace(0, np.pi, num_angles, endpoint=True)     # Angles
# pg = ts.parallel(angles=angles, shape=(1, H))                 # Parallel beam structure
# A = ts.operator(vg,pg)


# A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
# AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)


# # Import data
# data = MnistDataset(training = True,
#                 testing= False, 
#                 A = A, 
#                 dimension = (H,W), 
#                 num_angles = num_angles, 
#                 sigma = sigma_data,
#                 i_0 = i_0,
#                 cut = False,
#                 seed = 29072000,
#                 max_len = None,
#                 device = device)

# #for reproducibility
# g_split = torch.Generator()
# g_split.manual_seed(29072000)
# labels = np.array([data[i]["label"] for i in range(len(data))])

# # Split estratificado
# train_idx, val_idx = train_test_split(
#     np.arange(len(data)),
#     train_size=n_train,
#     stratify=labels,
#     random_state=29072000
#)

# train_data = Subset(data, train_idx)

# # create dataloader (ANTES)
# train_dataloader = DataLoader(
#     train_data,
#     batch_size=batch_size,
#     shuffle=True,
#     generator=g_split,
#     drop_last=True
# )
# train_iter = iter(train_dataloader)




# Train model
max_iter = 50000

loss_hist = np.array([])

optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

for i in tqdm(range(max_iter)):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    
    # try:
    #     batch = next(train_iter)
    # except StopIteration:
    #     train_iter = iter(train_dataloader)
    #     batch = next(train_iter)

    # t0=time.time()
    # x, y = batch_to_crop_ct(batch)
    # t1=time.time()



    optimizer.zero_grad(set_to_none=True)
    loss = model.forward_kld(x.to(device), y.to(device))
    #t2=time.time()
    
    #print("CT:", t1-t0, "flow:", t2-t1)

    if torch.isfinite(loss):
        loss.backward()
        optimizer.step()

    loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
    del(x, y, loss)

plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig("loss_curve_ct.png", dpi=300)



torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "glow_mnist_checkpoint_ct.pt")



# Model samples
num_sample = 10

with torch.no_grad():
    y = torch.arange(num_classes).repeat(num_sample).to(device)
    x, _ = model.sample(y=y)
    x_ = torch.clamp(x, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0)))
    plt.savefig("sample_nf_ct.png", dpi=300)


# test_data = Subset(data, val_idx)
# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

# n = 0
# bpd_cum = 0
# with torch.no_grad():
#     for batch in test_dataloader:
#         x, y = batch_to_crop_ct(batch)   # x es crop_ct
#         nll = model(x, y)                # nll por sample
#         nll_np = nll.detach().cpu().numpy()
#         bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
#         n += len(x) - np.sum(np.isnan(nll_np))

# print("Bits per dim:", bpd_cum / n)

# Get bits per dim
n = 0
bpd_cum = 0
with torch.no_grad():
    for x, y in iter(test_loader):
        nll = model(x.to(device), y.to(device))
        nll_np = nll.cpu().numpy() 
        bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
        n += len(x) - np.sum(np.isnan(nll_np))
        
    print('Bits per dim: ', bpd_cum / n)



""".    without gaussian noise
(multi_task) as3628@holly-b:~/multi-task-learning$ CUDA_VISIBLE_DEVICES=1 python3 -m simpler_pipeline.nf
/home/as3628/.local/lib/python3.10/site-packages/torch/_tensor.py:857: UserWarning: torch.lu is deprecated in favor of torch.linalg.lu_factor / torch.linalg.lu_factor_ex and will be removed in a future PyTorch release.
LU, pivots = torch.lu(A, compute_pivots)
should be replaced with
LU, pivots = torch.linalg.lu_factor(A, compute_pivots)
and
LU, pivots, info = torch.lu(A, compute_pivots, get_infos=True)
should be replaced with
LU, pivots, info = torch.linalg.lu_factor_ex(A, compute_pivots) (Triggered internally at ../aten/src/ATen/native/BatchLinearAlgebra.cpp:1990.)
  LU, pivots, infos = torch._lu_with_info(
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [2:21:10<00:00,  5.90it/s]
Bits per dim:  1.2682499378204346
"""