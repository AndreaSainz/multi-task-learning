import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
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
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix

device = "cuda"

# Import trained mu(x_cropped|digit)
# Define flows
L = 2               #number of latent variables (z_l)
K = 24 #24              #how complex the invertible function is
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

checkpoint = torch.load("glow_mnist_checkpoint1.pt", map_location=device)
mu_model.load_state_dict(checkpoint["model_state_dict"])

mu_model.eval() 
for p in mu_model.parameters():
   p.requires_grad_(False)


############################################### DATA ##################################################

seed = 29072000
H = 100
W = 100
num_angles = 60
i_0 = 10**5
sigma_data = 0.5
n_train = 100
batch_size = 20
alpha = 1/(75.77664947509766)**2 #2e-1

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



gt_images = []
sirt_images = []
logx_ct = []
logx_gt = []
labels_list = []

def preprocess_for_nf(x):
    x = x.clamp(0.0, 1.0)
    x = x * (255.0 / 256.0)
    x = x + torch.rand_like(x) * (1.0 / 256.0)
    return x

for batch_id, batch in enumerate(train_dataloader):

    ground_truth_1 = batch["ground_truth_image"]
    ground_truth_2 = batch["label"]
    ground_truth_3 = batch["centre"]
    input_data = batch["noisy_sinogram"]
    labels_list.append(ground_truth_2.detach().cpu())

    B = input_data.shape[0]

    #### SIRT algorithm 
    x = torch.zeros(B, 1, H, W, device=input_data.device)
    
    #x_tres = ground_truth_1[0]
    for t in range(30):
        x_prev = x
        Ax = A_torch(x)
        r = input_data - Ax
        AT_r = AT_torch(r)
        step_data = alpha * AT_r
        x = x + step_data

    x = x.clamp(0,1)

    ##### Ct crop and GT crop
    crop_ct = torch.zeros(B, 1, 28, 28, device=device, dtype=x.dtype)
    crop_true = torch.zeros(B, 1, 28, 28, device=device, dtype=ground_truth_1.dtype)

    for b in range(B):
        cy = int(ground_truth_3[b, 0].item())
        cx = int(ground_truth_3[b, 1].item())
        crop_ct[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
        crop_true[b:b+1] = ground_truth_1[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
    
    crop_true = preprocess_for_nf(crop_true)
    crop_ct   = preprocess_for_nf(crop_ct)

    ##### probability of model
    with torch.no_grad():
        map_digit_ct = torch.full((B, 10), -torch.inf, device=input_data.device)
        map_digit_true = torch.full((B, 10), -torch.inf, device=input_data.device)
        for d in range(0,10):
            d_vec = torch.full((B,), d, dtype=torch.long, device=input_data.device) #it could be one hot or one digit for this model
            map_digit_ct[:, d] =  - mu_model(crop_ct, d_vec).view(B)
            map_digit_true[:, d] =  - mu_model(crop_true, d_vec).view(B)
            
    prob_ct = torch.softmax(map_digit_ct, dim=1)
    prob_true = torch.softmax(map_digit_true, dim=1)

    
    gt_images.append(crop_true)
    sirt_images.append(crop_ct)
    logx_ct.append(prob_ct)
    logx_gt.append(prob_true)


results = {
    "gt_crops": gt_images,     
    "sirt_crops": sirt_images,
    "prob_ct_digits": logx_ct,
    "prob_true_centers": logx_gt,
    "labels": labels_list,
    }

path_saving = "simpler_pipeline/results/nf_checks_results1.pt"
torch.save(results, path_saving)





def to_tensor(x):
    """Convert list-of-batches (or tensor) into one tensor."""
    if isinstance(x, torch.Tensor):
        return x
    # common case: list of tensors [batch][B,...]
    return torch.cat([t.detach().cpu() for t in x], dim=0)


def topk_accuracy(probs: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    """
    probs: [N,10], labels: [N]
    """
    topk = probs.topk(k, dim=1).indices  # [N,k]
    correct = (topk == labels.view(-1, 1)).any(dim=1).float().mean().item()
    return correct


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    probs: [N,10]
    returns: [N]
    """
    p = probs.clamp(min=eps)
    return -(p * p.log()).sum(dim=1)


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def plot_accuracy_bar(acc_gt, acc_ct, top2_gt, top2_ct, savepath="simpler_pipeline/results/accuracy_bar1.png"):
    labels = ["GT", "CT"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, [acc_gt, acc_ct], width, label="Top-1")
    plt.bar(x + width/2, [top2_gt, top2_ct], width, label="Top-2")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("NF accuracy: GT vs CT")
    plt.legend()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def plot_confusion_heatmap(cm, title, savepath="simpler_pipeline/results/confusion_heatmap1.png"):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def plot_entropy_hist(ent_gt, ent_ct, bins=40, savepath= "simpler_pipeline/results/entropy_hist1.png"):
    plt.figure()
    plt.hist(ent_gt.numpy(), bins=bins, alpha=0.6, label="GT")
    plt.hist(ent_ct.numpy(), bins=bins, alpha=0.6, label="CT")
    plt.xlabel("Entropy  H(p)")
    plt.ylabel("Count")
    plt.title("Entropy of p(d|x): GT vs CT")
    plt.legend()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


# ----------------------------
# MAIN: load your saved results
# ----------------------------
results = torch.load(path_saving, map_location="cpu")

# You MUST have labels saved. If you didn't, fix your saving code to include them.
# Expected:
# results["labels"] = list of label tensors [batch][B]
if "labels" not in results:
    raise KeyError(
        "Your results file does not contain labels. Save them as results['labels'] "
        "(e.g., append ground_truth_2 each batch) and re-run."
    )

probs_gt = to_tensor(results["prob_true_centers"])   # [N,10]
probs_ct = to_tensor(results["prob_ct_digits"])      # [N,10]
labels   = to_tensor(results["labels"]).long().view(-1)  # [N]

# Predictions
pred_gt = probs_gt.argmax(dim=1)
pred_ct = probs_ct.argmax(dim=1)

# 1) Accuracy bars (Top-1 + Top-2)
acc_gt  = topk_accuracy(probs_gt, labels, k=1)
acc_ct  = topk_accuracy(probs_ct, labels, k=1)
top2_gt = topk_accuracy(probs_gt, labels, k=2)
top2_ct = topk_accuracy(probs_ct, labels, k=2)

plot_accuracy_bar(acc_gt, acc_ct, top2_gt, top2_ct)

# 2) Confusion matrices (two heatmaps)
cm_gt = compute_confusion(labels.numpy(), pred_gt.numpy(), num_classes=10)
cm_ct = compute_confusion(labels.numpy(), pred_ct.numpy(), num_classes=10)

plot_confusion_heatmap(cm_gt, "Confusion matrix (GT crops)", savepath="simpler_pipeline/results/confusion_heatmap_gt1.png")
plot_confusion_heatmap(cm_ct, "Confusion matrix (CT crops)", savepath="simpler_pipeline/results/confusion_heatmap_ct1.png")

# 3) Entropy histograms
ent_gt = entropy_from_probs(probs_gt)  # [N]
ent_ct = entropy_from_probs(probs_ct)  # [N]

plot_entropy_hist(ent_gt, ent_ct)