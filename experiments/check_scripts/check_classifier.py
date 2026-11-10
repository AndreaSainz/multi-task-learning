import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import json
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_import import import_data
from code.ct_recon.operators import ct_operator
from code.ct_recon.SIRT import normalization_matrices, sirt_method
from code.models.classifier import CNNClassifier
from code.utils.image_cropping import extract_crops, insert_crop
from code.dataloader.data_loader import FashionMnistDataset
from code.utils.plots import images_plots, psnr_plots, ssim_plots, save_confusion_matrix
from code.utils.logger import setup_logger
import matplotlib
import logging
from sklearn.metrics import confusion_matrix
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

## seeds
seed = 29072000
torch.manual_seed(0)

# Model name
model_name = "cnn_classifier_mnist_noisy_noiseaware_best"

# Directories
checkpoint_dir = Path("checkpoints/classifier")

# Paths
checkpoint_path = checkpoint_dir / f"{model_name}.pt"
config_path = checkpoint_dir / f"{model_name.replace('_best', '')}_config.json"

# Load configuration
with open(config_path, "r") as f:
    config = json.load(f)

# Load checkpoint
checkpoint = torch.load(checkpoint_path,map_location="cpu")


### data
accelerator = Accelerator()
device = accelerator.device
batch_size = 50
H = config["H"]
W = config["W"]
num_angles = config["num_angles"]
i_0 = config["i_0"]
sigma_data = config["sigma_data"]
angle_range = config["angle_range"]
noise_aware = config["noise_aware"]



# logger and file names
log_dir = "logs"
name = "cnn_classifier_mnist_noisy_noiseaware_best_check"
os.makedirs(log_dir, exist_ok=True)

# saving directory
results_dir = "experiments/Results/Checks/Classifier"
os.makedirs(results_dir, exist_ok=True)

## logger
logger = setup_logger(log_dir, name, accelerator = accelerator)

# SIRT 
sirt_alpha = 10**4/(75.77664947509766)**2
sirt_iterations = 20


# define CT geometry
A = ct_operator(H, W, angle_range, num_angles, "parallel")
A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)



# create pytorch test dataloader
# Import data with curstom dataloader 
data = FashionMnistDataset(training = False, testing= True, A = A, dimension = (H,W), 
    num_angles = num_angles, sigma = sigma_data,i_0 = i_0, sirt_images = True, 
    sirt_iterations = sirt_iterations, sirt_alpha = sirt_alpha,seed = seed,device = device)

    
test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)



# LOAD CLASSIFIER 
class_model = CNNClassifier(noise_aware=noise_aware)

if isinstance(checkpoint, dict) and "model" in checkpoint:
    class_model.load_state_dict(checkpoint["model"])
else:
    class_model.load_state_dict(checkpoint)

class_model.eval() 
for p in class_model.parameters():
   p.requires_grad_(False)

class_model = class_model.to("cuda")



y_true_all = []
y_pred_clean_all = []
y_pred_sirt_all = []

#for loop
for batch_id, batch in enumerate(test_dataloader):

    mnist_clean = batch["ground_truth_small"].to(device)
    mnist_sirt = batch["sirt_reconstructions"][-1].unsqueeze(1).to(device)
    labels = batch["label"].to(device)
    #                                                                                                                                                                                                          noise = batch["iteration_normalized"].to(device)
   
    if noise_aware:
        residual = mnist_sirt - mnist_clean
        sigma_sirt = residual.flatten(1).std(dim=1, unbiased=False)
        
        noise_channel = 0.0 * torch.ones_like(mnist_clean[:, :1, :1, :1]).to(device)
    
        # test classifier
        logits_img_true = class_model(mnist_clean,  noise_level=noise_channel)
        digit_img_true = torch.softmax(logits_img_true, dim=1).argmax(dim=1)

        logits_sirt = class_model(mnist_sirt, noise_level = sigma_sirt)
        digit_sirt = torch.softmax(logits_sirt, dim=1).argmax(dim=1)

    else:
        # test classifier
        logits_img_true = class_model(mnist_clean)
        digit_img_true = torch.softmax(logits_img_true, dim=1).argmax(dim=1)

        logits_sirt = class_model(mnist_sirt)
        digit_sirt = torch.softmax(logits_sirt, dim=1).argmax(dim=1)

    # save results
    y_true_all.extend(labels.cpu().numpy().tolist())
    y_pred_clean_all.extend(digit_img_true.cpu().numpy().tolist())
    y_pred_sirt_all.extend(digit_sirt.cpu().numpy().tolist())

    # checkpoint of batches
    if (batch_id + 1) % 10 == 0:
        print(f"Processed {batch_id + 1} batches")



# Confusion Matrices
y_true = np.asarray(y_true_all)
y_pred_clean = np.asarray(y_pred_clean_all)
y_pred_sirt = np.asarray(y_pred_sirt_all)
digits = list(range(10))

cm_clean = confusion_matrix(y_true, y_pred_clean, labels=digits)

cm_sirt = confusion_matrix(y_true, y_pred_sirt, labels=digits)


save_confusion_matrix(cm_clean,digits, "Clean MNIST", f"{name}_clean.png", results_dir)

save_confusion_matrix(cm_sirt,digits, "SIRT reconstruction", f"{name}_SIRT.png", results_dir)


# Global metrics
results = []

for method, y_pred in [("Clean MNIST", y_pred_clean), ("SIRT reconstruction", y_pred_sirt)]:
    results.append({
        "Method": method,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(y_true, y_pred, labels=digits, average="macro", zero_division=0),
        "Recall_macro": recall_score( y_true, y_pred, labels=digits, average="macro", zero_division=0),
        "F1_macro": f1_score( y_true, y_pred, labels=digits, average="macro", zero_division=0),
        "Precision_weighted": precision_score( y_true, y_pred, average="weighted", zero_division=0),
        "Recall_weighted": recall_score( y_true, y_pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score( y_true, y_pred, average="weighted", zero_division=0)
    })


# Save metrics
df_metrics = pd.DataFrame(results)
df_metrics.to_csv( f"{results_dir}/{name}_metrics.csv", index=False)












    





# for i in range(10):
#    x = torch.rand(1, 1, 28, 28, device="cuda")

#    fig, axes = plt.subplots(1, 1, figsize=(20, 4))

#    im = axes.imshow(x[0,0].detach().to("cpu"), cmap="gray", vmin=0, vmax=1)
#    axes.set_title("SIRT only")
#    axes.axis("off")

#    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
#    fig.colorbar(im, cax=cbar_ax)

#    fig.subplots_adjust(right=0.9)
#    plt.savefig(f"/home/as3628/multi-task-learning/simpler_pipeline/Results/{i}_uniform", dpi=300, bbox_inches="tight")
#    plt.close(fig)

#    logits = class_model(x)
#    prob = torch.softmax(logits, dim=1)

#    print(prob)
