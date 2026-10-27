import torch
import torch.nn.functional as F
import pandas as pd
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

### data
accelerator = Accelerator()
device = accelerator.device
batch_size = 50
H = 100
W = 100
num_angles = 60
angle_range = [0, np.pi]
i_0 = 10**5
sigma_data = 0.3

# logger and file names
log_dir = "logs"
name = "cnn_classifier_mnist_clean_best"
os.makedirs(log_dir, exist_ok=True)

# saving directory
results_dir = "experiments/Results/Checks/Classifier"
os.makedirs(results_dir, exist_ok=True)

## logger
logger = setup_logger(log_dir, name, accelerator = accelerator)

# SIRT 
alpha = 10**4/(75.77664947509766)**2
sirt_iter = 20


# define CT geometry
A = ct_operator(H, W, angle_range, num_angles, "parallel")
A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)



# create pytorch test dataloader
# Import data with curstom dataloader 
data = FashionMnistDataset(training = False,
               testing= True, 
               A = A, 
               dimension = (H,W), 
               num_angles = num_angles, 
               sigma = sigma_data,
               i_0 = i_0,
               cut = False,
               seed = seed,
               max_len = None,
               device = device)

test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)



# LOAD CLASSIFIER 
class_model = CNNClassifier()

checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/classifier/cnn_classifier_mnist_clean_best.pt", map_location="cpu")

if isinstance(checkpoint, dict) and "model" in checkpoint:
    class_model.load_state_dict(checkpoint["model"])
else:
    class_model.load_state_dict(checkpoint)

class_model.eval() 
for p in class_model.parameters():
   p.requires_grad_(False)

class_model = class_model.to("cuda")


#normalization matrices
M , D = normalization_matrices(A_torch, AT_torch, batch_size, H, W, num_angles, device)


y_true_all = []
y_pred_clean_all = []
y_pred_sirt_all = []

#for loop
for batch_id, batch in enumerate(test_dataloader):

   ground_truth_1 = batch["ground_truth_image"]
   labels = batch["label"]
   ground_truth_3 = batch["centre"]
   input_data = batch["noisy_sinogram"]

   # CT reconstruction
   x = torch.zeros(batch_size, 1, H, W, device=device)

   for i in range(sirt_iter):
      x, _ = sirt_method(A_torch, AT_torch, x, input_data, D, M, alpha)


   # extracting crops
   crop_x = extract_crops(ground_truth_1, ground_truth_3).detach().requires_grad_(True)
   crop_x_sirt = extract_crops(x, ground_truth_3).detach().requires_grad_(True)


   # test classifier
   logits_img_true = class_model(crop_x)
   digit_img_true = torch.softmax(logits_img_true, dim=1).argmax(dim=1)

   logits_sirt = class_model(crop_x_sirt)
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
