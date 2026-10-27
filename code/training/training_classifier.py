import os
import json
import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from tomosipo.torch_support import to_autograd
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_import import import_data
from torchvision import datasets, transforms
from code.models.classifier import CNNClassifier
from code.utils.logger import setup_logger
from code.ct_recon.operators import ct_operator
from code.ct_recon.SIRT import normalization_matrices, sirt_method
from code.utils.image_cropping import extract_crops, insert_crop
from accelerate import Accelerator
from pathlib import Path
import matplotlib
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SEED
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


# PARAMETERS
model_name = "cnn_classifier_mnist_sirt_time"
batch_size = 128
epochs = 1
lr= 1e-3
weight_decay = 1e-4
accelerator = Accelerator()
num_workers = 0
device= "cuda" 
noise_aware = False
save_dir = "./checkpoints/classifier"
os.makedirs(save_dir, exist_ok=True)

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
sirt_iterations = 10
sirt_alpha= 10**4/(75.77664947509766)**2


# logger and file names
log_dir = "logs"
name = "classifier_training"
os.makedirs(log_dir, exist_ok=True)

# saving directory
results_dir = "multi-task-learning/checkpoints/classifier"
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


def get_model_input(batch):
    """
    Returns:
        x: input image, shape (B, C, H, W)
        y: classification labels, shape (B,)
        noise_level: optional conditioning value, shape (B, 1)
    """

    y = batch["label"].long().to(device)

    # Select input image
    if noisy_images:
        x = batch["noisy_ground_truth_image"]
        noise_level = batch["sigma"].float().to(device)

    elif sirt_images:
        # List of tensors: one reconstruction per iteration.
        reconstructions = batch["sirt_reconstructions"]


        # Each tensor has shape (B, H, W) and we need (B,1,H,W).
        x = torch.cat(
        [
            reconstruction.float().to(device).unsqueeze(1)
            if reconstruction.ndim == 3
            else reconstruction.float().to(device)
            for reconstruction in reconstructions
        ],
        dim=0)

        # Repeat labels for each reconstruction.
        num_reconstructions = len(reconstructions)
        y = y.repeat(num_reconstructions)

        noise_level = None

        if noise_aware:
            iteration_levels = batch["iteration_normalized"]

            levels = torch.stack([ torch.as_tensor(level, dtype=torch.float32, device=device) for level in iteration_levels])
            noise_level = levels.reshape(-1, 1)

    else:
        x = batch["ground_truth_small"]
        noise_level = None


    x = x.float().to(device)
    return x, y, noise_level




# UTILS
def save_training_plots(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_accuracy.png"))
    plt.close()


def save_history(history):
    with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=4)



# TRAINING STEP
def classification_step(model, batch, loss_fn):
    t0 = time.perf_counter()
    x, y, noise_level = get_model_input(batch)
    t1 = time.perf_counter()

    print(f"Tiempo get_model_input: {t1 - t0:.6f} segundos")

    if noise_aware:
        logits = model(x, noise_level=noise_level)
    else:
        logits = model(x)

    loss = loss_fn(logits, y)

    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()

    return loss, correct, y.size(0)


# TRAIN LOOP
def train_one_epoch(model, loader, optimizer, loss_fn):
    
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = iter(loader)

    for batch_idx in range(len(loader)):
        t0 = time.perf_counter()
        batch = next(iterator)
        t1 = time.perf_counter()

        print(f"Data loading: {t1 - t0:.6f} segundos")

        optimizer.zero_grad()
        loss, correct, n = classification_step(model, batch, loss_fn)
        total_correct += correct

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()
        t3 = time.perf_counter()

        optimizer.step()

        torch.cuda.synchronize()
        t4 = time.perf_counter()

        print(f"Backward: {t3 - t2:.6f} segundos")
        print(f"Optimizer: {t4 - t3:.6f} segundos")

        total_loss += loss.item() * n
        total_samples += n

    avg_loss = total_loss / total_samples
    avg_metric = total_correct / total_samples

    t1 = time.perf_counter()

    print(f"Tiempo train_one_epoch: {t1 - t0:.6f} segundos")
    return avg_loss, avg_metric


# EVAL LOOP
@torch.no_grad()
def evaluate_classifier(model, loader, loss_fn):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:

        x, y, noise_level = get_model_input(batch)

        if noise_aware:
            logits = model(x, noise_level=noise_level)
        else:
            logits = model(x)

        loss = loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()

        n = y.size(0)
        total_loss += loss.item() * n
        total_correct += correct
        total_samples += n

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


# TRAIN FUNCTION
def train(model):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay= weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_metric = 0

    for epoch in range(epochs):

        # trainig step
        train_loss, train_metric = train_one_epoch(model, train_dataloader, optimizer, loss_fn)

        # validation step
        val_loss, val_acc = evaluate_classifier(model, val_dataloader, loss_fn)

        # scheduler step
        scheduler.step()

        # save metrics in history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_metric)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)


        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_metric:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
        )

        # save checkpoint 
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler" :  scheduler.state_dict(),
            "epoch": epoch}, os.path.join(save_dir, f"{model_name}_last.pt"))

        # save best model comparing the accuracy in the validation set
        if val_acc > best_metric :
            best_metric = val_acc

            torch.save( model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pt"))

    # save trining history
    save_history(history)

    # training plots
    save_training_plots(history)

    print("Training finished")



# MAIN
if __name__ == "__main__":

    # Training configurations
    CONFIG = {
        # General
        "model_name": model_name,
        "dataset": "MNIST",             # MNIST/FASHIONMNIST
        "save_dir": str(save_dir),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "device": str(device),

        # Model
        "noise_aware": noise_aware,

        # Dataset
        "n_train": n_train,
        "H": H,
        "W": W,

        # CT geometry
        "ct_geometry": "parallel",
        "num_angles": num_angles,
        "angle_range": angle_range,
        "i_0": i_0,

        # Noise
        "sigma_data": sigma_data,

        # Input configuration
        "input": {
            "noisy_images": noisy_images,
            "sigma_image": sigma_image,
            "sirt_images": sirt_images,
            "sirt_iterations": sirt_iterations,
            "sirt_alpha": sirt_alpha,
        },

        # Fine-tuning
        "fine_tuning": {
            "enabled": False,
            "pretrained_checkpoint": None,
            "parameters": {
                "learning_rate": None,
                "epochs": None,
                "freeze_backbone": False,
                "unfreeze_layers": None,
            },
        },
    }

    # Save configuration
    config_path = Path(save_dir) / f"{model_name}_config.json"
    

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4, ensure_ascii=False)

    print(f"Configuration saved to: {config_path}")

    # MODEL
    model = CNNClassifier(noise_aware=noise_aware).to(device)
    train(model)