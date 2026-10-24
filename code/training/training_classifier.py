import os
from dataclasses import dataclass
import json
import torch
import math
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from code.models.classifier import CNNClassifier
from code.models.denoiser import ResUNetDenoiser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)


# Change training options here.
CONFIG = {
    "model_name": "cnn_classifier",
    "dataset": "MNIST",                 # MNIST | FashionMNIST
    "save_dir": "./checkpoints/classifier",
    "batch_size": 128,
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 1,
    "seed": 42,
    "device": "cuda",                   # falls back to CPU if unavailable
    "noise_aware": False,
    "input": {
        "mode": "clean",               # clean | gaussian_fixed | gaussian_variable | sirt
        "sigma": 0.15,                  # gaussian_fixed
        "sigma_range": [0.0, 0.3],      # gaussian_variable; sampled per image
        "sirt": {
            "operator": "",       # e.g. my_project.sirt:reconstruct
            "alpha": 1.0,
            "steps": 50,
            "reconstruction_steps": [10, 25, 50],  # [] means final only
            "output_step": 50,           # used when steps list is empty
        },
    },
}

# PARAMETERS
model_name = "cnn_classifier_clean1yrp"
batch_size = 128
epochs = 30
lr= 1e-3
weight_decay = 1e-4
num_workers = 1
device= "cuda" 
save_dir = "./checkpoints/classifier"
os.makedirs(save_dir, exist_ok=True)
### data
accelerator = Accelerator()
device = accelerator.device
n_train = 10
batch_size = 1
H = 100
W = 100
num_angles = 60
angle_range = [0, np.pi]
i_0 = 10**5
sigma_data = 0.3

# logger and file names
log_dir = "logs"
name = "sirt_check"
os.makedirs(log_dir, exist_ok=True)

# saving directory
results_dir = "experiments/Results/Checks/SIRT"
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



def training_dataloader(mode):

    if mode == "clean":
        #train and validation dataloaders
        train_dataloader, val_dataloader = import_data(True, True, 48000, None, A, num_angles, sigma_data, i_0, H, W, batch_size, seed, logger, accelerator, device)
    
    elif mode == ""




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
    x, y = batch
    x = x.to(device)
    y = y.to(device)

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

    for batch in loader:
        optimizer.zero_grad()
        loss, correct, n = classification_step(model, batch, loss_fn)
        total_correct += correct

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * n
        total_samples += n

    avg_loss = total_loss / total_samples
    avg_metric = total_correct / total_samples

    return avg_loss, avg_metric


# EVAL LOOP
@torch.no_grad()
def evaluate_classifier(model, loader, loss_fn):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

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
        train_loss, train_metric = train_one_epoch(model, train_loader, optimizer, loss_fn)

        # validation step
        val_loss, val_acc = evaluate_classifier(model, test_loader, loss_fn)

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
    # MODEL
    model = CNNClassifier()
    train(model)