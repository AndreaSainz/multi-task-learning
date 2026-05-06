import os
from dataclasses import dataclass
import json
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from code.models.classifier import CNNClassifier
from code.models.denoiser import ResUNetDenoiser
import numpy as np
import random 

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

# parameters
model_name = "dncnn"
batch_size = 128
epochs = 30
lr= 1e-3
weight_decay = 1e-4
num_workers = 2
noise_std_max = 0.4
device= "cuda" 
save_dir = "./checkpoints/denoiser"
os.makedirs(save_dir, exist_ok=True)

# Data
transform = transforms.ToTensor()

train_dataset = datasets.MNIST( root="./data", train=True, download=True, transform=transform )
test_dataset = datasets.MNIST( root="./data", train=False, download=True, transform=transform )

train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader( test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)




# UTILS
def add_noise(x, sigma_max=0.4):
    B = x.shape[0]
    sigma = torch.rand(B, 1, 1, 1, device=x.device) * sigma_max
    noise = sigma * torch.randn_like(x)
    x_noisy = torch.clamp(x + noise, 0.0, 1.0)
    return x_noisy, sigma

def compute_psnr(x_hat, x, data_range=1.0):
    mse = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3))
    psnr = 10 * torch.log10((data_range ** 2) / torch.clamp(mse, min=1e-12))
    return psnr.mean().item()

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
    plt.plot(epochs, history["train_psnr"], label="train_psnr")
    plt.plot(epochs, history["val_psnr"], label="val_psnr")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title(f"{model_name} - PSNR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_psnr.png"))
    plt.close()

def save_history(history):
    with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=4)




# DATA
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)



# TRAINING STEPS

def denoising_step(model, batch, device, criterion, noise_std):

    x_clean, _ = batch
    x_clean = x_clean.to(device)

    x_noisy, _ = add_noise(x_clean, noise_std)

    x_denoised = model(x_noisy)

    loss = criterion(x_denoised, x_clean)
    psnr = compute_psnr(x_denoised.detach(), x_clean.detach())

    return loss, psnr, x_clean.size(0)



# TRAIN LOOP

def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_metric = 0.0
    total_samples = 0

    for batch in loader:
        optimizer.zero_grad()

        loss, psnr, n = denoising_step(model, batch, device, criterion, noise_std_max)
        total_metric += psnr * n

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * n
        total_samples += n

    avg_loss = total_loss / total_samples
    avg_metric = total_metric / total_samples

    return avg_loss, avg_metric



# EVAL LOOP

@torch.no_grad()
def evaluate_denoiser(model, loader, device,criterion):
    model.eval()

    total_loss = 0.0
    total_metric = 0.0
    total_samples = 0

    for batch in loader:
        loss, psnr, n = denoising_step(model, batch, device, criterion, noise_std_max )

        total_loss += loss.item() * n
        total_metric += psnr * n
        total_samples += n

    avg_loss = total_loss / total_samples
    avg_psnr = total_metric / total_samples

    return avg_loss, avg_psnr



# TRAIN FUNCTION

def train(model):

    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_psnr": [],
        "val_psnr": []
    }

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW( model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_metric = 0

    for epoch in range(config.epochs):
        # training
        train_loss, train_metric = train_one_epoch( model, train_loader, optimizer, config.device, config, criterion)

        # validation
        val_loss, val_psnr = evaluate_denoiser(model, test_loader, config.device, config, criterion)

        # scheduler step
        scheduler.step()

        # saving training history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_psnr"].append(train_metric)
        history["val_psnr"].append(val_psnr)

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_psnr={train_metric:.2f} | val_psnr={val_psnr:.2f}"
        )

        # save checkpoint 
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler" :  scheduler.state_dict(),
            "epoch": epoch}, os.path.join(save_dir, f"{model_name}_last.pt"))


        if  val_loss < best_metric
            best_metric = metric

            torch.save(model.state_dict(), os.path.join(config.save_dir, f"{model_name}_best.pt"))

    # save trining history
    save_history(history)

    # training plots
    save_training_plots(historyg)
    print("Training finished")



# MAIN
if __name__ == "__main__":

    #denoiser
    model = ResUNetDenoiser()
    train(model)