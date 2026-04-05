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


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    model_name: str = "cnn_classifier"
    task: str = "classification"     
    batch_size: int = 128
    epochs: int = 13
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./checkpoints/classifier"
    noise_std_max: float = 0.4


# ============================================================
# UTILS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

def save_training_plots(history, config):
    os.makedirs(config.save_dir, exist_ok=True)

    if config.task == "classification":
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure()
        plt.plot(epochs, history["train_loss"], label="train_loss")
        plt.plot(epochs, history["val_clean_loss"], label="val_clean_loss")
        plt.plot(epochs, history["val_noisy_loss"], label="val_noisy_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{config.model_name} - Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(config.save_dir, f"{config.model_name}_loss.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, history["train_acc"], label="train_acc")
        plt.plot(epochs, history["val_clean_acc"], label="val_clean_acc")
        plt.plot(epochs, history["val_noisy_acc"], label="val_noisy_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{config.model_name} - Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(config.save_dir, f"{config.model_name}_accuracy.png"))
        plt.close()

    elif config.task == "denoising":
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure()
        plt.plot(epochs, history["train_loss"], label="train_loss")
        plt.plot(epochs, history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{config.model_name} - Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(config.save_dir, f"{config.model_name}_loss.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, history["train_psnr"], label="train_psnr")
        plt.plot(epochs, history["val_psnr"], label="val_psnr")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.title(f"{config.model_name} - PSNR")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(config.save_dir, f"{config.model_name}_psnr.png"))
        plt.close()

def save_history(history, config):
    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, f"{config.model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=4)

def attenuate_full_digit( x, attenuation_range=(0.35, 0.85), foreground_threshold=0.0 ):
    """
    Atenúa TODO el foreground del dígito.
    x: [B,1,H,W] en [0,1]
    """
    x_out = x.clone()
    device = x.device
    B = x.shape[0]

    # un factor distinto por imagen
    factors = torch.empty(B, 1, 1, 1, device=device).uniform_(
        attenuation_range[0], attenuation_range[1]
    )

    fg_mask = (x > foreground_threshold).float()

    # fondo igual, foreground atenuado
    x_out = x * (1.0 - fg_mask) + x * factors * fg_mask
    return x_out

def corrupt_faint_digit(
    x,
    p_clean=0.5,
    attenuation_range=(0.35, 0.85),
    foreground_threshold=0.05
):
    """
    Devuelve:
    - imagen limpia
    - imagen con el dígito más tenue
    """
    r = torch.rand(1, device=x.device).item()

    if r < p_clean:
        return x
    else:
        return attenuate_full_digit( x, attenuation_range=attenuation_range, foreground_threshold=foreground_threshold )

def attenuate_digit_structure( x, num_patches_range=(1, 3), patch_size_range=(4, 10), attenuation_range=(0.2, 0.7), foreground_threshold=0.0 ):
    """
    x: [B,1,H,W] en [0,1]

    Atenúa regiones del foreground del dígito sin borrar completamente
    su estructura.
    """
    x_out = x.clone()
    B, C, H, W = x.shape
    device = x.device

    for b in range(B):
        fg = (x[b, 0] > foreground_threshold).nonzero(as_tuple=False)

        if fg.numel() == 0:
            continue

        num_patches = torch.randint(
            num_patches_range[0],
            num_patches_range[1] + 1,
            (1,),
            device=device
        ).item()

        for _ in range(num_patches):
            idx = torch.randint(0, fg.shape[0], (1,), device=device).item()
            cy, cx = fg[idx].tolist()

            patch_size = torch.randint(
                patch_size_range[0],
                patch_size_range[1] + 1,
                (1,),
                device=device
            ).item()

            half = patch_size // 2
            y0, y1 = max(0, cy - half), min(H, cy + half + 1)
            x0, x1 = max(0, cx - half), min(W, cx + half + 1)

            factor = torch.empty(1, device=device).uniform_(
                attenuation_range[0], attenuation_range[1]
            ).item()

            patch = x_out[b:b+1, :, y0:y1, x0:x1]
            fg_patch = (patch > foreground_threshold).float()

            # atenúa solo el trazo
            x_out[b:b+1, :, y0:y1, x0:x1] = patch * (1 - fg_patch) + patch * factor * fg_patch

    return x_out

def corrupt_ct_like( x, sigma_max=0.4, p_clean=0.25, p_gaussian=0.25, p_attenuation=0.25, p_both=0.2 ):
    """
    Devuelve una versión corrupta de x con una mezcla de:
    - limpio
    - ruido gaussiano
    - atenuación estructural
    - atenuación estructural + ruido
    """
    r = torch.rand(1, device=x.device).item()

    if r < p_clean:
        return x

    elif r < p_clean + p_gaussian:
        x, _ = add_noise(x, sigma_max)
        return x

    elif r < p_clean + p_gaussian + p_attenuation:
        x = attenuate_digit_structure(x)
        return x

    else:
        x = attenuate_digit_structure(x)
        x, _ = add_noise(x, sigma_max)
        return x

# ============================================================
# DATA
# ============================================================

def get_mnist_loaders(batch_size, num_workers):

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

    return train_loader, test_loader


# ============================================================
# TRAINING STEPS
# ============================================================

def classification_step(model, batch, device, criterion, sigma_max, add_noise_flag=True):
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    if add_noise_flag:
        x = corrupt_faint_digit( x, p_clean=0.5, attenuation_range=(0.35, 0.85), foreground_threshold=0.0 )
    logits = model(x)
    loss = criterion(logits, y)

    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()

    return loss, correct, y.size(0)



def denoising_step(model, batch, device, criterion, noise_std):

    x_clean, _ = batch
    x_clean = x_clean.to(device)

    x_noisy, _ = add_noise(x_clean, noise_std)

    x_denoised = model(x_noisy)

    loss = criterion(x_denoised, x_clean)
    psnr = compute_psnr(x_denoised.detach(), x_clean.detach())

    return loss, psnr, x_clean.size(0)


# ============================================================
# TRAIN LOOP
# ============================================================

def train_one_epoch(model, loader, optimizer, device, config, criterion):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_metric = 0.0
    total_samples = 0

    for batch in loader:
        optimizer.zero_grad()

        if config.task == "classification":
            loss, correct, n = classification_step(
                model, batch, device, criterion, config.noise_std_max, add_noise_flag=True
            )
            total_correct += correct

        elif config.task == "denoising":
            loss, psnr, n = denoising_step(
                model, batch, device, criterion, config.noise_std_max
            )
            total_metric += psnr * n

        else:
            raise ValueError(f"Unknown task: {config.task}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * n
        total_samples += n

    avg_loss = total_loss / total_samples

    if config.task == "classification":
        avg_metric = total_correct / total_samples
    else:
        avg_metric = total_metric / total_samples

    return avg_loss, avg_metric


# ============================================================
# EVAL LOOP
# ============================================================

@torch.no_grad()
def evaluate_clean_classifier(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()

        n = y.size(0)
        total_loss += loss.item() * n
        total_correct += correct
        total_samples += n

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc

@torch.no_grad()
def evaluate_noisy_classifier(model, loader, device, criterion, sigma_max):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        loss, correct, n = classification_step(
            model, batch, device, criterion, sigma_max, add_noise_flag=True
        )

        total_loss += loss.item() * n
        total_correct += correct
        total_samples += n

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc

@torch.no_grad()
def evaluate_denoiser(model, loader, device, config, criterion):
    model.eval()

    total_loss = 0.0
    total_metric = 0.0
    total_samples = 0

    for batch in loader:
        loss, psnr, n = denoising_step(
            model, batch, device, criterion, config.noise_std_max
        )

        total_loss += loss.item() * n
        total_metric += psnr * n
        total_samples += n

    avg_loss = total_loss / total_samples
    avg_psnr = total_metric / total_samples

    return avg_loss, avg_psnr


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train(model, config):

    if config.task == "classification":
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_clean_loss": [],
            "val_clean_acc": [],
            "val_noisy_loss": [],
            "val_noisy_acc": []
        }
    else:
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_psnr": [],
            "val_psnr": []
        }

    ensure_dir(config.save_dir)
    model = model.to(config.device)

    print("Parameters:", count_parameters(model))

    train_loader, test_loader = get_mnist_loaders(
        config.batch_size,
        config.num_workers
    )

    if config.task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    best_metric = None

    for epoch in range(config.epochs):

        train_loss, train_metric = train_one_epoch(
            model,
            train_loader,
            optimizer,
            config.device,
            config,
            criterion
        )

        if config.task == "classification":
            val_clean_loss, val_clean_acc = evaluate_clean_classifier(
                model, test_loader, config.device, criterion
            )

            val_noisy_loss, val_noisy_acc = evaluate_noisy_classifier(
                model, test_loader, config.device, criterion, config.noise_std_max
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_metric)
            history["val_clean_loss"].append(val_clean_loss)
            history["val_clean_acc"].append(val_clean_acc)
            history["val_noisy_loss"].append(val_noisy_loss)
            history["val_noisy_acc"].append(val_noisy_acc)

            print(
                f"Epoch {epoch+1}/{config.epochs} | "
                f"train_loss={train_loss:.4f} | train_acc={train_metric:.4f} | "
                f"val_clean_loss={val_clean_loss:.4f} | val_clean_acc={val_clean_acc:.4f} | "
                f"val_noisy_loss={val_noisy_loss:.4f} | val_noisy_acc={val_noisy_acc:.4f}"
            )

            # Decide aquí qué métrica usar para guardar el mejor modelo
            metric = val_noisy_acc

        else:
            val_loss, val_psnr = evaluate_denoiser(
                model, test_loader, config.device, config, criterion
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_psnr"].append(train_metric)
            history["val_psnr"].append(val_psnr)

            print(
                f"Epoch {epoch+1}/{config.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"train_psnr={train_metric:.2f} | val_psnr={val_psnr:.2f}"
            )

            metric = val_loss

        torch.save(
            model.state_dict(),
            os.path.join(config.save_dir, f"{config.model_name}_last.pt")
        )

        if best_metric is None or (
            config.task == "classification" and metric > best_metric
        ) or (
            config.task == "denoising" and metric < best_metric
        ):
            best_metric = metric

            torch.save(
                model.state_dict(),
                os.path.join(config.save_dir, f"{config.model_name}_best.pt")
            )

    save_history(history, config)

    with open(os.path.join(config.save_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    save_training_plots(history, config)
    print("Training finished")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Example usage
    from code.models.classifier import CNNClassifier
    from code.models.denoiser import ResUNetDenoiser

    #classifier
    model = CNNClassifier()

    config = Config(
        model_name="cnn_classifier_clean",
        task="classification",
        epochs=30,
        save_dir="./checkpoints/classifier"
    )

    # #denoiser
    # model = ResUNetDenoiser()

    # config = Config(
    #     model_name="dncnn",
    #     task="denoising",
    #     epochs=20,
    #     save_dir="./checkpoints/denoiser"
    # )

    os.makedirs(config.save_dir, exist_ok=True)

    with open(os.path.join(config.save_dir, "config_clean.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    train(model, config)