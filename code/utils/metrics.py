import numpy as np
from pytorch_msssim import ssim
import math
import torch
from torch.nn import MSELoss
import torch.nn.functional as F


def compute_psnr(mse, max_val=1.0):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) from a given MSE value.

    PSNR is a logarithmic metric that quantifies the difference between two images,
    typically used to evaluate reconstruction quality. Higher PSNR values indicate
    better image fidelity.

    Args:
        mse (float or torch.Tensor): 
            Mean squared error between the images.
        max_val (float, optional): 
            Maximum possible pixel value (default: 1.0).

    Returns:
        float: 
            PSNR value in decibels (dB). Returns infinity if MSE is zero.

    Example:
        >>> mse_value = 0.001
        >>> psnr = compute_psnr(mse_value)
    """
    
    # if mse is a tensor, extract the value
    if not isinstance(mse, torch.Tensor):
        mse = torch.tensor(mse, dtype=torch.float32)

    psnr = torch.empty_like(mse)
    nonzero = mse > 0
    psnr[nonzero] = 10 * torch.log10((max_val ** 2) / mse[nonzero])
    psnr[~nonzero] = float("inf")

    return psnr


def compute_psnr_results(pred, target, max_val=1.0):
    """
    Computes the PSNR between a predicted and a ground truth image.

    This function internally calculates the MSE and then applies the PSNR formula.
    Input tensors must have the same shape.

    Args:
        pred (torch.Tensor): 
            Reconstructed image tensor of shape (B, C, H, W).
        target (torch.Tensor): 
            Ground truth image tensor of the same shape.
        max_val (float, optional): 
            Maximum possible pixel value (default: 1.0).

    Returns:
        float: 
            PSNR value in decibels (dB).

    Example:
        >>> psnr = compute_psnr_results(pred, target)
    """


    #calculate the L2 value for the ground truth and prediction
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)   # [B]

    psnr = torch.empty_like(mse)
    nonzero = mse > 0
    psnr[nonzero] = 10 * torch.log10((max_val ** 2) / mse[nonzero])
    psnr[~nonzero] = float("inf")

    return psnr


def compute_ssim(reconstructed, ground_truth, data_range):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    SSIM is a perceptual metric that evaluates image similarity based on luminance,
    contrast, and structural information. Values close to 1.0 indicate high similarity.

    Args:
        reconstructed (torch.Tensor): 
            Reconstructed image tensor of shape (B, C, H, W) or (B, H, W).
        ground_truth (torch.Tensor): 
            Ground truth image tensor of shape (B, C, H, W) or (B, H, W).
        data_range (float): 
            The dynamic range of the input data (e.g., 1.0 or 255).

    Returns:
        float: 
            SSIM value in the range [-1, 1], where 1.0 indicates perfect similarity.

    Example:
        >>> ssim_score = compute_ssim(reconstructed, ground_truth, data_range=1.0)
    """
    
    # Ensure input tensors are 4D: [B, C, H, W]
    if reconstructed.ndim == 3:
        reconstructed = reconstructed.unsqueeze(1)
    if ground_truth.ndim == 3:
        ground_truth = ground_truth.unsqueeze(1)

    B = reconstructed.shape[0]
    vals = []

    for b in range(B):
        val = ssim(
            reconstructed[b:b+1],
            ground_truth[b:b+1],
            data_range=data_range
        )
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().float()
            if val.numel() == 1:
                val = val.squeeze(0)
        else:
            val = torch.tensor(val, dtype=torch.float32)
        vals.append(val)

    return torch.stack(vals)   # [B]


def compute_mse(pred, target):
    return ((pred - target) ** 2).flatten(1).mean(dim=1)

def update_confusion_matrix(cm, logits, target, num_classes=10):
    """
    Accumulates confusion matrix for multiclass classification.
    cm: (C, C) where rows=true, cols=pred
    logits: (B, C)
    target: (B,)
    """
    pred = torch.argmax(logits, dim=1)
    for t, p in zip(target.view(-1), pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def metrics_from_confusion_matrix(cm, eps=1e-8):
    """
    Returns accuracy, macro precision, macro recall, macro f1 from confusion matrix.
    """
    cm = cm.float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    precision_per_class = tp / (tp + fp + eps)
    recall_per_class    = tp / (tp + fn + eps)
    f1_per_class        = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

    macro_precision = precision_per_class.mean()
    macro_recall    = recall_per_class.mean()
    macro_f1        = f1_per_class.mean()

    accuracy = tp.sum() / (cm.sum() + eps)

    return {
        "accuracy": accuracy.item(),
        "precision": macro_precision.item(),
        "recall": macro_recall.item(),
        "f1": macro_f1.item(),
    }

def compute_metric_classifier(pred, target, num_classes=10):
    """
    Computes macro-averaged precision, recall, F1-score and accuracy.
    """

    # Predicted class labels
    pred_labels = torch.argmax(pred, dim=1)

    # Accuracy
    accuracy = (pred_labels == target).float().mean()

    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        tp = ((pred_labels == c) & (target == c)).sum().float()
        fp = ((pred_labels == c) & (target != c)).sum().float()
        fn = ((pred_labels != c) & (target == c)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "accuracy": accuracy.item(),
        "precision": torch.mean(torch.stack(precision_list)).item(),
        "recall": torch.mean(torch.stack(recall_list)).item(),
        "f1": torch.mean(torch.stack(f1_list)).item(),
    }
