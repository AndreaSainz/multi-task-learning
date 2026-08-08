import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

FILE_1 = "/home/as3628/multi-task-learning/simpler_pipeline/Results/ExperimentsClassDeno/Experiments4/60_angles_sigma_0.3/results_mnist_ct_60_angles_sigma_0.3_rank0.pt"
FILE_2 = "/home/as3628/multi-task-learning/simpler_pipeline/Results/ExperimentsNF/Experiments4/60_angles_sigma_0.3/results_mnist_ct_nfs_60_angles_sigma_0.3_rank0.pt"
FILE_3 = "/home/as3628/multi-task-learning/simpler_pipeline/Results/ExperimentsScore/Experiments4/60_angles_sigma_0.3/results_mnist_ct_conditional_score_60_angles_sigma_0.3_rank0.pt"


# Nombres que aparecerán en las leyendas para model
MODEL_2_NAME = "Model 2"
MODEL_3_NAME = "Model 3"


# ============================================================
# LOAD RESULTS
# ============================================================

results_1 = torch.load(FILE_1, map_location="cpu", weights_only=False)
results_2 = torch.load(FILE_2, map_location="cpu", weights_only=False)
results_3 = torch.load(FILE_3, map_location="cpu", weights_only=False)


# ============================================================
# HELPER
# ============================================================

def to_numpy(x):
    """
    Convierte una lista de floats o una lista de tensors de un elemento
    en un numpy array 1D.
    """
    return np.array([
        item.detach().cpu().item()
        if isinstance(item, torch.Tensor)
        else float(item)
        for item in x
    ])


# ============================================================
# CONFIGURACIÓN DE LAS MÉTRICAS
# ============================================================

metrics = {
    "PSNR": {
        "file1": [
            ("SIRT", "sirt_psnr"),
            ("Classifier + Denoiser", "both_psnr"),
            ("Classifier", "clas_psnr"),
            ("Denoiser", "deno_psnr"),
        ],
        "file2": [
            ("NF", "model_psnr"),
        ],
        "file3": [
            ("CDM", "model_psnr"),
        ],
        "ylabel": "PSNR (dB)",
    },

    "PSNR Crop": {
        "file1": [
            ("SIRT", "sirt_psnr_crop"),
            ("Classifier + Denoiser", "both_psnr_crop"),
            ("Classifier", "clas_psnr_crop"),
            ("Denoiser", "deno_psnr_crop"),
        ],
        "file2": [
            ("NF", "model_psnr_crop"),
        ],
        "file3": [
            ("CDM", "model_psnr_crop"),
        ],
        "ylabel": "PSNR Crop (dB)",
    },

    "SSIM": {
        "file1": [
            ("SIRT", "sirt_ssim"),
            ("Classifier + Denoiser", "both_ssim"),
            ("Classifier", "clas_ssim"),
            ("Denoiser", "deno_ssim"),
        ],
        "file2": [
            ("NF", "model_ssim"),
        ],
        "file3": [
            ("CDM", "model_ssim"),
        ],
        "ylabel": "SSIM",
    },

    "SSIM Crop": {
        "file1": [
            ("SIRT", "sirt_ssim_crop"),
            ("Classifier + Denoiser", "both_ssim_crop"),
            ("Classifier", "clas_ssim_crop"),
            ("Denoiser", "deno_ssim_crop"),
        ],
        "file2": [
            ("NF", "model_ssim_crop"),
        ],
        "file3": [
            ("CDM", "model_ssim_crop"),
        ],
        "ylabel": "SSIM Crop",
    },
}



# ============================================================
# PLOTS
# ============================================================

for metric_name, config in metrics.items():

    plt.figure(figsize=(10, 6))

    # File 1
    for label, key in config["file1"]:
        values = to_numpy(results_1[key][0])  # primer batch
        plt.plot(values, label=label)

    # File 2
    for label, key in config["file2"]:
        values = to_numpy(results_2[key][0])  # primer batch
        plt.plot(values, label=label)

    # File 3
    for label, key in config["file3"]:
        values = to_numpy(results_3[key][0])  # primer batch
        plt.plot(values, label=label)

    plt.xlabel("Iteration")
    plt.ylabel(config["ylabel"])
    plt.title(f"{metric_name}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"/home/as3628/multi-task-learning/simpler_pipeline/Results/{metric_name}.png",dpi=300,bbox_inches="tight")
    plt.close()

# ============================================================
# RECONSTRUCTION IMAGES - LAST ITERATION, FIRST BATCH
# ============================================================
# ============================================================
# EXTRACT IMAGES
# ============================================================

# File 1:
# [batch] -> [1, H, W]
sirt_img = results_1["sirt_images"][0][0].detach().cpu().numpy()
both_img = results_1["both_images"][0][0].detach().cpu().numpy()
clas_img = results_1["clas_images"][0][0].detach().cpu().numpy()
deno_img = results_1["deno_images"][0][0].detach().cpu().numpy()

# File 2 and 3:
# [batch][iteration] -> [B, 1, H, W]
# First batch, last iteration, first image
model_2_img = results_2["model_images"][0][-1][0, 0].detach().cpu().numpy()
model_3_img = results_3["model_images"][0][-1][0, 0].detach().cpu().numpy()


# ============================================================
# IMAGES
# ============================================================

images = [
    sirt_img,
    both_img,
    clas_img,
    deno_img,
    model_2_img,
    model_3_img,
]

titles = [
    "SIRT",
    "Classifier + Denoiser",
    "Classifier",
    "Denoiser",
    "NF",
    "CDM",
]


# Check shapes
for title, img in zip(titles, images):
    print(f"{title}: {img.shape}")


# ============================================================
# COMMON COLOR SCALE
# ============================================================

vmin = min(img.min() for img in images)
vmax = max(img.max() for img in images)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    len(images),
    figsize=(18, 3.5),
    constrained_layout=True
)

for ax, img, title in zip(axes, images, titles):

    im = ax.imshow(
        img,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)
    ax.axis("off")


# Shared colorbar
cbar = fig.colorbar(
    im,
    ax=axes,
    orientation="vertical",
    fraction=0.02,
    pad=0.02
)

cbar.set_label("Intensity")



# Save
plt.savefig(
    "/home/as3628/multi-task-learning/simpler_pipeline/Results/reconstructions_last_iteration.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()