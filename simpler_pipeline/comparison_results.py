import os
import torch
import numpy as np
import matplotlib.pyplot as plt



# CONFIG
model_configs = [
    {
        "name": "SIRT+Classifier+Denoiser",
        "file": "/home/as3628/multi-task-learning/simpler_pipeline/results/ExperimentsClassDeno/Experiments4/60_angles_sigma_0.3_5e-1_1e-2/results_mnist_ct_60_angles_sigma_0.3_5e-1_1e-2_conjunto.pt",
        "image_key": "both_images",
        "psnr_key": "both_psnr",
        "ssim_key": "both_ssim",
        "psnr_crop_key": "both_psnr_crop",
        "ssim_crop_key": "both_ssim_crop",
        "digits_key": "model_digits",
        "centers_key": "model_centers",
    },
    # {
    #     "name": "SIRT+Conditional Score",
    #     "file": "/home/as3628/multi-task-learning/simpler_pipeline/results/ExperimentsScore/Experiments4/60_angles_sigma_0.3/results_mnist_ct_conditional_score_60_angles_sigma_0.3.pt",
    #     "image_key": "model_images",
    #     "psnr_key": "model_psnr",
    #     "ssim_key": "model_ssim",
    #     "psnr_crop_key": "model_psnr_crop",
    #     "ssim_crop_key": "model_ssim_crop",
    #     "digits_key": "model_digits",
    #     "centers_key": "model_centers",
    # },
]
output_dir = "/home/as3628/multi-task-learning/simpler_pipeline/results/Comparison/60_angles_sigma_0.3_conjunto"

os.makedirs(output_dir, exist_ok=True)



def to_numpy_image(x):
    """
    Convierte tensores tipo [1,H,W], [H,W], [B,1,H,W] con B=1 a numpy [H,W].
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()

    if x.ndim == 4:
        # [B,1,H,W] -> coger primera imagen del batch interno si B=1
        x = x[0, 0]
    elif x.ndim == 3:
        # [1,H,W] -> [H,W]
        x = x[0]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Shape no soportada para imagen: {x.shape}")

    return x.numpy()


def get_last_image(batch_images, sample_idx=0):
    """
    batch_images: estructura esperada [iter][B,1,H,W] para un batch concreto
    Devuelve la última imagen de la muestra sample_idx.
    """
    last_iter = batch_images[-1]  # [B,1,H,W]
    if isinstance(last_iter, torch.Tensor):
        img = last_iter[sample_idx]   # [1,H,W]
    else:
        img = torch.tensor(last_iter)[sample_idx]
    return to_numpy_image(img)


def get_gt_image(gt_images, batch_idx, sample_idx=0):
    """
    gt_images puede venir como:
    - lista por batch: gt_images[batch_idx] -> [B,1,H,W]
    - tensor global: [num_batches, B,1,H,W]
    """
    x = gt_images[batch_idx]

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    img = x[sample_idx]
    return to_numpy_image(img)


def get_metric_curve(metric_container, batch_idx, sample_idx=0):
    """
    Extrae la curva de una métrica para un batch concreto.
    Esperado:
    - metric_container[batch_idx] = [iter] o [iter, B] o [iter][B]
    Devuelve array de longitud = num_iters
    """
    data = metric_container[batch_idx]

    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.array(data, dtype=object)

    # Casos posibles
    if arr.ndim == 1:
        # ya es [iter]
        try:
            return arr.astype(float)
        except:
            # puede ser lista de tensores escalares
            return np.array([
                float(x[sample_idx] if hasattr(x, "__len__") and not np.isscalar(x) else x)
                for x in data
            ])

    elif arr.ndim == 2:
        # [iter, B]
        return arr[:, sample_idx].astype(float)

    else:
        # fallback para listas raras tipo [iter][B]
        curve = []
        for item in data:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()

            item = np.array(item)

            if item.ndim == 0:
                curve.append(float(item))
            else:
                curve.append(float(item[sample_idx]))
        return np.array(curve)


def safe_title_text(results1, results2, batch_idx, sample_idx=0):
    parts = [f"batch {batch_idx}", f"sample {sample_idx}"]

    if "gt_digits" in results1:
        try:
            digit = results1["gt_digits"][batch_idx][sample_idx]
            if isinstance(digit, torch.Tensor):
                digit = digit.item()
            parts.append(f"digit={digit}")
        except:
            pass

    if "gt_centers" in results1:
        try:
            center = results1["gt_centers"][batch_idx][sample_idx]
            if isinstance(center, torch.Tensor):
                center = center.detach().cpu().numpy()
            parts.append(f"center={center}")
        except:
            pass

    return " | ".join(parts)

def get_digit_curve(results, digits_key, batch_idx, sample_idx=0):
    data = results[digits_key][batch_idx]
    curve = []

    for item in data:
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item)
        probs = item[sample_idx].detach().cpu().numpy()
        curve.append(probs)

    return np.stack(curve, axis=0)  # [num_iters, 10]


def get_center_hard_curve(results, centers_key, batch_idx, ymin, ymax, xmin, xmax, sample_idx=0):
    data = results[centers_key][batch_idx]
    centers = []
    W_valid = xmax - xmin + 1

    for item in data:
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item)

        submap = item[sample_idx, ymin:ymax+1, xmin:xmax+1]
        idx = torch.argmax(submap).item()

        iy = idx // W_valid
        ix = idx % W_valid

        cy = ymin + iy
        cx = xmin + ix

        centers.append([cy, cx])

    return np.array(centers, dtype=float)


def get_true_digit(results, batch_idx, sample_idx=0):
    x = results["gt_digits"][batch_idx]
    if isinstance(x, torch.Tensor):
        return int(x[sample_idx].item())
    return int(torch.tensor(x)[sample_idx].item())


def get_true_center(results, batch_idx, sample_idx=0):
    x = results["gt_centers"][batch_idx]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x[sample_idx].detach().cpu().numpy().astype(float)


def digit_accuracy_curve(results, digits_key, batch_idx, sample_idx=0):
    digit_probs = get_digit_curve(results, digits_key, batch_idx, sample_idx)
    true_digit = get_true_digit(results, batch_idx, sample_idx)
    pred_digits = np.argmax(digit_probs, axis=1)
    return (pred_digits == true_digit).astype(float)


def center_mse_curve(results, centers_key, batch_idx, ymin, ymax, xmin, xmax, sample_idx=0):
    pred_centers = get_center_hard_curve(results, centers_key, batch_idx, ymin, ymax, xmin, xmax, sample_idx)
    true_center = get_true_center(results, batch_idx, sample_idx)
    return np.mean((pred_centers - true_center[None, :]) ** 2, axis=1)

# LOAD

loaded_models = []
for cfg in model_configs:
    results = torch.load(cfg["file"], map_location="cpu", weights_only=False)
    loaded_models.append({
        "config": cfg,
        "results": results
    })

num_batches = min(len(m["results"]["gt_images"]) for m in loaded_models)
print(f"Número de batches comparables: {num_batches}")


ref_results = loaded_models[0]["results"]

for batch_idx in range(num_batches):
    last_sirt = ref_results["sirt_images"][batch_idx][-1]
    B = last_sirt.shape[0]

    batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)

    for sample_idx in range(B):
        gt_img = get_gt_image(ref_results["gt_images"], batch_idx, sample_idx)
        sirt_img = get_last_image(ref_results["sirt_images"][batch_idx], sample_idx)

        # =========================
        # Imagen final comparativa
        # =========================
        imgs = [gt_img, sirt_img]
        titles = ["Ground Truth", "SIRT"]

        for model_pack in loaded_models:
            cfg = model_pack["config"]
            results = model_pack["results"]

            model_img = get_last_image(results[cfg["image_key"]][batch_idx], sample_idx)
            imgs.append(model_img)
            titles.append(cfg["name"])

        vmin = min(img.min() for img in imgs)
        vmax = max(img.max() for img in imgs)

        fig, axes = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4))
        if len(imgs) == 1:
            axes = [axes]

        for ax, img, title in zip(axes, imgs, titles):
            im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fig.suptitle(safe_title_text(ref_results, ref_results, batch_idx, sample_idx), fontsize=12)
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.18, 0.015, 0.64])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel("Intensity", rotation=270, labelpad=15)

        plt.savefig(
            os.path.join(batch_dir, f"sample_{sample_idx}_final_images.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        # =========================
        # PSNR
        # =========================
        plt.figure(figsize=(8, 5))

        sirt_psnr = get_metric_curve(ref_results["sirt_psnr"], batch_idx, sample_idx)
        plt.plot(np.arange(len(sirt_psnr)), sirt_psnr, label="SIRT", marker="o", markersize=3)

        for model_pack in loaded_models:
            cfg = model_pack["config"]
            results = model_pack["results"]

            curve = get_metric_curve(results[cfg["psnr_key"]], batch_idx, sample_idx)
            plt.plot(np.arange(len(curve)), curve, label=cfg["name"], marker="o", markersize=3)

        plt.xlabel("Iteration")
        plt.ylabel("PSNR")
        plt.title(f"PSNR vs iteration | batch {batch_idx} sample {sample_idx}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, f"sample_{sample_idx}_psnr.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # =========================
        # SSIM
        # =========================
        plt.figure(figsize=(8, 5))

        sirt_ssim = get_metric_curve(ref_results["sirt_ssim"], batch_idx, sample_idx)
        plt.plot(np.arange(len(sirt_ssim)), sirt_ssim, label="SIRT", marker="o", markersize=3)

        for model_pack in loaded_models:
            cfg = model_pack["config"]
            results = model_pack["results"]

            curve = get_metric_curve(results[cfg["ssim_key"]], batch_idx, sample_idx)
            plt.plot(np.arange(len(curve)), curve, label=cfg["name"], marker="o", markersize=3)

        plt.xlabel("Iteration")
        plt.ylabel("SSIM")
        plt.title(f"SSIM vs iteration | batch {batch_idx} sample {sample_idx}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, f"sample_{sample_idx}_ssim.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # =========================
        # PSNR crop
        # =========================
        plt.figure(figsize=(8, 5))

        sirt_psnr_crop = get_metric_curve(ref_results["sirt_psnr_crop"], batch_idx, sample_idx)
        plt.plot(np.arange(len(sirt_psnr_crop)), sirt_psnr_crop, label="SIRT", marker="o", markersize=3)

        for model_pack in loaded_models:
            cfg = model_pack["config"]
            results = model_pack["results"]

            curve = get_metric_curve(results[cfg["psnr_crop_key"]], batch_idx, sample_idx)
            plt.plot(np.arange(len(curve)), curve, label=cfg["name"], marker="o", markersize=3)

        plt.xlabel("Iteration")
        plt.ylabel("PSNR crop")
        plt.title(f"PSNR crop vs iteration | batch {batch_idx} sample {sample_idx}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, f"sample_{sample_idx}_psnr_crop.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # =========================
        # SSIM crop
        # =========================
        plt.figure(figsize=(8, 5))

        sirt_ssim_crop = get_metric_curve(ref_results["sirt_ssim_crop"], batch_idx, sample_idx)
        plt.plot(np.arange(len(sirt_ssim_crop)), sirt_ssim_crop, label="SIRT", marker="o", markersize=3)

        for model_pack in loaded_models:
            cfg = model_pack["config"]
            results = model_pack["results"]

            curve = get_metric_curve(results[cfg["ssim_crop_key"]], batch_idx, sample_idx)
            plt.plot(np.arange(len(curve)), curve, label=cfg["name"], marker="o", markersize=3)

        plt.xlabel("Iteration")
        plt.ylabel("SSIM crop")
        plt.title(f"SSIM crop vs iteration | batch {batch_idx} sample {sample_idx}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, f"sample_{sample_idx}_ssim_crop.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # # =========================
        # # Accuracy clasificación
        # # =========================
        # plt.figure(figsize=(8, 5))

        # for model_pack in loaded_models:
        #     cfg = model_pack["config"]
        #     results = model_pack["results"]

        #     if cfg["digits_key"] in results:
        #         curve = digit_accuracy_curve(results, cfg["digits_key"], batch_idx, sample_idx)
        #         plt.plot(np.arange(len(curve)), curve, label=cfg["name"], marker="o", markersize=3)

        # plt.xlabel("Iteration")
        # plt.ylabel("Classification Accuracy")
        # plt.title(f"Digit accuracy vs iteration | batch {batch_idx} sample {sample_idx}")
        # plt.ylim(-0.05, 1.05)
        # plt.grid(True, alpha=0.3)
        # plt.legend(fontsize=8)
        # plt.tight_layout()
        # plt.savefig(os.path.join(batch_dir, f"sample_{sample_idx}_digit_accuracy.png"), dpi=200, bbox_inches="tight")
        # plt.close()

        # # =========================
        # # MSE centro
        # # =========================
        # plt.figure(figsize=(8, 5))

        # for model_pack in loaded_models:
        #     cfg = model_pack["config"]
        #     results = model_pack["results"]

        #     if cfg["centers_key"] in results:
        #         curve = center_mse_curve(results, cfg["centers_key"], batch_idx, ymin, ymax, xmin, xmax, sample_idx)
        #         plt.plot(np.arange(len(curve)), curve, label=cfg["name"], marker="o", markersize=3)

        # plt.xlabel("Iteration")
        # plt.ylabel("Center MSE")
        # plt.title(f"Center MSE vs iteration | batch {batch_idx} sample {sample_idx}")
        # plt.grid(True, alpha=0.3)
        # plt.legend(fontsize=8)
        # plt.tight_layout()
        # plt.savefig(os.path.join(batch_dir, f"sample_{sample_idx}_center_mse.png"), dpi=200, bbox_inches="tight")
        # plt.close()
        