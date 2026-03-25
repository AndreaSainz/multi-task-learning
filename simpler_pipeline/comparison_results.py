import os
import torch
import numpy as np
import matplotlib.pyplot as plt



# CONFIG
file_model_1 = "/home/as3628/multi-task-learning/simpler_pipeline/results/ExperimentsClassDeno/Experiments4/60_angles_sigma_0.3/results_mnist_ct_60_angles_sigma_0.3_10_1e-2.pt"   # primer modelo
file_model_2 = "/home/as3628/multi-task-learning/simpler_pipeline/results/ExperimentsScore/Experiments4/60_angles_sigma_0.3/results_mnist_ct_conditional_score_60_angles_sigma_0.3.pt"   # segundo modelo
output_dir = "/home/as3628/multi-task-learning/simpler_pipeline/results/Comparison/60_angles_sigma_0.3"

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



# LOAD

results1 = torch.load(file_model_1, map_location="cpu", weights_only=False)
results2 = torch.load(file_model_2, map_location="cpu", weights_only=False)

num_batches_1 = len(results1["sirt_images"])
num_batches_2 = len(results2["sirt_images"])
print(num_batches_1)


if num_batches_1 != num_batches_2:
    print(f"AVISO: distinto número de batches: modelo1={num_batches_1}, modelo2={num_batches_2}")

num_batches = min(num_batches_1, num_batches_2)



for batch_idx in range(num_batches):
    # número de muestras dentro del batch interno
    last_sirt_1 = results1["sirt_images"][batch_idx][-1]   # [B,1,H,W]
    B = last_sirt_1.shape[0]
    

    batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)

    for sample_idx in range(B):
        gt_img = get_gt_image(results1["gt_images"], batch_idx, sample_idx)

        sirt_img = get_last_image(results1["sirt_images"][batch_idx], sample_idx)
        both_img = get_last_image(results1["both_images"][batch_idx], sample_idx)
        clas_img = get_last_image(results1["clas_images"][batch_idx], sample_idx)
        deno_img = get_last_image(results1["deno_images"][batch_idx], sample_idx)
        model_img = get_last_image(results2["model_images"][batch_idx], sample_idx)

        imgs = [gt_img, sirt_img, both_img, clas_img, deno_img, model_img]
        titles = [
            "Ground Truth",
            "SIRT",
            "SIRT+Classifier+Denoiser",
            "SIRT+Classifier",
            "SIRT+Denoiser",
            "SIRT+Conditional Score",
        ]

        vmin = min(img.min() for img in imgs)
        vmax = max(img.max() for img in imgs)

        fig, axes = plt.subplots(1, 6, figsize=(24, 4))

        for ax, img, title in zip(axes, imgs, titles):
            im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fig.suptitle(safe_title_text(results1, results2, batch_idx, sample_idx), fontsize=12)

        # deja espacio a la derecha para la colorbar
        fig.subplots_adjust(right=0.92)

        # eje manual para la colorbar: [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.94, 0.18, 0.015, 0.64])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel("Intensity", rotation=270, labelpad=15)

        plt.savefig(
            os.path.join(batch_dir, f"sample_{sample_idx}_final_images.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        
        # Curvas PSNR
        sirt_psnr = get_metric_curve(results1["sirt_psnr"], batch_idx, sample_idx)
        both_psnr = get_metric_curve(results1["both_psnr"], batch_idx, sample_idx)
        clas_psnr = get_metric_curve(results1["clas_psnr"], batch_idx, sample_idx)
        deno_psnr = get_metric_curve(results1["deno_psnr"], batch_idx, sample_idx)
        model_psnr = get_metric_curve(results2["model_psnr"], batch_idx, sample_idx)

        iters_psnr = np.arange(len(sirt_psnr))

        plt.figure(figsize=(8, 5))
        plt.plot(iters_psnr, sirt_psnr, label="SIRT", marker="o", markersize=3)
        plt.plot(np.arange(len(both_psnr)), both_psnr, label="SIRT+Classifier+Denoiser", marker="o", markersize=3)
        plt.plot(np.arange(len(clas_psnr)), clas_psnr, label="SIRT+Classifier", marker="o", markersize=3)
        plt.plot(np.arange(len(deno_psnr)), deno_psnr, label="SIRT+Denoiser", marker="o", markersize=3)
        plt.plot(np.arange(len(model_psnr)), model_psnr, label="SIRT+Conditional Score", marker="o", markersize=3)

        plt.xlabel("Iteration")
        plt.ylabel("PSNR")
        plt.title(f"PSNR vs iteration | batch {batch_idx} sample {sample_idx}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(batch_dir, f"sample_{sample_idx}_psnr.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        # Curvas SSIM
        sirt_ssim = get_metric_curve(results1["sirt_ssim"], batch_idx, sample_idx)
        both_ssim = get_metric_curve(results1["both_ssim"], batch_idx, sample_idx)
        clas_ssim = get_metric_curve(results1["clas_ssim"], batch_idx, sample_idx)
        deno_ssim = get_metric_curve(results1["deno_ssim"], batch_idx, sample_idx)
        model_ssim = get_metric_curve(results2["model_ssim"], batch_idx, sample_idx)

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(len(sirt_ssim)), sirt_ssim, label="SIRT", marker="o", markersize=3)
        plt.plot(np.arange(len(both_ssim)), both_ssim, label="SIRT+Classifier+Denoiser", marker="o", markersize=3)
        plt.plot(np.arange(len(clas_ssim)), clas_ssim, label="SIRT+Classifier", marker="o", markersize=3)
        plt.plot(np.arange(len(deno_ssim)), deno_ssim, label="SIRT+Denoiser", marker="o", markersize=3)
        plt.plot(np.arange(len(model_ssim)), model_ssim, label="SIRT+Conditional Score", marker="o", markersize=3)

        plt.xlabel("Iteration")
        plt.ylabel("SSIM")
        plt.title(f"SSIM vs iteration | batch {batch_idx} sample {sample_idx}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(batch_dir, f"sample_{sample_idx}_ssim.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

                # Curvas PSNR crop
        sirt_psnr_crop = get_metric_curve(results1["sirt_psnr_crop"], batch_idx, sample_idx)
        both_psnr_crop = get_metric_curve(results1["both_psnr_crop"], batch_idx, sample_idx)
        clas_psnr_crop = get_metric_curve(results1["clas_psnr_crop"], batch_idx, sample_idx)
        deno_psnr_crop = get_metric_curve(results1["deno_psnr_crop"], batch_idx, sample_idx)
        model_psnr_crop = get_metric_curve(results2["model_psnr_crop"], batch_idx, sample_idx)

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(len(sirt_psnr_crop)), sirt_psnr_crop, label="SIRT", marker="o", markersize=3)
        plt.plot(np.arange(len(both_psnr_crop)), both_psnr_crop, label="SIRT+Classifier+Denoiser", marker="o", markersize=3)
        plt.plot(np.arange(len(clas_psnr_crop)), clas_psnr_crop, label="SIRT+Classifier", marker="o", markersize=3)
        plt.plot(np.arange(len(deno_psnr_crop)), deno_psnr_crop, label="SIRT+Denoiser", marker="o", markersize=3)
        plt.plot(np.arange(len(model_psnr_crop)), model_psnr_crop, label="SIRT+Conditional Score", marker="o", markersize=3)

        plt.xlabel("Iteration")
        plt.ylabel("PSNR crop")
        plt.title(f"PSNR crop vs iteration | batch {batch_idx} sample {sample_idx}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(batch_dir, f"sample_{sample_idx}_psnr_crop.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        # Curvas SSIM crop
        sirt_ssim_crop = get_metric_curve(results1["sirt_ssim_crop"], batch_idx, sample_idx)
        both_ssim_crop = get_metric_curve(results1["both_ssim_crop"], batch_idx, sample_idx)
        clas_ssim_crop = get_metric_curve(results1["clas_ssim_crop"], batch_idx, sample_idx)
        deno_ssim_crop = get_metric_curve(results1["deno_ssim_crop"], batch_idx, sample_idx)
        model_ssim_crop = get_metric_curve(results2["model_ssim_crop"], batch_idx, sample_idx)

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(len(sirt_ssim_crop)), sirt_ssim_crop, label="SIRT", marker="o", markersize=3)
        plt.plot(np.arange(len(both_ssim_crop)), both_ssim_crop, label="SIRT+Classifier+Denoiser", marker="o", markersize=3)
        plt.plot(np.arange(len(clas_ssim_crop)), clas_ssim_crop, label="SIRT+Classifier", marker="o", markersize=3)
        plt.plot(np.arange(len(deno_ssim_crop)), deno_ssim_crop, label="SIRT+Denoiser", marker="o", markersize=3)
        plt.plot(np.arange(len(model_ssim_crop)), model_ssim_crop, label="SIRT+Conditional Score", marker="o", markersize=3)

        plt.xlabel("Iteration")
        plt.ylabel("SSIM crop")
        plt.title(f"SSIM crop vs iteration | batch {batch_idx} sample {sample_idx}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(batch_dir, f"sample_{sample_idx}_ssim_crop.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

print(f"Listo. Resultados guardados en: {output_dir}")