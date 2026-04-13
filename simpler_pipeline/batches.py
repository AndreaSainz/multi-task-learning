import torch
import numpy as np

path_bs1 = "/home/as3628/multi-task-learning/simpler_pipeline/results/ExperimentsClassDeno/Experiments4/results_mnist_ct_30_angles_sigma_0.3_1e-1_5e-2_conjunto_1batch_rank0.pt"
path_bs10 = "/home/as3628/multi-task-learning/simpler_pipeline/results/ExperimentsClassDeno/Experiments4/results_mnist_ct_30_angles_sigma_0.3_1e-1_5e-2_conjunto_10batch_rank0.pt"

r1 = torch.load(path_bs1, map_location="cpu", weights_only=False)
r10 = torch.load(path_bs10, map_location="cpu", weights_only=False)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def compare_scalar_series(name, a, b, atol=1e-7, rtol=1e-7):
    if len(a) != len(b):
        print(f"{name}: distinta longitud -> {len(a)} vs {len(b)}")
        return False

    ok = True
    max_diff = 0.0

    for i, (xa, xb) in enumerate(zip(a, b)):
        xa = to_numpy(xa).astype(np.float64)
        xb = to_numpy(xb).astype(np.float64)

        if xa.shape != xb.shape:
            print(f"{name}: shape distinta en muestra {i} -> {xa.shape} vs {xb.shape}")
            ok = False
            continue

        diff = np.max(np.abs(xa - xb))
        max_diff = max(max_diff, diff)

        if not np.allclose(xa, xb, atol=atol, rtol=rtol):
            print(f"{name}: diferencia en muestra {i}, max abs diff = {diff}")
            ok = False

    if ok:
        print(f"{name}: OK, idéntico dentro de tolerancia (max abs diff = {max_diff})")
    return ok

def compare_tensor_list(name, a, b, atol=1e-7, rtol=1e-7):
    if len(a) != len(b):
        print(f"{name}: distinta longitud -> {len(a)} vs {len(b)}")
        return False

    ok = True
    max_diff = 0.0

    for i, (xa, xb) in enumerate(zip(a, b)):
        xa = to_numpy(xa).astype(np.float64)
        xb = to_numpy(xb).astype(np.float64)

        if xa.shape != xb.shape:
            print(f"{name}: shape distinta en muestra {i} -> {xa.shape} vs {xb.shape}")
            ok = False
            continue

        diff = np.max(np.abs(xa - xb))
        max_diff = max(max_diff, diff)

        if not np.allclose(xa, xb, atol=atol, rtol=rtol):
            print(f"{name}: diferencia en muestra {i}, max abs diff = {diff}")
            ok = False

    if ok:
        print(f"{name}: OK, idéntico dentro de tolerancia (max abs diff = {max_diff})")
    return ok

# métricas por muestra y por iteración
series_keys = [
    "both_psnr", "sirt_psnr", "clas_psnr", "deno_psnr",
    "both_ssim", "sirt_ssim", "clas_ssim", "deno_ssim",
    "both_psnr_crop", "sirt_psnr_crop", "clas_psnr_crop", "deno_psnr_crop",
    "both_ssim_crop", "sirt_ssim_crop", "clas_ssim_crop", "deno_ssim_crop",
    "both_digit_prob_true", "sirt_digit_prob_true",
    "both_digit_nll", "sirt_digit_nll",
    "both_digit_entropy", "sirt_digit_entropy",
    "both_centre_w1", "sirt_centre_w1",
]

# tensores finales por muestra
tensor_keys = [
    "gt_images", "gt_digits", "gt_centers",
    "both_images", "sirt_images", "clas_images", "deno_images",
    "both_digits", "sirt_digits",
    "both_centers", "sirt_centers",
]

all_ok = True

print("=== Comparando series ===")
for k in series_keys:
    ok = compare_scalar_series(k, r1[k], r10[k])
    all_ok = all_ok and ok

print("\n=== Comparando tensores finales ===")
for k in tensor_keys:
    ok = compare_tensor_list(k, r1[k], r10[k])
    all_ok = all_ok and ok

print("\n=== Resultado final ===")
if all_ok:
    print("Las 10 muestras coinciden en ambos .pt dentro de la tolerancia.")
else:
    print("Hay diferencias entre los dos .pt.")