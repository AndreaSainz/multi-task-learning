import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import tomosipo as ts
import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
from code.ct_recon.SIRT import sirt_step
from code.utils.image_cropping import extract_crops, insert_crop_grad_in_image


def update_case(x, y, A_torch, AT_torch, centre, digit, alpha, i, r, batch_id, mode, regularization_grad, results_dir, name):
    x, step_data = sirt_step(A_torch, AT_torch, x, y, alpha)
    if mode == "none":
        return x.clamp(0, 1), step_data

    crop_x = extract_crops(x, centre).detach().requires_grad_(True)
    grad_crop = regularization_grad(crop_x, digit, i, batch_id, r)
    if  batch_id == 0 and i % 2 == 0 and r==0: #batch_id == 0 and
        plt.imshow(grad_crop[0,0].detach().cpu().numpy(), cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_grad_crop_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()
    grad_F = insert_crop_grad_in_image(x, grad_crop, centre)
    x = (x - grad_F).clamp(0, 1)

    return x, step_data



def score_centers_difussion_model(
    model,
    img,
    *,
    denoiser=None,
    H=100,
    W=100,
    patch=28,
    ymin=14,
    ymax=85,
    xmin=14,
    xmax=85,
    chunk_size=512,
    return_digit_map=False,
):
    """
    Para cada centro candidato (y,x), extrae el crop 28x28 y calcula:

        score(y,x) = max_c p(c | crop_{y,x})

    donde p(c|crop) viene del clasificador.

    Args:
        model: clasificador que devuelve logits [B,10]
        img: [B,1,H,W]
        denoiser: opcional. Si se da, se aplica al crop antes del clasificador
        H, W: tamaño de la imagen
        patch: tamaño del crop (28)
        ymin, ymax, xmin, xmax: rango de centros válidos
        chunk_size: número de crops procesados por chunk
        return_digit_map: si True, devuelve también el mapa del dígito predicho

    Returns:
        map_x: [B,H,W] con score = max softmax prob en centros válidos y -inf fuera
        digit_map (opcional): [B,H,W] con dígito predicho en centros válidos y -1 fuera
    """

    device = img.device
    B, C, Hx, Wx = img.shape
    assert C == 1
    assert Hx == H and Wx == W
    assert patch % 2 == 0

    rad = patch // 2

    # Todos los patches posibles
    patches = img.unfold(2, patch, 1).unfold(3, patch, 1)
    # [B,1,H-patch+1,W-patch+1,patch,patch]

    ty_min, ty_max = ymin - rad, ymax - rad
    tx_min, tx_max = xmin - rad, xmax - rad

    # Solo la zona válida de centros
    patches = patches[:, :, ty_min:ty_max+1, tx_min:tx_max+1, :, :]
    # [B,1,gh,gw,patch,patch]

    gh, gw = patches.shape[2], patches.shape[3]
    N = gh * gw

    # [B*N,1,patch,patch]
    patches_flat = patches.contiguous().view(B * N, 1, patch, patch)

    scores_flat = torch.empty(B * N, device=device, dtype=torch.float32)

    if return_digit_map:
        digits_flat = torch.empty(B * N, device=device, dtype=torch.long)

    for start in range(0, B * N, chunk_size):
        end = min(start + chunk_size, B * N)
        p = patches_flat[start:end]

        if denoiser is not None:
            p = denoiser(p).clamp(0.0, 1.0)

        logits = model(p)                 # [chunk, 10]
        probs = torch.softmax(logits, dim=1)

        conf, pred = probs.max(dim=1)     # [chunk], [chunk]

        scores_flat[start:end] = conf.float()

        if return_digit_map:
            digits_flat[start:end] = pred

    scores_grid = scores_flat.view(B, gh, gw)

    map_x = torch.full((B, H, W), -torch.inf, device=device, dtype=torch.float32)
    map_x[:, ymin:ymax+1, xmin:xmax+1] = scores_grid

    if return_digit_map:
        digits_grid = digits_flat.view(B, gh, gw)
        digit_map = torch.full((B, H, W), -1, device=device, dtype=torch.long)
        digit_map[:, ymin:ymax+1, xmin:xmax+1] = digits_grid
        return map_x, digit_map

    return map_x