import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_loader import MnistDataset
from code.models.classifier import CNNClassifier
from code.models.denoiser import ResUNetDenoiser
from code.utils.metrics import compute_psnr_results, compute_ssim, compute_mse, update_confusion_matrix, metrics_from_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import logging
import os
from torch.utils.data import Subset
import normflows as nf
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


seed = 29072000
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

###################################################################### Parameters #####################################################################

### data
seed = 29072000
H = 100
W = 100
num_angles = 30 #60
i_0 = 10**5
sigma_data = 0.3
patch = 28
rad = patch // 2
ymin, ymax = rad, H - rad - 1
xmin, xmax = rad, W - rad - 1


### training parameters
T_ini = 100
T_max = 80
T_min = 1
decay_iters = 10
I = 10
tol_dx = 1e-3
patience_dx = 3
alpha = 2e-1/(53.595951080322266)**2 #1/(liptchitz constant)^2. . 75.77664947509766
beta1 = [1e-1, 1e-1]
beta2 = [5e-2, 5e-2] #[5e-4, 5e-3]
alpha_img = 0.01
accelerator = Accelerator()
device = accelerator.device
n_train = 500
batch_size = 10
save_batch_id = 0
topk_digits = 5184
results_dir = "simpler_pipeline/results/ExperimentsClassDeno/Experiments4/30_angles_sigma_0.3_1e-1_5e-2"


###################################################################### Data #####################################################################

# define CT geometry
vg = ts.volume(shape=(1,H,W))                                 # Volume
angles = np.linspace(0, np.pi, num_angles, endpoint=True)     # Angles
pg = ts.parallel(angles=angles, shape=(1, H))                 # Parallel beam structure
A = ts.operator(vg,pg)


A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)


# Import data
data = MnistDataset(training = True,
                testing= False, 
                A = A, 
                dimension = (H,W), 
                num_angles = num_angles, 
                sigma = sigma_data,
                i_0 = i_0,
                cut = False,
                seed = seed,
                max_len = None,
                device = device)

#for reproducibility
g_split = torch.Generator()
g_split.manual_seed(seed)
labels = np.array([data[i]["label"] for i in range(len(data))])

# Split estratificado
train_idx, val_idx = train_test_split(
    np.arange(len(data)),
    train_size=n_train,
    stratify=labels,
    random_state=seed
)

train_data = Subset(data, train_idx)


# create dataloader
train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    generator=g_split)


###################################################################### Logger #####################################################################
# create logger for checkpoints
log_dir = "logs"
name = "30_angles_sigma_0.3_1e-1_5e-2_500samples"

os.makedirs(log_dir, exist_ok=True)
handlers = [logging.StreamHandler()]
if accelerator.is_main_process:
    handlers.insert(0, logging.FileHandler(os.path.join(log_dir, f"{name}.log")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=handlers
)

logger = logging.getLogger(__name__)




#Check if the dataset is balanced
labels_bal = torch.tensor([train_data[i]["label"] for i in range(len(train_data))])
counts = torch.bincount(labels_bal, minlength=10)

if accelerator.is_main_process:
    logger.info("Balanced dataset label counts:")
    for d in range(10):
        logger.info(f"Digit {d}: {counts[d].item()}")


###################################################################### Trained Classifier and Denoiser Models #####################################################################

class_model = CNNClassifier()

checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/classifier/cnn_classifier_clean_best.pt", map_location="cpu")
class_model.load_state_dict(checkpoint)

class_model.eval() 
for p in class_model.parameters():
   p.requires_grad_(False)

den_model = ResUNetDenoiser()

checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/denoiser/dncnn_best.pt", map_location="cpu")
den_model.load_state_dict(checkpoint)

den_model.eval() 
for p in den_model.parameters():
   p.requires_grad_(False)


class_model, den_model, train_dataloader = accelerator.prepare(class_model, den_model, train_dataloader)
###################################################################### Center finder function #####################################################################
@torch.no_grad()
def score_centers_with_unfold(
    model,
    digit_cond,
    img,
    *,
    denoiser,
    H=100,
    W=100,
    patch=28,
    ymin=14,
    ymax=85,
    xmin=14,
    xmax=85,
    chunk_size=512
):
    """
    For each candidate center (y, x), extracts the 28x28 crop and computes:

        score(y,x) = sum_k digit_cond[k] * p(k | crop_{y,x})

    Then normalizes over all valid centers to obtain a proper probability map
    over centers.

    Args:
        model: classifier returning logits of shape [B, 10]
        img: input images of shape [B, 1, H, W]
        digit_cond: [B, 10], probability distribution over digits
                    (can be one-hot or soft probabilities)
        denoiser: model applied to each crop before classification (always used)
        H, W: image dimensions
        patch: crop size (28)
        ymin, ymax, xmin, xmax: valid center range
        chunk_size: number of crops processed per chunk
        return_digit_map: if True, also returns the predicted digit at each center

    Returns:
        map_x: [B, H, W] probability map over centers
               (0 outside valid region, sums to 1 inside valid region)
        digit_map (optional): [B, H, W] predicted digit at valid centers, -1 outside
    """

    device = img.device
    B, C, Hx, Wx = img.shape
    assert C == 1
    assert Hx == H and Wx == W
    assert patch % 2 == 0
    assert denoiser is not None, "denoiser must be provided and is always used"
    

    rad = patch // 2

    # Extract all possible patches
    patches = img.unfold(2, patch, 1).unfold(3, patch, 1)
    # shape: [B, 1, H-patch+1, W-patch+1, patch, patch]

    # Convert valid centers to top-left patch coordinates
    ty_min, ty_max = ymin - rad, ymax - rad
    tx_min, tx_max = xmin - rad, xmax - rad

    # Keep only valid region
    patches = patches[:, :, ty_min:ty_max+1, tx_min:tx_max+1, :, :]
    # shape: [B, 1, gh, gw, patch, patch]

    gh, gw = patches.shape[2], patches.shape[3]
    N = gh * gw

    # Flatten patches
    patches_flat = patches.contiguous().view(B * N, 1, patch, patch)

    # Expand digit probabilities to match patches
    digit_cond_expanded = digit_cond.unsqueeze(1).repeat(1, N, 1)   # [B, N, 10]
    digit_cond_expanded = digit_cond_expanded.view(B * N, 10)       # [B*N, 10]

    scores_flat = torch.empty(B * N, device=device, dtype=torch.float32)

    for start in range(0, B * N, chunk_size):
        end = min(start + chunk_size, B * N)

        p = patches_flat[start:end]                      # [chunk,1,patch,patch]
        w = digit_cond_expanded[start:end]              # [chunk,10]
        # Apply denoiser (the center also affects the denoiser part)
        # p = denoiser(p).clamp(0.0, 1.0)
        active_pixels = (p > sigma_data).float().sum(dim=(1, 2, 3))
        active_ratio = active_pixels / (patch * patch)         # [chunk], between 0 y 1

        logits = model(p)                               # [chunk,10]
        probs = torch.softmax(logits, dim=1)            # [chunk,10]

        # Weighted score using digit distribution
        score_cls = (w * probs).sum(dim=1)                  # [chunk]
        score = 0.8 * score_cls + 0.2 * active_ratio
        scores_flat[start:end] = score.float()


    # Reshape back to grid
    scores_grid = scores_flat.view(B, gh, gw)

    # Build full map (0 outside valid region)
    map_x = torch.zeros((B, H, W), device=device, dtype=torch.float32)
    map_x[:, ymin:ymax+1, xmin:xmax+1] = scores_grid

    # Normalize to obtain a probability distribution over centers
    Z = map_x.flatten(1).sum(dim=1, keepdim=True) + 1e-12
    map_x = map_x / Z.view(B, 1, 1)

    del p, w, logits, probs, score

    return map_x

def patch_scores_for_digits(
    crop,
    digit_probs,
    class_model,
    den_model,
    mode="both",
    digit_threshold=0.1,
    beta_class=1.0,
    beta_deno=1.0,
    sigma_data=1.0,
):
    """
    crop: [B,1,28,28]
    digit_probs: [B,10]
    devuelve:
        grad_patch: [B,1,28,28]
    """
    crop_req = crop.clone().detach().requires_grad_(True)
    grad_patch = torch.zeros_like(crop_req)

    if mode in ["both", "class"]:
        logits = class_model(crop_req)
        probs = torch.softmax(logits, dim=1)

        weights = digit_probs.clone()
        weights[weights < digit_threshold] = 0.0
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        mixture = (weights * probs).sum(dim=1)
        objective = torch.log(mixture.clamp_min(1e-12)).sum()
        grad_clf = torch.autograd.grad(objective, crop_req, retain_graph=False)[0]

        grad_patch = grad_patch + beta_class * grad_clf

    if mode in ["both", "deno"]:
        grad_prior = (den_model(crop_req) - crop_req).detach()
        grad_patch = grad_patch + beta_deno * grad_prior

    grad_patch = -(sigma_data ** 2) * grad_patch
    grad_patch = grad_patch.detach()

    return grad_patch

def select_centres(centre_map, ymin=14, ymax=85, xmin=14, xmax=85,
                   threshold=None, topk=topk_digits, renormalize=True):
    """
    centre_map: [B,H,W]
    devuelve listas por batch:
        selected_indices[b] = [(cy,cx,p), ...]
    """
    B, H, W = centre_map.shape
    selected = []

    for b in range(B):
        probs = centre_map[b, ymin:ymax+1, xmin:xmax+1]   # [Hy,Wx]
        Hy, Wx = probs.shape
        flat = probs.reshape(-1)

        if topk is not None:
            k = min(topk, flat.numel())
            vals, idxs = torch.topk(flat, k)
            if threshold is not None:
                keep = vals > threshold
                vals = vals[keep]
                idxs = idxs[keep]
        else:
            idxs = torch.nonzero(flat > threshold, as_tuple=False).squeeze(1)
            vals = flat[idxs]

        if vals.numel() == 0:
            # fallback: coger el máximo
            vals, idxs = torch.topk(flat, 1)

        if renormalize:
            vals = vals / vals.sum()

        coords = []
        for v, idx in zip(vals, idxs):
            iy = (idx // Wx).item()
            ix = (idx %  Wx).item()
            cy = ymin + iy
            cx = xmin + ix
            coords.append((cy, cx, v))
        selected.append(coords)

    return selected

def expected_task_gradient(
    x,
    i ,
    digit_probs,
    centre_map,
    class_model,
    den_model,
    mode="both",
    ymin=14, ymax=85, xmin=14, xmax=85,
    digit_threshold=0,
    centre_threshold=None,
    topk_centres=20,
    renormalize_centres=True,
    beta_class=1.0,
    beta_deno=1.0,
    sigma_data=1.0  
):
    """
    x: [B,1,H,W]
    digit_probs: [B,10]
    centre_map: [B,H,W]
    devuelve grad_full: [B,1,H,W]
    """
    B, _, H, W = x.shape
    grad_full = torch.zeros_like(x)

    selected_centres = select_centres(
        centre_map,
        ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
        threshold=centre_threshold,
        topk=topk_centres,
        renormalize=renormalize_centres
    )

    for b in range(B):
        xb = x[b:b+1]                         # [1,1,H,W]
        xb_norm = xb.norm() + 1e-12
        dp = digit_probs[b:b+1]

        for cy, cx, pc in selected_centres[b]:
            crop = xb[:, :, cy-14:cy+14, cx-14:cx+14]

            grad_patch = patch_scores_for_digits(
                crop=crop,
                digit_probs=dp,
                class_model=class_model,
                den_model=den_model,
                mode=mode,
                digit_threshold=digit_threshold,
                beta_class=beta_class,
                beta_deno=beta_deno,
                sigma_data=sigma_data,
            )  # [1,1,28,28]


            # complete gradient for this center
            grad_c = alpha_img * xb    # [1,1,H,W]

            # subtitution for grad_crop in the crop 
            grad_c = grad_c.clone()
            grad_c[:, :, cy-14:cy+14, cx-14:cx+14] = grad_patch

            # esperanza respecto al centro
            grad_full[b:b+1] += pc * grad_c

    del crop, dp, grad_patch
    return grad_full

def sirt_step(x, y, alpha):
    Ax = A_torch(x)
    r = y - Ax
    AT_r = AT_torch(r)
    step_data = alpha * AT_r
    x_new = x + step_data
    return x_new, step_data


def extract_crops(x, centre):
    B = x.shape[0]
    crops = torch.zeros(B, 1, 28, 28, device=x.device, dtype=x.dtype)
    for b in range(B):
        cy = int(centre[b, 0].item())
        cx = int(centre[b, 1].item())
        crops[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
    return crops



def regularization_grad(crop_x, digit, mode, i, batch_id, t):
    grad_classi = 0.0
    grad_deno = 0.0

    if mode in ["both", "class"]:
        logits = class_model(crop_x)
        log_probs = torch.log_softmax(logits, dim=1)
        classi = log_probs.gather(1, digit.unsqueeze(1)).squeeze(1)
        grad_classi = torch.autograd.grad(classi.sum(), crop_x, create_graph=False)[0]

    if mode in ["both", "deno"]:
        deno = den_model(crop_x)
        grad_deno = deno - crop_x

    idx = 0 if i == 0 else 1

    if mode == "both":
        grad_crop = -(sigma_data**2) * (beta1[idx] * grad_classi + beta2[idx] * grad_deno)
        if accelerator.is_main_process:
            logger.info(f"Gradient descent step {batch_id}_{i}_{t}| classifier : {beta1[idx] * grad_classi.flatten(1).norm(dim=1).mean().item()}, denoiser : {beta2[idx] * grad_deno.flatten(1).norm(dim=1).mean().item()}")
    elif mode == "class":
        grad_crop = -(sigma_data**2) * (beta1[idx] * grad_classi)
        if accelerator.is_main_process:
            logger.info(f"Gradient descent step {batch_id}_{i}_{t}| classifier : {beta1[idx] * grad_classi.flatten(1).norm(dim=1).mean().item()}")
    elif mode == "deno":
        grad_crop = -(sigma_data**2) * (beta2[idx] * grad_deno)
        if accelerator.is_main_process:
            logger.info(f"Gradient descent step {batch_id}_{i}_{t}| denoiser : {beta2[idx] * grad_deno.flatten(1).norm(dim=1).mean().item()}")
    elif mode == "none":
        grad_crop = torch.zeros_like(crop_x)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return grad_crop


def update_case(x, y, centre, digit, alpha, i, t, batch_id, mode):
    x, step_data = sirt_step(x, y, alpha)

    if mode == "none":
        return x.clamp(0, 1), step_data

    # crop_x = extract_crops(x, centre).detach().requires_grad_(True)
    # grad_crop = regularization_grad(crop_x, digit, mode, i, batch_id, t)
    # if  i % 2 == 0 and t==0 and mode == "both": #batch_id == 0 and
    #     plt.imshow(grad_crop[0,0].detach().cpu().numpy(), cmap="gray")
    #     plt.colorbar()
    #     plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_grad_crop_{name}.png"), dpi=300, bbox_inches="tight")
    #     plt.close()
    # grad_F = insert_crop_grad_in_image(x, grad_crop, centre)
    # x = (x - grad_F).clamp(0, 1)

    idx = 0 if i == 0 else 1

    grad_task = expected_task_gradient(
        x=x,
        i=i,
        digit_probs=digit,
        centre_map=centre,
        class_model=class_model,
        den_model=den_model,
        mode=mode,
        ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
        digit_threshold=0.1,
        centre_threshold=None,
        topk_centres=20,
        renormalize_centres=True,
        beta_class=beta1[idx],
        beta_deno=beta2[idx],
        sigma_data=sigma_data,
        #alpha = 1 + (10 - 1) * (i / (I - 1))
    )
    
    x = (x - grad_task).clamp(0, 1).detach()
    return x, step_data

###################################################################### Multi-task function #####################################################################

def model(y, batch_id, x_true, centre_true, digit_true):
    images_SIRT = []
    images_class = []
    images_deno = []
    images = []
    centers = []
    digits = []
    centers_sirt_all = []
    digits_sirt_all = []
    psnr_history = []
    ssim_history = []   
    digit_acc_history = []
    centre_mse_history = []
    
    

    B = y.shape[0]


    ##### Initilizing tasks


    # RECONSTRUCTION
    x = torch.zeros(B, 1, H, W, device=y.device)
    

    if accelerator.is_main_process and batch_id == 0:
        dx_norm_mean = torch.zeros(T_ini, device=y.device) 
        dx_norm_b0 = torch.zeros(T_ini, device=y.device)  

    for t in range(T_ini):
        x_prev = x
        x , step_data = sirt_step(x, y, alpha)
        if accelerator.is_main_process and batch_id ==0 and t % 5 == 0:
            logger.info(f"Gradient descent step -1_{t}: {step_data.flatten(1).norm(dim=1).mean().item()}")

        # debugging || x^{k+1} -x^k||
        if accelerator.is_main_process and batch_id == 0:
            dx = x - x_prev 
            dx_norm_b0[t]   = dx[0].flatten().norm()
            dx_norm_mean[t] = dx.flatten(1).norm(dim=1).mean().item()


    # metrics after the initialization with SIRT (equi to I = -1, with T_ini steps)
    crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
    crop_x_sirt = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
    for b in range(B):
        cy = int(centre_true[b, 0].item())
        cx = int(centre_true[b, 1].item())
        crop_x[b:b+1] = x_true[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
        crop_x_sirt[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
        
    psnr_x_init = compute_psnr_results(x.detach(), x_true.detach())
    ssim_sirt_init = compute_ssim(x.detach(), x_true.detach(), data_range=1.0)
    psnr_x_init_crop = compute_psnr_results(crop_x_sirt.detach(), crop_x.detach())
    ssim_sirt_init_crop = compute_ssim(crop_x_sirt.detach(), crop_x.detach(), data_range=1.0)

    ssim_history.append({
        "I": -1,
        "num_inner_steps": T_ini,
        "ssim_x": ssim_sirt_init.detach().cpu(),
        "ssim_x_sirt": ssim_sirt_init.detach().cpu(),
        "ssim_x_clas": ssim_sirt_init.detach().cpu(),
        "ssim_x_deno": ssim_sirt_init.detach().cpu(),
        "ssim_x_crop": ssim_sirt_init_crop.detach().cpu(),
        "ssim_x_sirt_crop": ssim_sirt_init_crop.detach().cpu(),
        "ssim_x_clas_crop": ssim_sirt_init_crop.detach().cpu(),
        "ssim_x_deno_crop": ssim_sirt_init_crop.detach().cpu()

    })

    if accelerator.is_main_process:
        logger.info(
            f"[PSNR] I=-1 | steps={T_ini} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_x_init.mean().item():.4f} dB"
        )
        logger.info(
            f"[SSIM] I=-1 | steps={T_ini} | x={ssim_sirt_init.mean().item():.4f} | x_sirt={ssim_sirt_init.mean().item():.4f}"
        )


    psnr_history.append({
        "I": -1,
        "num_inner_steps": T_ini,
        "psnr_x": psnr_x_init.detach().cpu(),
        "psnr_x_sirt": psnr_x_init.detach().cpu(),
        "psnr_x_clas": psnr_x_init.detach().cpu(),
        "psnr_x_deno": psnr_x_init.detach().cpu(),
        "psnr_x_crop": psnr_x_init_crop.detach().cpu(),
        "psnr_x_sirt_crop": psnr_x_init_crop.detach().cpu(),
        "psnr_x_clas_crop": psnr_x_init_crop.detach().cpu(),
        "psnr_x_deno_crop": psnr_x_init_crop.detach().cpu()
    })

    if accelerator.is_main_process and batch_id == 0:
        plt.figure() 
        plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch")
        plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("$||x_{t+1}-x_t||$")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_initial_dx_norm_batch0_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()


    # x for only SIRT
    x = x.clamp(0,1)
    x_both = x.clone()
    x_sirt = x.clone()
    x_class = x.clone()
    x_deno = x.clone()

    # #if batch_id == 0:
    # plt.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")
    # plt.colorbar()
    # plt.savefig(os.path.join(results_dir, f"{batch_id}_first_sirt_image_{name}.png"), dpi=300, bbox_inches="tight")
    # plt.close()

    # DIGIT
    digit = torch.full((B, 10), 0.1, device=y.device)

    centre = torch.zeros((B, H, W), device=x.device)
    area = (ymax - ymin + 1) * (xmax - xmin + 1)
    centre[:, ymin:ymax+1, xmin:xmax+1] = 1.0 / area
    centre_sirt = centre.clone()
    eps = 1e-12

    
    # CENTRE
    with torch.no_grad():
        # map_x[b, y, x] will store a "probability" for placing the 28x28 crop center at (y, x).
        # We only fill valid centers (14..85) so the crop stays inside the 100x100 image.
        map_x = score_centers_with_unfold(
                        model =class_model,
                        img = x_both,
                        digit_cond=digit,      # [B,10], one-hot o soft
                        denoiser=den_model,
                        chunk_size=512  
                    )

    map_digit = torch.zeros(B, 10, device=y.device, dtype=x.dtype)

    W_valid = xmax - xmin + 1

    with torch.no_grad():
        for b in range(B):
            probs_centres = centre[b, ymin:ymax+1, xmin:xmax+1].reshape(-1)
            k = min(5, probs_centres.numel())
        

            vals, idxs = torch.topk(probs_centres, k=k)

            crops_topk = torch.zeros(k, 1, 28, 28, device=y.device, dtype=x.dtype)

            for j, idx in enumerate(idxs):
                idx = idx.item()
                iy = idx // W_valid
                ix = idx % W_valid

                cy = ymin + iy
                cx = xmin + ix

                crops_topk[j:j+1] = x_both[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

            logits = class_model(crops_topk)
            probs_digits = torch.softmax(logits, dim=1)

            weights = vals / vals.sum().clamp_min(1e-12)
            map_digit[b] = (weights.unsqueeze(1) * probs_digits).sum(dim=0)


    digit = map_digit
    digit_sirt = digit.clone()
    centre = map_x
    centre_sirt = centre.clone()
    
    
    if accelerator.is_main_process and batch_id == 0:
        logger.info(f"First digit: {digit}") 
        logger.info(f"First centre: {centre}")


    digit_prob_true = digit.gather(1, digit_true.unsqueeze(1)).squeeze(1)
    digit_nll = -torch.log(digit_prob_true.clamp_min(eps))
    digit_entropy = -(digit * torch.log(digit.clamp_min(eps))).sum(dim=1)

    
    # Wasserstein-1
    centre_true = centre_true.long()

    Hc, Wc = centre.shape[1], centre.shape[2]

    ys = torch.arange(Hc, device=centre.device).view(1, Hc, 1).float()
    xs = torch.arange(Wc, device=centre.device).view(1, 1, Wc).float()

    cy_true = centre_true[:, 0].view(B, 1, 1).float()
    cx_true = centre_true[:, 1].view(B, 1, 1).float()

    dist_map = torch.sqrt((ys - cy_true)**2 + (xs - cx_true)**2)   # [B,H,W]
    centre_w1 = (centre * dist_map).sum(dim=(1, 2))   # [B]

    

    digit_acc_history.append({
        "I": -1,
        "digit_nll": digit_nll.detach().cpu(),
        "digit_prob_true": digit_prob_true.detach().cpu(),
        "digit_entropy": digit_entropy.detach().cpu(),
        "digit_nll_sirt": digit_nll.detach().cpu(),
        "digit_prob_true_sirt": digit_prob_true.detach().cpu(),
        "digit_entropy_sirt": digit_entropy.detach().cpu()
        })

    centre_mse_history.append({
        "I": -1,
        "centre_w1": centre_w1.detach().cpu(),
        "centre_w1_sirt": centre_w1.detach().cpu()
    })

    

    images.append(x.detach().cpu())
    images_SIRT.append(x.detach().cpu())
    centers.append(centre.detach().cpu())
    digits.append(digit.detach().cpu())
    centers_sirt_all.append(centre_sirt.detach().cpu())
    digits_sirt_all.append(digit_sirt.detach().cpu())

    ###### model loop

    for i in range(I):
        if i < decay_iters:
            T = int(T_max - (T_max - T_min) * (i / (decay_iters - 1)))
        else:
            T = T_min

        if accelerator.is_main_process and batch_id == 0 and i % 5 == 0:
            dx_norm_mean = torch.zeros(T, device=y.device) 
            dx_norm_b0 = torch.zeros(T, device=y.device)  # una muestra
            dx_norm_mean_sirt = torch.zeros(T, device=y.device)
            dx_norm_b0_sirt = torch.zeros(T, device=y.device)  # una muestra
        
        # reconstruction
        for t in range(T):
            x_prev_both = x_both 
            x_both , step_data = update_case(x_both, y, centre, digit, alpha, i, t, batch_id, "both")

        
            # debugging || x^{k+1} -x^k||
            if accelerator.is_main_process and batch_id == 0 and i % 5 == 0:
                logger.info(f"Gradient descent step {i}_{t}: {step_data.flatten(1).norm(dim=1).mean().item()}")

                dx = x_both - x_prev_both 
                dx_norm_b0[t] = dx[0].flatten().norm()          # norma para el primer elemento del batch
                dx_norm_mean[t] = dx.flatten(1).norm(dim=1).mean().item()                  # media batch


            #only SIRT
            x_sirt_prev = x_sirt
            x_sirt , _ = update_case(x_sirt, y, centre, digit, alpha, i, t, batch_id, "none")

            # debugging || x^{k+1} -x^k||
            if accelerator.is_main_process and batch_id == 0 and i % 5 == 0:
                dx = x_sirt - x_sirt_prev 
                dx_norm_b0_sirt[t] = dx[0].flatten().norm()          # norma por elemento del batch
                dx_norm_mean_sirt[t] = dx.flatten(1).norm(dim=1).mean()            # media batch


            #only classifier
            x_class , _ = update_case(x_class, y, centre, digit, alpha, i, t, batch_id, "class")


            #only denoiser
            x_deno , _ = update_case(x_deno, y, centre, digit, alpha, i, t, batch_id, "deno")

        crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        crop_x_both = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        crop_x_sirt = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        crop_x_class = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        crop_x_deno = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        for b in range(B):
            cy = int(centre_true[b, 0].item())
            cx = int(centre_true[b, 1].item())
            crop_x[b:b+1] = x_true[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
            crop_x_both[b:b+1] = x_both[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
            crop_x_sirt[b:b+1] = x_sirt[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
            crop_x_class[b:b+1] = x_class[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
            crop_x_deno[b:b+1] = x_deno[b:b+1, :, cy-14:cy+14, cx-14:cx+14]


        # métricas 
        psnr_x_init = compute_psnr_results(x_both.detach(), x_true.detach())
        psnr_sirt_init = compute_psnr_results(x_sirt.detach(), x_true.detach())
        psnr_x_clas = compute_psnr_results(x_class.detach(), x_true.detach())
        psnr_x_deno = compute_psnr_results(x_deno.detach(), x_true.detach())
        ssim_x_init = compute_ssim(x_both.detach(), x_true.detach(), data_range=1.0)
        ssim_sirt_init = compute_ssim(x_sirt.detach(), x_true.detach(), data_range=1.0)
        ssim_x_clas = compute_ssim(x_class.detach(), x_true.detach(), data_range=1.0)
        ssim_x_deno = compute_ssim(x_deno.detach(), x_true.detach(), data_range=1.0)
        psnr_x_init_crop = compute_psnr_results(crop_x_both.detach(), crop_x.detach())
        psnr_sirt_init_crop = compute_psnr_results(crop_x_sirt.detach(), crop_x.detach())
        psnr_x_clas_crop = compute_psnr_results(crop_x_class.detach(), crop_x.detach())
        psnr_x_deno_crop = compute_psnr_results(crop_x_deno.detach(), crop_x.detach())
        ssim_x_init_crop = compute_ssim(crop_x_both.detach(), crop_x.detach(), data_range=1.0)
        ssim_sirt_init_crop = compute_ssim(crop_x_sirt.detach(), crop_x.detach(), data_range=1.0)
        ssim_x_clas_crop = compute_ssim(crop_x_class.detach(), crop_x.detach(), data_range=1.0)
        ssim_x_deno_crop = compute_ssim(crop_x_deno.detach(), crop_x.detach(), data_range=1.0)

        ssim_history.append({
            "I": i,
            "num_inner_steps": T,
            "ssim_x": ssim_x_init.detach().cpu(),
            "ssim_x_sirt": ssim_sirt_init.detach().cpu(),
            "ssim_x_clas": ssim_x_clas.detach().cpu(),
            "ssim_x_deno": ssim_x_deno.detach().cpu(),
            "ssim_x_crop": ssim_x_init_crop.detach().cpu(),
            "ssim_x_sirt_crop": ssim_sirt_init_crop.detach().cpu(),
            "ssim_x_clas_crop": ssim_x_clas_crop.detach().cpu(),
            "ssim_x_deno_crop": ssim_x_deno_crop.detach().cpu()
            
        })

        if batch_id == 0 and accelerator.is_main_process:
            logger.info(
                f"[PSNR] I={i} | steps={T} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_sirt_init.mean().item():.4f} dB"
            )
            logger.info(
                f"[SSIM] I={i} | steps={T} | x={ssim_x_init.mean().item():.4f} | x_sirt={ssim_sirt_init.mean().item():.4f}"
            )

        psnr_history.append({
            "I": i,
            "num_inner_steps": T,
            "psnr_x": psnr_x_init.detach().cpu(),
            "psnr_x_sirt": psnr_sirt_init.detach().cpu(),
            "psnr_x_clas": psnr_x_clas.detach().cpu(),
            "psnr_x_deno": psnr_x_deno.detach().cpu(),
            "psnr_x_crop": psnr_x_init_crop.detach().cpu(),
            "psnr_x_sirt_crop": psnr_sirt_init_crop.detach().cpu(),
            "psnr_x_clas_crop": psnr_x_clas_crop.detach().cpu(),
            "psnr_x_deno_crop": psnr_x_deno_crop.detach().cpu()
        })


        if accelerator.is_main_process and batch_id == 0 and i % 2 == 0:
            plt.figure() 
            plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch") 
            plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
            plt.plot(dx_norm_mean_sirt.detach().cpu().numpy(), label="mean over batch SIRT")
            plt.plot(dx_norm_b0_sirt.detach().cpu().numpy(), label="sample 0 SIRT")
            plt.yscale("log")
            plt.xlabel("t")
            plt.ylabel("$||x_{t+1}-x_t||$")
            plt.legend()
            plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_dx_norm_batch0_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

        if accelerator.is_main_process and batch_id == 0 and i % 2 == 0: 
            for b in range(B):

                # --- SIRT ---
                fig, ax = plt.subplots()
                im = ax.imshow(x_sirt[b,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax)
                fig.savefig(
                    os.path.join(results_dir, f"{batch_id}_{i}_sample{b}_sirt_{name}.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close(fig)

                # --- BOTH ---
                fig, ax = plt.subplots()
                im = ax.imshow(x_both[b,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax)
                fig.savefig(
                    os.path.join(results_dir, f"{batch_id}_{i}_sample{b}_flow_{name}.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close(fig)

        

        #center 
        with torch.no_grad():
            # map_x[b, y, x] will store a "probability" for placing the 28x28 crop center at (y, x).
            # We only fill valid centers (14..85) so the crop stays inside the 100x100 image.
            map_x = score_centers_with_unfold(
                            model =class_model,
                            img = x_both,
                            digit_cond=digit,      # [B,10], one-hot o soft
                            denoiser=den_model,
                            chunk_size=512  
                        )
            map_x_sirt = score_centers_with_unfold(
                            model =class_model,
                            img = x_sirt,
                            digit_cond=digit_sirt,      # [B,10], one-hot o soft
                            denoiser=den_model,
                            chunk_size=512  
                        )

        

        # digit 
        map_digit = torch.zeros(B, 10, device=y.device, dtype=x.dtype)
        map_digit_sirt = torch.zeros(B, 10, device=y.device, dtype=x.dtype)

        W_valid = xmax - xmin + 1

        with torch.no_grad():
            for b in range(B):
                probs_centres = centre[b, ymin:ymax+1, xmin:xmax+1].reshape(-1)
                probs_centres_sirt = centre_sirt[b, ymin:ymax+1, xmin:xmax+1].reshape(-1)
                k = min(5, probs_centres.numel())
                k_sirt = min(5, probs_centres_sirt.numel())

                vals, idxs = torch.topk(probs_centres, k=k)
                vals_sirt, idxs_sirt = torch.topk(probs_centres_sirt, k=k_sirt)

                crops_topk = torch.zeros(k, 1, 28, 28, device=y.device, dtype=x.dtype)
                crops_topk_sirt = torch.zeros(k_sirt, 1, 28, 28, device=y.device, dtype=x.dtype)

                for j, idx in enumerate(idxs):
                    idx = idx.item()
                    iy = idx // W_valid
                    ix = idx % W_valid

                    cy = ymin + iy
                    cx = xmin + ix

                    crops_topk[j:j+1] = x_both[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

                for j, idx in enumerate(idxs_sirt):
                    idx = idx.item()
                    iy = idx // W_valid
                    ix = idx % W_valid

                    cy = ymin + iy
                    cx = xmin + ix
                    crops_topk_sirt[j:j+1] = x_sirt[b:b+1, :, cy-14:cy+14, cx-14:cx+14]


                logits = class_model(crops_topk)
                probs_digits = torch.softmax(logits, dim=1)

                weights = vals / vals.sum().clamp_min(1e-12)
                map_digit[b] = (weights.unsqueeze(1) * probs_digits).sum(dim=0)

                logits_sirt = class_model(crops_topk_sirt)
                probs_digits_sirt = torch.softmax(logits_sirt, dim=1)

                weights_sirt = vals_sirt / vals_sirt.sum().clamp_min(1e-12)
                map_digit_sirt[b] = (weights_sirt.unsqueeze(1) * probs_digits_sirt).sum(dim=0)


        digit = map_digit
        digit_sirt = map_digit_sirt
        centre = map_x
        centre_sirt = map_x_sirt

        digit_prob_true = digit.gather(1, digit_true.unsqueeze(1)).squeeze(1)
        digit_nll = -torch.log(digit_prob_true.clamp_min(eps))
        digit_entropy = -(digit * torch.log(digit.clamp_min(eps))).sum(dim=1)
        #centre_prob_true = centre[ torch.arange(B, device=centre.device), centre_true[:, 0], centre_true[:, 1]]   # [B]
        # Wasserstein-1
        centre_true = centre_true.long()

        Hc, Wc = centre.shape[1], centre.shape[2]

        ys = torch.arange(Hc, device=centre.device).view(1, Hc, 1).float()
        xs = torch.arange(Wc, device=centre.device).view(1, 1, Wc).float()

        cy_true = centre_true[:, 0].view(B, 1, 1).float()
        cx_true = centre_true[:, 1].view(B, 1, 1).float()

        dist_map = torch.sqrt((ys - cy_true)**2 + (xs - cx_true)**2)   # [B,H,W]

        centre_w1 = (centre * dist_map).sum(dim=(1, 2))   # [B]


        digit_prob_true_sirt = digit_sirt.gather(1, digit_true.unsqueeze(1)).squeeze(1)
        digit_nll_sirt = -torch.log(digit_prob_true_sirt.clamp_min(eps))
        digit_entropy_sirt = -(digit_sirt * torch.log(digit_sirt.clamp_min(eps))).sum(dim=1)
        #centre_prob_true = centre[ torch.arange(B, device=centre.device), centre_true[:, 0], centre_true[:, 1]]   # [B]
        # Wasserstein-1

        Hc, Wc = centre_sirt.shape[1], centre_sirt.shape[2]

        ys = torch.arange(Hc, device=centre_sirt.device).view(1, Hc, 1).float()
        xs = torch.arange(Wc, device=centre_sirt.device).view(1, 1, Wc).float()


        dist_map_sirt = torch.sqrt((ys - cy_true)**2 + (xs - cx_true)**2)   # [B,H,W]

        centre_w1_sirt = (centre_sirt * dist_map_sirt).sum(dim=(1, 2))   # [B]

        digit_acc_history.append({
            "I": i,
            "digit_nll": digit_nll.detach().cpu(),
            "digit_prob_true": digit_prob_true.detach().cpu(),
            "digit_entropy": digit_entropy.detach().cpu(),
            "digit_nll_sirt": digit_nll_sirt.detach().cpu(),
            "digit_prob_true_sirt": digit_prob_true_sirt.detach().cpu(),
            "digit_entropy_sirt": digit_entropy_sirt.detach().cpu()
            })

        centre_mse_history.append({
            "I": i,
            "centre_w1": centre_w1.detach().cpu(),
            "centre_w1_sirt": centre_w1_sirt.detach().cpu()
        })

        

        if accelerator.is_main_process and i % 5 == 0 and batch_id == 0:
            logger.info(f"{i}th digit: {digit}")
            logger.info(f"{i}th centre: {centre}")

        #save information
        images.append(x_both.detach().cpu())
        images_SIRT.append(x_sirt.detach().cpu())
        images_class.append(x_class.detach().cpu())
        images_deno.append(x_deno.detach().cpu())
        centers.append(centre.detach().cpu())
        digits.append(digit.detach().cpu())
        centers_sirt_all.append(centre_sirt.detach().cpu())
        digits_sirt_all.append(digit_sirt.detach().cpu())

        del crop_x, crop_x_both, crop_x_sirt, crop_x_class, crop_x_deno
        del map_x, logits, map_digit
        torch.cuda.empty_cache()

    return images, images_SIRT, images_class, images_deno, centers, digits, centers_sirt_all, digits_sirt_all, psnr_history, ssim_history, digit_acc_history, centre_mse_history



###################################################################### training part #####################################################################

gt_images = []
gt_digits = []
gt_centers = []
both_digits = []
sirt_digits = []
both_centers = []
sirt_centers = []
both_digit_prob_true = []
sirt_digit_prob_true = []
both_digit_nll = []
sirt_digit_nll = []
both_digit_entropy = []
sirt_digit_entropy = []
both_centre_w1 = []
sirt_centre_w1 = []
both_images = []
sirt_images = []
clas_images = []
deno_images = []
sirt_psnr = []
both_psnr = []
clas_psnr = []
deno_psnr = []
sirt_ssim = []
both_ssim = []
clas_ssim = []
deno_ssim = []
sirt_psnr_crop = []
both_psnr_crop = []
clas_psnr_crop = []
deno_psnr_crop = []
sirt_ssim_crop = []
both_ssim_crop = []
clas_ssim_crop = []
deno_ssim_crop = []

for batch_id, batch in enumerate(train_dataloader):

    ground_truth_1 = batch["ground_truth_image"].to(device)
    ground_truth_2 = batch["label"].to(device)
    ground_truth_3 = batch["centre"].to(device)
    input_data = batch["noisy_sinogram"].to(device)

    B = ground_truth_1.shape[0]


   
    # x = torch.zeros(B, 1, 100, 100, device=ground_truth_1.device)

    
    # for b in range(B):
    #     cy = int(ground_truth_3[b, 0].item())
    #     cx = int(ground_truth_3[b, 1].item())
    #     crop = ground_truth_1[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

    #     # sigma aleatorio entre 0.1 y 0.3
    #     sigma = torch.empty(1, device=crop.device).uniform_(0.1, 0.3)
    #     crop = crop + torch.randn_like(crop) * sigma
 
    #     # # columnas centrales
    #     crop[:, :, :, 12:15] = 0.0

    #     x[b:b+1, :, cy-14:cy+14, cx-14:cx+14] = crop

    #     # # máscara de píxeles no cero
    #     # nonzero_mask = crop != 0

    #     # # índices de esos píxeles
    #     # indices = nonzero_mask.nonzero(as_tuple=False)

    #     # # si hay suficientes píxeles no cero
    #     # if indices.shape[0] >= 5:
    #     #     # elegimos k aleatorios
    #     #     selected = indices[torch.randperm(indices.shape[0])[:30]]

    #     #     # los ponemos a 0
    #     #     for idx in selected:
    #     #         crop[idx[0], idx[1], idx[2], idx[3]] = 0.0

    #     # x[b:b+1, :, cy-14:cy+14, cx-14:cx+14] = crop
    
 
    if accelerator.is_main_process and batch_id == 0:
        logger.info(f"Original digit: {ground_truth_2}")
        logger.info(f"original centre: {ground_truth_3}")

    
    images, images_SIRT, images_class, images_deno, centers, digits, centers_sirt, digits_sirt, psnr_history, ssim_history, digit_acc_history, centre_mse_history = model(input_data, batch_id, ground_truth_1, ground_truth_3, ground_truth_2)
    
    if accelerator.is_main_process and batch_id ==0: 
        for b in range(B):

            # take last image from the list
            img = ground_truth_1[b,0].detach().cpu().numpy()
            img_both = images[-1][b, 0].detach().cpu().numpy()
            img_sirt = images_SIRT[-1][b, 0].detach().cpu().numpy()
            img_class = images_class[-1][b, 0].detach().cpu().numpy()
            img_deno = images_deno[-1][b, 0].detach().cpu().numpy()

            all_imgs = np.stack([img, img_sirt, img_both, img_class, img_deno])
            vmin = all_imgs.min()
            vmax = all_imgs.max()

            fig, axes = plt.subplots(1, 5, figsize=(20, 4))

            im = axes[0].imshow(img_sirt, cmap="gray", vmin=vmin, vmax=vmax)
            axes[0].set_title("SIRT only")
            axes[0].axis("off")

            axes[1].imshow(img_both, cmap="gray", vmin=vmin, vmax=vmax)
            axes[1].set_title("SIRT + classifier + denoiser")
            axes[1].axis("off")

            axes[2].imshow(img_class, cmap="gray", vmin=vmin, vmax=vmax)
            axes[2].set_title("SIRT + classifier")
            axes[2].axis("off")

            axes[3].imshow(img_deno, cmap="gray", vmin=vmin, vmax=vmax)
            axes[3].set_title("SIRT + denoiser")
            axes[3].axis("off")

            axes[4].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            axes[4].set_title("Ground Truth")
            axes[4].axis("off")


            # common color bar 
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax)

            fig.subplots_adjust(right=0.9)
            plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_comparison_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            I_vals = [d["I"] for d in psnr_history]
            psnr_x_vals = [d["psnr_x"][b].mean().item() for d in psnr_history]
            psnr_x_sirt_vals = [d["psnr_x_sirt"][b].mean().item() for d in psnr_history]
            psnr_x_clas_vals = [d["psnr_x_clas"][b].mean().item() for d in psnr_history]
            psnr_x_deno_vals = [d["psnr_x_deno"][b].mean().item() for d in psnr_history]

            plt.figure()
            plt.plot(I_vals, psnr_x_sirt_vals, marker="o", label="Mean SIRT")
            plt.plot(I_vals, psnr_x_vals, marker="o", label="Mean SIRT + classifier + denoiser")
            plt.plot(I_vals, psnr_x_clas_vals, marker="o", label="Mean SIRT + classifier")
            plt.plot(I_vals, psnr_x_deno_vals, marker="o", label="Mean SIRT + denoiser")
            plt.xlabel("I")
            plt.ylabel("PSNR (dB)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_psnr_curve_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

            
            psnr_x_crop_vals = [d["psnr_x_crop"][b].mean().item() for d in psnr_history]
            psnr_x_sirt_crop_vals = [d["psnr_x_sirt_crop"][b].mean().item() for d in psnr_history]
            psnr_x_clas_crop_vals = [d["psnr_x_clas_crop"][b].mean().item() for d in psnr_history]
            psnr_x_deno_crop_vals = [d["psnr_x_deno_crop"][b].mean().item() for d in psnr_history]

            plt.figure()
            plt.plot(I_vals, psnr_x_sirt_crop_vals, marker="o", label="Mean SIRT crop")
            plt.plot(I_vals, psnr_x_crop_vals, marker="o", label="Mean SIRT + classifier + denoiser crop")
            plt.plot(I_vals, psnr_x_clas_crop_vals, marker="o", label="Mean SIRT + classifier crop")
            plt.plot(I_vals, psnr_x_deno_crop_vals, marker="o", label="Mean SIRT + denoiser crop")
            plt.xlabel("I")
            plt.ylabel("PSNR crop (dB)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_psnr_crop_curve_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

            
            ssim_x_vals = [d["ssim_x"][b].mean().item() for d in ssim_history]
            ssim_x_sirt_vals = [d["ssim_x_sirt"][b].mean().item() for d in ssim_history]
            ssim_x_clas_vals = [d["ssim_x_clas"][b].mean().item() for d in ssim_history]
            ssim_x_deno_vals = [d["ssim_x_deno"][b].mean().item() for d in ssim_history]

            plt.figure()
            plt.plot(I_vals, ssim_x_sirt_vals, marker="o", label="Mean SIRT")
            plt.plot(I_vals, ssim_x_vals, marker="o", label="Mean SIRT + classifier + denoiser")
            plt.plot(I_vals, ssim_x_clas_vals, marker="o", label="Mean SIRT + classifier")
            plt.plot(I_vals, ssim_x_deno_vals, marker="o", label="Mean SIRT + denoiser")
            plt.xlabel("I")
            plt.ylabel("SSIM")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_ssim_curve_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

        
            ssim_x_crop_vals = [d["ssim_x_crop"][b].mean().item() for d in ssim_history]
            ssim_x_sirt_crop_vals = [d["ssim_x_sirt_crop"][b].mean().item() for d in ssim_history]
            ssim_x_clas_crop_vals = [d["ssim_x_clas_crop"][b].mean().item() for d in ssim_history]
            ssim_x_deno_crop_vals = [d["ssim_x_deno_crop"][b].mean().item() for d in ssim_history]

            plt.figure()
            plt.plot(I_vals, ssim_x_sirt_crop_vals, marker="o", label="Mean SIRT crop")
            plt.plot(I_vals, ssim_x_crop_vals, marker="o", label="Mean SIRT + classifier + denoiser crop")
            plt.plot(I_vals, ssim_x_clas_crop_vals, marker="o", label="Mean SIRT + classifier crop")
            plt.plot(I_vals, ssim_x_deno_crop_vals, marker="o", label="Mean SIRT + denoiser crop")
            plt.xlabel("I")
            plt.ylabel("SSIM crop")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_ssim_crop_curve_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()


        
            digit_prob_vals = [d["digit_prob_true"][b].mean().item() for d in digit_acc_history]
            digit_prob_vals_sirt = [d["digit_prob_true_sirt"][b].mean().item() for d in digit_acc_history]

            plt.figure()
            plt.plot(I_vals, digit_prob_vals, marker="o", label="Mean Digit prob (model)")
            plt.plot(I_vals, digit_prob_vals_sirt, marker="o", label="Mean Digit prob (SIRT)")
            plt.xlabel("I")
            plt.ylabel("Probability")
            plt.ylim(0.0, 1.0)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_digit_prob_curve_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

            digit_nll_vals = [d["digit_nll"][b].mean().item() for d in digit_acc_history]
            digit_nll_vals_sirt = [d["digit_nll_sirt"][b].mean().item() for d in digit_acc_history]

            plt.figure()
            plt.plot(I_vals, digit_nll_vals, marker="o", label="Mean Digit NLL (model)")
            plt.plot(I_vals, digit_nll_vals_sirt, marker="o", label="Mean Digit NLL (SIRT)")
            plt.xlabel("I")
            plt.ylabel("-log p(true)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_digit_nll_curve_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

            digit_margin_vals = [d["digit_entropy"][b].mean().item() for d in digit_acc_history]
            digit_margin_vals_sirt = [d["digit_entropy_sirt"][b].mean().item() for d in digit_acc_history]

            plt.figure()
            plt.plot(I_vals, digit_margin_vals, marker="o", label="Mean Digit entropy (model)")
            plt.plot(I_vals, digit_margin_vals_sirt, marker="o", label="Mean Digit entropy (SIRT)")
            plt.xlabel("I")
            plt.ylabel("Entropy")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_digit_entropy_curve_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()


            centre_w1_vals = [d["centre_w1"][b].mean().item() for d in centre_mse_history]
            centre_w1_vals_sirt = [d["centre_w1_sirt"][b].mean().item() for d in centre_mse_history]

            plt.figure()
            plt.plot(I_vals, centre_w1_vals, marker="o", label="Mean Centre W1 (model)")
            plt.plot(I_vals, centre_w1_vals_sirt, marker="o", label="Mean Centre W1 (SIRT)")
            plt.xlabel("I")
            plt.ylabel("Wasserstein-1 (pixels)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_centre_w1_curve_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()
        
    
            gt_images.append(ground_truth_1[b].detach().cpu())
            gt_digits.append(ground_truth_2[b].detach().cpu())
            gt_centers.append(ground_truth_3[b].detach().cpu())

            both_images.append(images[-1][b].detach().cpu())
            sirt_images.append(images_SIRT[-1][b].detach().cpu())
            clas_images.append(images_class[-1][b].detach().cpu())
            deno_images.append(images_deno[-1][b].detach().cpu())

            both_digits.append(digits[-1][b].detach().cpu())
            sirt_digits.append(digits_sirt[-1][b].detach().cpu())

            both_centers.append(centers[-1][b].detach().cpu())
            sirt_centers.append(centers_sirt[-1][b].detach().cpu())

            both_digit_prob_true.append([d["digit_prob_true"][b].item() for d in digit_acc_history])
            sirt_digit_prob_true.append([d["digit_prob_true_sirt"][b].item() for d in digit_acc_history])

            both_digit_nll.append([d["digit_nll"][b].item() for d in digit_acc_history])
            sirt_digit_nll.append([d["digit_nll_sirt"][b].item() for d in digit_acc_history])

            both_digit_entropy.append([d["digit_entropy"][b].item() for d in digit_acc_history])
            sirt_digit_entropy.append([d["digit_entropy_sirt"][b].item() for d in digit_acc_history])

            both_centre_w1.append([d["centre_w1"][b].item() for d in centre_mse_history])
            sirt_centre_w1.append([d["centre_w1_sirt"][b].item() for d in centre_mse_history])

            both_psnr.append([d["psnr_x"][b].item() for d in psnr_history])
            sirt_psnr.append([d["psnr_x_sirt"][b].item() for d in psnr_history])
            clas_psnr.append([d["psnr_x_clas"][b].item() for d in psnr_history])
            deno_psnr.append([d["psnr_x_deno"][b].item() for d in psnr_history])

            both_ssim.append([d["ssim_x"][b].item() for d in ssim_history])
            sirt_ssim.append([d["ssim_x_sirt"][b].item() for d in ssim_history])
            clas_ssim.append([d["ssim_x_clas"][b].item() for d in ssim_history])
            deno_ssim.append([d["ssim_x_deno"][b].item() for d in ssim_history])

            both_psnr_crop.append([d["psnr_x_crop"][b].item() for d in psnr_history])
            sirt_psnr_crop.append([d["psnr_x_sirt_crop"][b].item() for d in psnr_history])
            clas_psnr_crop.append([d["psnr_x_clas_crop"][b].item() for d in psnr_history])
            deno_psnr_crop.append([d["psnr_x_deno_crop"][b].item() for d in psnr_history])

            both_ssim_crop.append([d["ssim_x_crop"][b].item() for d in ssim_history])
            sirt_ssim_crop.append([d["ssim_x_sirt_crop"][b].item() for d in ssim_history])
            clas_ssim_crop.append([d["ssim_x_clas_crop"][b].item() for d in ssim_history])
            deno_ssim_crop.append([d["ssim_x_deno_crop"][b].item() for d in ssim_history])

results = {
    "T_ini": T_ini,
    "T" : [d["num_inner_steps"] for d in psnr_history],
    "I" : I,
    "sigma_data": sigma_data,
    "angles": angles,
    "gt_images": gt_images,
    "gt_digits": gt_digits,
    "gt_centers": gt_centers,
    "both_images": both_images,      # [batch][iter][B,1,H,W]
    "sirt_images": sirt_images,
    "clas_images": clas_images,
    "deno_images": deno_images,
    "sirt_psnr": sirt_psnr,
    "both_psnr": both_psnr,
    "clas_psnr": clas_psnr,
    "deno_psnr": deno_psnr,
    "sirt_ssim": sirt_ssim,
    "both_ssim": both_ssim,
    "clas_ssim": clas_ssim,
    "deno_ssim": deno_ssim,
    "sirt_psnr_crop": sirt_psnr_crop,
    "both_psnr_crop": both_psnr_crop,
    "clas_psnr_crop": clas_psnr_crop,
    "deno_psnr_crop": deno_psnr_crop,
    "sirt_ssim_crop": sirt_ssim_crop,
    "both_ssim_crop": both_ssim_crop,
    "clas_ssim_crop": clas_ssim_crop,
    "deno_ssim_crop": deno_ssim_crop,
    "both_digits": both_digits,
    "both_centers": both_centers,
    "sirt_digits": sirt_digits,
    "sirt_centers": sirt_centers,
    "both_digit_prob_true": both_digit_prob_true,
    "sirt_digit_prob_true": sirt_digit_prob_true,
    "both_digit_nll": both_digit_nll,
    "sirt_digit_nll": sirt_digit_nll,
    "both_digit_entropy": both_digit_entropy,
    "sirt_digit_entropy": sirt_digit_entropy,
    "both_centre_w1": both_centre_w1,
    "sirt_centre_w1": sirt_centre_w1
    }


rank = accelerator.process_index
torch.save( results, os.path.join(results_dir, f"results_mnist_ct_{name}_rank{rank}.pt" ))

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        finally:
            logger.removeHandler(h)

    logging.shutdown()
    
accelerator.wait_for_everyone()

if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()