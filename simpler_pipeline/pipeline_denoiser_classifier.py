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





###################################################################### Parameters #####################################################################

### data
seed = 29072000
H = 100
W = 100
num_angles = 60 #60
i_0 = 10**5
sigma_data = 0.3


### training parameters
T_ini = 80
T = 70
I = 10
tol_dx = 1e-3
patience_dx = 3
alpha = 2e-1/(75.77664947509766)**2 #1/(liptchitz constant)^2
beta1 = [0.001, 1e1]
beta2 = [0.001, 1e-2] #[5e-4, 5e-3]
device = "cuda"
n_train = 10
batch_size = 1
save_batch_id = 0
k = int(0.05 * H * W)   # top 5%
results_dir = "simpler_pipeline/results/ExperimentsClassDeno/Experiments4/60_angles_sigma_0.3"


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
name = "60_angles_sigma_0.3_10_1e-2"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"{name}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)




#Check if the dataset is balanced
labels_bal = torch.tensor([train_data[i]["label"] for i in range(len(train_data))])
counts = torch.bincount(labels_bal, minlength=10)

logger.info("Balanced dataset label counts:")
for d in range(10):
    logger.info(f"Digit {d}: {counts[d].item()}")


###################################################################### Trained Classifier and Denoiser Models #####################################################################

class_model = CNNClassifier()
class_model = class_model.to(device)

checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/classifier/cnn_classifier_best.pt", map_location=device)
class_model.load_state_dict(checkpoint)

class_model.eval() 
for p in class_model.parameters():
   p.requires_grad_(False)

den_model = ResUNetDenoiser()
den_model = den_model.to(device)

checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/denoiser/dncnn_best.pt", map_location=device)
den_model.load_state_dict(checkpoint)

den_model.eval() 
for p in den_model.parameters():
   p.requires_grad_(False)



###################################################################### Center finder function #####################################################################
@torch.no_grad()
def score_centers_with_unfold(
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


def insert_crop_grad_in_image(x, grad_crop, centre, fill_mode="scaled_x"):
    grad_F = torch.zeros_like(x)

    if fill_mode == "scaled_x":
        x_scaled = x.clone()
        scale = grad_crop.norm() / (x_scaled.norm() + 1e-8)
        grad_F = 10 * x_scaled * scale

    for b in range(x.shape[0]):
        cy = int(centre[b, 0].item())
        cx = int(centre[b, 1].item())
        grad_F[b:b+1, :, cy-14:cy+14, cx-14:cx+14] = grad_crop[b:b+1]

    return grad_F


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
        logger.info(f"Gradient descent step {batch_id}_{i}_{t}| classifier : {beta1[idx] * grad_classi.flatten(1).norm(dim=1).mean().item()}, denoiser : {beta2[idx] * grad_deno.flatten(1).norm(dim=1).mean().item()}")
    elif mode == "class":
        grad_crop = -(sigma_data**2) * (beta1[idx] * grad_classi)
        logger.info(f"Gradient descent step {batch_id}_{i}_{t}| classifier : {beta1[idx] * grad_classi.flatten(1).norm(dim=1).mean().item()}")
    elif mode == "deno":
        grad_crop = -(sigma_data**2) * (beta2[idx] * grad_deno)
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

    crop_x = extract_crops(x, centre).detach().requires_grad_(True)
    grad_crop = regularization_grad(crop_x, digit, mode, i, batch_id, t)
    if  i % 2 == 0 and t==0 and mode == "both": #batch_id == 0 and
        plt.imshow(grad_crop[0,0].detach().cpu().numpy(), cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_grad_crop_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()
    grad_F = insert_crop_grad_in_image(x, grad_crop, centre)
    x = (x - grad_F).clamp(0, 1)

    return x, step_data

###################################################################### Multi-task function #####################################################################

def model(y, batch_id, x_true, centre, digit):
    images_SIRT = []
    images_class = []
    images_deno = []
    images = []
    centers = []
    digits = []
    psnr_history = []
    ssim_history = []   
    

    B = y.shape[0]


    ##### Initilizing tasks


    # RECONSTRUCTION
    x = torch.zeros(B, 1, H, W, device=y.device)
    

    if batch_id == 0:
        dx_norm_mean = torch.zeros(T_ini, device=y.device) ################
        dx_norm_b0 = torch.zeros(T_ini, device=y.device)  # una muestra

    for t in range(T_ini):
        x_prev = x
        x , step_data = sirt_step(x, y, alpha)
        if batch_id ==0:
            logger.info(f"Gradient descent step -1_{t}: {step_data.flatten(1).norm(dim=1).mean().item()}")

        # debugging || x^{k+1} -x^k||
        if batch_id == 0:
            dx = x - x_prev ################
            dx_norm_b0[t]   = dx[0].flatten().norm()
            dx_norm_mean[t] = dx.flatten(1).norm(dim=1).mean()


    # métricas tras la inicialización SIRT (equivale a I = -1, con T_ini pasos)
    crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
    crop_x_sirt = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
    for b in range(B):
        cy = int(centre[b, 0].item())
        cx = int(centre[b, 1].item())
        crop_x[b:b+1] = x_true[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
        crop_x_sirt[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
        
    psnr_x_init = compute_psnr_results(x.detach(), x_true.detach())
    ssim_sirt_init = compute_ssim(x.detach(), x_true.detach(), data_range=1.0)
    psnr_x_init_crop = compute_psnr_results(crop_x_sirt.detach(), crop_x.detach())
    ssim_sirt_init_crop = compute_ssim(crop_x_sirt.detach(), crop_x.detach(), data_range=1.0)

    ssim_history.append({
        "I": -1,
        "num_inner_steps": T_ini,
        "ssim_x": ssim_sirt_init,
        "ssim_x_sirt": ssim_sirt_init,
        "ssim_x_clas": ssim_sirt_init,
        "ssim_x_deno": ssim_sirt_init,
        "ssim_x_crop": ssim_sirt_init_crop,
        "ssim_x_sirt_crop": ssim_sirt_init_crop,
        "ssim_x_clas_crop": ssim_sirt_init_crop,
        "ssim_x_deno_crop": ssim_sirt_init_crop

    })

    logger.info(
        f"[SSIM] I=-1 | steps={T_ini} | x={ssim_sirt_init:.4f}  | x_sirt={ssim_sirt_init:.4f} "
    )

    psnr_history.append({
        "I": -1,
        "num_inner_steps": T_ini,
        "psnr_x": psnr_x_init,
        "psnr_x_sirt": psnr_x_init,
        "psnr_x_clas": psnr_x_init,
        "psnr_x_deno": psnr_x_init,
        "psnr_x_crop": psnr_x_init_crop,
        "psnr_x_sirt_crop": psnr_x_init_crop,
        "psnr_x_clas_crop": psnr_x_init_crop,
        "psnr_x_deno_crop": psnr_x_init_crop
    })

    logger.info(
        f"[PSNR] I=-1 | steps={T_ini} | x={psnr_x_init:.4f} dB | x_sirt={psnr_x_init:.4f} dB"
    )


    if batch_id == 0:
        plt.figure() ################
        plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch")
        plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("$||x_{t+1}-x_t||$")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_initial_dx_norm_batch0_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()


    #if batch_id == 0:
    plt.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")################
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, f"{batch_id}_first_sirt_image_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # x for only SIRT
    x = x.clamp(0,1)
    x_both = x.clone()
    x_sirt = x.clone()
    x_class = x.clone()
    x_deno = x.clone()

    # DIGIT
    #digit = torch.randint(0, 10, (B,), device=y.device)
    # digit = torch.full((B,), 3, device=y.device, dtype=torch.long)
    #digit = torch.nn.functional.one_hot(digit, num_classes=10).float()
    digit = digit.to(x.device)

    #if batch_id == 0:
    logger.info(f"First digit: {digit}") ################

    # CENTER
    # xmin, xmax = 14, 85
    # ymin, ymax = 14, 85
    # centre = torch.randint(xmin, xmax+1, (B, 2), device=y.device)
    # centre = torch.ones((B, 2), device=y.device)*14
    centre = centre.to(x.device)


    #if batch_id == 0:
    logger.info(f"First centre: {centre}") ################


    images.append(x.detach().cpu())
    images_SIRT.append(x.detach().cpu())
    centers.append(centre.detach().cpu())
    digits.append(digit.detach().cpu())

    ###### model loop

    for i in range(I):

        if batch_id == 0:
            dx_norm_mean = torch.zeros(T, device=y.device) ################
            dx_norm_b0 = torch.zeros(T, device=y.device)  # una muestra
            dx_norm_mean_sirt = torch.zeros(T, device=y.device)
            dx_norm_b0_sirt = torch.zeros(T, device=y.device)  # una muestra

        # reconstruction
        for t in range(T):
            x_prev_both = x_both 
            x_both , step_data = update_case(x_both, y, centre, digit, alpha, i, t, batch_id, "both")
            if batch_id ==0:
                logger.info(f"Gradient descent step {i}_{t}: {step_data.flatten(1).norm(dim=1).mean().item()}")

        
            # debugging || x^{k+1} -x^k||
            if batch_id == 0:
                dx = x_both - x_prev_both ################
                dx_norm_b0[t] = dx[0].flatten().norm()          # norma para el primer elemento del batch
                dx_norm_mean[t] = dx.flatten(1).norm(dim=1).mean()                  # media batch


            #only SIRT
            x_sirt_prev = x_sirt
            x_sirt , _ = update_case(x_sirt, y, centre, digit, alpha, i, t, batch_id, "none")

            # debugging || x^{k+1} -x^k||
            if batch_id == 0:
                dx = x_sirt - x_sirt_prev ################
                dx_norm_b0_sirt[t] = dx[0].flatten().norm()          # norma por elemento del batch
                dx_norm_mean_sirt[t] = dx.flatten(1).norm(dim=1).mean()             # media batch


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
            cy = int(centre[b, 0].item())
            cx = int(centre[b, 1].item())
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
            "ssim_x": ssim_x_init,
            "ssim_x_sirt": ssim_sirt_init,
            "ssim_x_clas": ssim_x_clas,
            "ssim_x_deno": ssim_x_deno,
            "ssim_x_crop": ssim_x_init_crop,
            "ssim_x_sirt_crop": ssim_sirt_init_crop,
            "ssim_x_clas_crop": ssim_x_clas_crop,
            "ssim_x_deno_crop": ssim_x_deno_crop
            
        })

        logger.info(
            f"[SSIM] I={i} | steps={T} | x={ssim_x_init:.4f}  | x_sirt={ssim_sirt_init:.4f} "
        )

        psnr_history.append({
            "I": i,
            "num_inner_steps": T,
            "psnr_x": psnr_x_init,
            "psnr_x_sirt": psnr_sirt_init,
            "psnr_x_clas": psnr_x_clas,
            "psnr_x_deno": psnr_x_deno,
            "psnr_x_crop": psnr_x_init_crop,
            "psnr_x_sirt_crop": psnr_sirt_init_crop,
            "psnr_x_clas_crop": psnr_x_clas_crop,
            "psnr_x_deno_crop": psnr_x_deno_crop
        })

        logger.info(
            f"[PSNR] I={i} | steps={T} | x={psnr_x_init:.4f} dB | x_sirt={psnr_sirt_init:.4f} dB"
        )

        if batch_id == 0:
            plt.figure() ################
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

        if  i % 2 == 0: #batch_id == 0 and
            plt.imshow(x_sirt[0,0].detach().cpu().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_sirt__{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

            plt.imshow(x_both[0,0].detach().cpu().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_flow_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

        

        #center 
        # with torch.no_grad():
        #     # map_x[b, y, x] will store a "score" for placing the 28x28 crop center at (y, x).
        #     # We only fill valid centers (14..85) so the crop stays inside the 100x100 image.
        #     map_x = map_x = score_centers_with_unfold(
        #                     model =class_model,
        #                     img = x,
        #                     chunk_size=512  # prueba 256/512/1024 según memoria
        #                 )

        # flat = map_x.view(B, -1)          # [B, H*W]
        # best_idx = flat.argmax(dim=1)     # [B]

        # cy = best_idx // W
        # cx = best_idx %  W

        # centre_new = torch.stack([cy, cx], dim=1)   # [B,2]
        # centre_new = torch.stack([cy, cx], dim=1)                     # [B, 2]
        # centre_new[:,0] = centre_new[:,0].clamp(14, H-14)
        # centre_new[:,1] = centre_new[:,1].clamp(14, W-14)

        

        # # digit 
        # crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        # for b in range(B):
        #     cy = int(centre[b, 0].item())
        #     cx = int(centre[b, 1].item())
        #     crop_x[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

        # with torch.no_grad():
        #     logits = class_model(crop_x)                  # [B,10]
        #     map_digit = torch.softmax(logits, dim=1)      # [B,10]

        # digit = map_digit.argmax(dim=1)
        # centre = centre_new

        ## digit = torch.full((B,), 3, device=y.device, dtype=torch.long)
        ## digit = torch.nn.functional.one_hot(digit, num_classes=10).float()
        ## centre = torch.ones((B, 2), device=y.device)*14


        if i % 2 == 0: #batch_id == 0 and
            logger.info(f"{i}th digit: {digit}")
            logger.info(f"{i}th centre: {centre}")

        #save information
        images.append(x_both.detach().cpu())
        images_SIRT.append(x_sirt.detach().cpu())
        images_class.append(x_class.detach().cpu())
        images_deno.append(x_deno.detach().cpu())
        centers.append(centre.detach().cpu())
        digits.append(digit.detach().cpu())

    return images, images_SIRT, images_class, images_deno, centers, digits, psnr_history, ssim_history



###################################################################### training part #####################################################################

gt_images = []
gt_digits = []
gt_centers = []
both_images = []
model_digits = []
model_centers = []
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

    ground_truth_1 = batch["ground_truth_image"]
    ground_truth_2 = batch["label"]
    ground_truth_3 = batch["centre"]
    input_data = batch["noisy_sinogram"]

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
    
 
    if batch_id == 0 :
        logger.info(f"Original digit: {ground_truth_2}")
        logger.info(f"original centre: {ground_truth_3}")

    #if 1 in ground_truth_2 :
    images, images_SIRT, images_class, images_deno, centers, digits, psnr_history, ssim_history = model(input_data, batch_id, ground_truth_1, ground_truth_3, ground_truth_2)
    
    # coger la última imagen de cada lista
    img = ground_truth_1[0,0].detach().cpu().numpy()
    img_both = images[-1][0, 0].numpy()
    img_sirt = images_SIRT[-1][0, 0].numpy()
    img_class = images_class[-1][0, 0].numpy()
    img_deno = images_deno[-1][0, 0].numpy()
    # stack para calcular rango global
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


    # barra de color común
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(results_dir, f"{batch_id}_comparison_{name}.png"), dpi=300, bbox_inches="tight")

    I_vals = [d["I"] for d in psnr_history]
    psnr_x_vals = [d["psnr_x"] for d in psnr_history]
    psnr_x_sirt_vals = [d["psnr_x_sirt"] for d in psnr_history]

    plt.figure()
    plt.plot(I_vals, psnr_x_vals, marker="o", label="x")
    plt.plot(I_vals, psnr_x_sirt_vals, marker="o", label="x SIRT")
    plt.xlabel("I")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_psnr_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    I_vals = [d["I"] for d in ssim_history]
    ssim_x_vals = [d["ssim_x"] for d in ssim_history]
    ssim_x_sirt_vals = [d["ssim_x_sirt"] for d in ssim_history]

    plt.figure()
    plt.plot(I_vals, ssim_x_vals, marker="o", label="x")
    plt.plot(I_vals, ssim_x_sirt_vals, marker="o", label="x SIRT")
    plt.xlabel("I")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{batch_id}_ssim_curve_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    
    gt_images.append(ground_truth_1)
    gt_digits.append(ground_truth_2)
    gt_centers.append(ground_truth_3)
    both_images.append(images)
    model_digits.append(digits)
    model_centers.append(centers)
    sirt_images.append(images_SIRT)
    clas_images.append(images_class)
    deno_images.append(images_deno)
    sirt_psnr.append([d["psnr_x_sirt"] for d in psnr_history])
    both_psnr.append([d["psnr_x"] for d in psnr_history])
    clas_psnr.append([d["psnr_x_clas"] for d in psnr_history])
    deno_psnr.append([d["psnr_x_deno"] for d in psnr_history])
    sirt_ssim.append([d["ssim_x_sirt"] for d in ssim_history])
    both_ssim.append([d["ssim_x"] for d in ssim_history])
    clas_ssim.append([d["ssim_x_clas"] for d in ssim_history])
    deno_ssim.append([d["ssim_x_deno"] for d in ssim_history])
    sirt_psnr_crop.append([d["psnr_x_sirt_crop"] for d in psnr_history])
    both_psnr_crop.append([d["psnr_x_crop"] for d in psnr_history])
    clas_psnr_crop.append([d["psnr_x_clas_crop"] for d in psnr_history])
    deno_psnr_crop.append([d["psnr_x_deno_crop"] for d in psnr_history])
    sirt_ssim_crop.append([d["ssim_x_sirt_crop"] for d in ssim_history])
    both_ssim_crop.append([d["ssim_x_crop"] for d in ssim_history])
    clas_ssim_crop.append([d["ssim_x_clas_crop"] for d in ssim_history])
    deno_ssim_crop.append([d["ssim_x_deno_crop"] for d in ssim_history])

results = {
    "T_ini": T_ini,
    "T" : T,
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
    "model_digits": model_digits,
    "model_centers": model_centers,
    }


torch.save(results, os.path.join(results_dir, f"results_mnist_ct_{name}.pt"))


for h in list(logger.handlers):
    try:
        h.flush()
        h.close()
    finally:
        logger.removeHandler(h)

logging.shutdown()