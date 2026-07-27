import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_loader import MnistDataset
from code.models.score_unet import ContextUnet, ddpm_schedules
from code.utils.metrics import compute_psnr_results, compute_ssim, compute_mse, update_confusion_matrix, metrics_from_confusion_matrix
from code.utils.gradients import update_case, regularization_grad, score_centers_difussion_model
from code.utils.image_cropping import extract_crops, insert_crop_grad_in_image
from code.ct_recon.SIRT import sirt_step
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
import torch, gc




###################################################################### Parameters #####################################################################

### data
accelerator = Accelerator()
device = accelerator.device
seed = 29072000
H = 100
W = 100
num_angles = 60
i_0 = 10**5
sigma_data = 0.3


### training parameters
R_ini = 80
R = 70
I = 10
tol_dx = 1e-3
patience_dx = 3
alpha = 2e-1/(75.77664947509766)**2 #1/(liptchitz constant)^2 2e-1/(53.595951080322266)**2 #
gamma = [1e-3, 1e-3]
n_train = 10
batch_size = 1
t = torch.full((batch_size,), 5, device=device) #time step in the denoiser
save_batch_id = 0
topk_digits = 5184
results_dir = "simpler_pipeline/results/ExperimentsScore/Experiments4/60_angles_sigma_0.3"


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
name = "conditional_score_60_angles_sigma_0.3"
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


###################################################################### Trained Score Models #####################################################################

score_model = ContextUnet(in_channels=1, n_feat=128, n_classes=10)

checkpoint = torch.load("/home/as3628/multi-task-learning/simpler_pipeline/model_39.pth", map_location="cpu")
alpha_t = checkpoint["alpha_t"]
beta_t = 1 - alpha_t
beta1 = beta_t[0].item()
beta2 = beta_t[-1].item()
T_max =  len(alpha_t) - 1
model_state = {
    k.replace("nn_model.", ""): v
    for k, v in checkpoint.items()
    if k.startswith("nn_model.")
}
score_model.load_state_dict(model_state)
sqrtab = checkpoint["sqrtab"].to(device)
sqrtmab = checkpoint["sqrtmab"].to(device)
score_model.eval() 
for p in score_model.parameters():
   p.requires_grad_(False)

# for using the score
t_input = t.float().view(batch_size, 1) / T_max
fixed_noise = torch.randn((batch_size, 1, 28, 28), device=device)
context_mask = torch.zeros(batch_size, device=device) #we want to use the condition

score_model, train_dataloader = accelerator.prepare(score_model, train_dataloader)


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
    

    B = y.shape[0]


    ##### Initilizing tasks


    # RECONSTRUCTION
    x = torch.zeros(B, 1, H, W, device=y.device)
    

    #if batch_id == 0:
    dx_norm_mean = torch.zeros(R_ini, device=y.device) 
    dx_norm_b0 = torch.zeros(R_ini, device=y.device)  # una muestra

    for r in range(R_ini):
        x_prev = x
        x , step_data = sirt_step(A_torch, AT_torch, x, y, alpha)
        if accelerator.is_main_process and batch_id ==0:
            logger.info(f"Gradient descent step -1_{r}: {step_data.flatten(1).norm(dim=1).mean().item()}")

        # debugging || x^{k+1} -x^k||
        if accelerator.is_main_process and batch_id == 0:
            dx = x - x_prev 
            dx_norm_b0[r]   = dx[0].flatten().norm()
            dx_norm_mean[r] = dx.flatten(1).norm(dim=1).mean()

    # metrics after the initialization with SIRT (equi to I = -1, with T_ini steps)
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
        "num_inner_steps": R_ini,
        "ssim_x": ssim_sirt_init.detach().cpu(),
        "ssim_x_sirt": ssim_sirt_init.detach().cpu(),
        "ssim_x_crop": ssim_sirt_init_crop.detach().cpu(),
        "ssim_x_sirt_crop": ssim_sirt_init_crop.detach().cpu(),
    })

    if accelerator.is_main_process:
        logger.info(
            f"[PSNR] I=-1 | steps={R_ini} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_x_init.mean().item():.4f} dB"
        )
        logger.info(
            f"[SSIM] I=-1 | steps={R_ini} | x={ssim_sirt_init.mean().item():.4f} | x_sirt={ssim_sirt_init.mean().item():.4f}"
        )

    psnr_history.append({
        "I": -1,
        "num_inner_steps": R_ini,
        "psnr_x": psnr_x_init.detach().cpu(),
        "psnr_x_sirt": psnr_x_init.detach().cpu(),
        "psnr_x_crop": psnr_x_init_crop.detach().cpu(),
        "psnr_x_sirt_crop": psnr_x_init_crop.detach().cpu(),
    })

    


    if accelerator.is_main_process and batch_id == 0:
        plt.figure() 
        plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch")
        plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
        plt.yscale("log")
        plt.xlabel("r")
        plt.ylabel("$||x_{r+1}-x_r||$")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_initial_dx_norm_batch0_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()


    

    # x for only SIRT
    x = x.clamp(0,1)
    x_score = x.clone()
    x_sirt = x.clone()

    if accelerator.is_main_process and batch_id == 0:
        plt.imshow(x_score[0,0].detach().cpu().numpy(), cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_first_sirt_image_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # DIGIT
    digit = torch.full((B, 10), 0.1, device=y.device)
    digit_sirt = digit.clone()

    centre = torch.zeros((B, H, W), device=x.device)
    area = (ymax - ymin + 1) * (xmax - xmin + 1)
    centre[:, ymin:ymax+1, xmin:xmax+1] = 1.0 / area
    centre_sirt = centre.clone()
    eps = 1e-12

    
    # CENTRE
    with torch.no_grad():
        # map_x[b, y, x] will store a "probability" for placing the 28x28 crop center at (y, x).
        # We only fill valid centers (14..85) so the crop stays inside the 100x100 image.
        map_x = score_centers_difussion_model(
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
                crops_topk_sirt[j:j+1] = x_sirt[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

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



    # digit distribution distance to true delta
    digit_prob_true = digit.gather(1, digit_true.unsqueeze(1)).squeeze(1)   # [B]
    digit_nll = -torch.log(digit_prob_true.clamp_min(eps)))
    digit_entropy = -(digit * torch.log(digit.clamp_min(eps))).sum(dim=1)

    

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
        "centre_w1_sirt": centre_w1.detach().cpu(),
    })


    images.append(x.detach().cpu())
    images_SIRT.append(x.detach().cpu())
    centers.append(centre.detach().cpu())
    digits.append(digit.detach().cpu())
    centers_sirt_all.append(centre_sirt.detach().cpu())
    digits_sirt_all.append(digit_sirt.detach().cpu())

    ###### model loop

    for i in range(I):

        if accelerator.is_main_process and batch_id == 0:
            dx_norm_mean = torch.zeros(R, device=y.device) 
            dx_norm_b0 = torch.zeros(R, device=y.device)  # una muestra
            dx_norm_mean_sirt = torch.zeros(R, device=y.device)
            dx_norm_b0_sirt = torch.zeros(R, device=y.device)  # una muestra

        # reconstruction
        for r in range(R):
            x_prev_both = x_score 
            x_score , step_data = update_case(x_score, y, centre, digit, alpha, i, r, batch_id, "both")
            if accelerator.is_main_process and batch_id ==0:
                logger.info(f"Gradient descent step {i}_{r}: {step_data.flatten(1).norm(dim=1).mean().item()}")

        
            # debugging || x^{k+1} -x^k||
            if accelerator.is_main_process and batch_id == 0:
                dx = x_score - x_prev_both
                dx_norm_b0[r] = dx[0].flatten().norm()          # norma para el primer elemento del batch
                dx_norm_mean[r] = dx.flatten(1).norm(dim=1).mean()                  # media batch


            #only SIRT
            x_sirt_prev = x_sirt
            x_sirt , _ = update_case(x_sirt, y, centre, digit, alpha, i, r, batch_id, "none")

            # debugging || x^{k+1} -x^k||
            if accelerator.is_main_process and batch_id == 0:
                dx = x_sirt - x_sirt_prev 
                dx_norm_b0_sirt[r] = dx[0].flatten().norm()          # norma por elemento del batch
                dx_norm_mean_sirt[r] = dx.flatten(1).norm(dim=1).mean()             # media batch

                
        # metrics after the minimization
        crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        crop_x_sirt = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        crop_x_score = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        for b in range(B):
            cy = int(centre[b, 0].item())
            cx = int(centre[b, 1].item())
            crop_x[b:b+1] = x_true[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
            crop_x_sirt[b:b+1] = x_sirt[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
            crop_x_score[b:b+1] = x_score[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

        psnr_x = compute_psnr_results(x_score.detach(), x_true.detach())
        psnr_sirt = compute_psnr_results(x_sirt.detach(), x_true.detach())
        ssim_x = compute_ssim(x_score.detach(), x_true.detach(), data_range=1.0)
        ssim_sirt = compute_ssim(x_sirt.detach(), x_true.detach(), data_range=1.0)
        psnr_score_crop = compute_psnr_results(crop_x_score.detach(), crop_x.detach())
        psnr_sirt_crop = compute_psnr_results(crop_x_sirt.detach(), crop_x.detach())
        ssim_score_crop = compute_ssim(crop_x_score.detach(), crop_x.detach(), data_range=1.0)
        ssim_sirt_crop = compute_ssim(crop_x_sirt.detach(), crop_x.detach(), data_range=1.0)

        ssim_history.append({
            "I": i,
            "num_inner_steps": R,
            "ssim_x": ssim_x,
            "ssim_x_sirt": ssim_sirt,
            "ssim_x_crop": ssim_score_crop,
            "ssim_x_sirt_crop": ssim_sirt_crop,
        })

        if accelerator.is_main_process:
            logger.info(
            f"[SSIM] I={i} | steps={R} | x={ssim_x.mean().item():.4f}  | x_sirt={ssim_sirt.mean().item():.4f} "
            )
            logger.info(
            f"[PSNR] I={i} | steps={R} | x={psnr_x.mean().item():.4f} dB | x_sirt={psnr_sirt.mean().item():.4f} dB"
            )

        psnr_history.append({
            "I": i,
            "num_inner_steps": R,
            "psnr_x": psnr_x,
            "psnr_x_sirt": psnr_sirt,
            "psnr_x_crop": psnr_score_crop,
            "psnr_x_sirt_crop": psnr_sirt_crop,
        })

        

        if batch_id == 0:
            plt.figure() 
            plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch") 
            plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
            plt.plot(dx_norm_mean_sirt.detach().cpu().numpy(), label="mean over batch SIRT")
            plt.plot(dx_norm_b0_sirt.detach().cpu().numpy(), label="sample 0 SIRT")
            plt.yscale("log")
            plt.xlabel("r")
            plt.ylabel("$||x_{r+1}-x_r||$")
            plt.legend()
            plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_dx_norm_batch0_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

        if  batch_id == 0 and i % 4 == 0:
            plt.imshow(x_sirt[0,0].detach().cpu().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_sirt__{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

            plt.imshow(x_score[0,0].detach().cpu().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_flow_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close()

        

        #center 
        with torch.no_grad():
            # map_x[b, y, x] will store a "probability" for placing the 28x28 crop center at (y, x).
            # We only fill valid centers (14..85) so the crop stays inside the 100x100 image.
            map_x = score_centers_difussion_model(
                            model =class_model,
                            img = x_both,
                            digit_cond=digit,      # [B,10], one-hot o soft
                            denoiser=den_model,
                            chunk_size=512  
                        )
            map_x_sirt = score_centers_difussion_model(
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
        images.append(x_score.detach().cpu())
        images_SIRT.append(x_sirt.detach().cpu())
        centers.append(centre.detach().cpu())
        digits.append(digit.detach().cpu())
        centers_sirt_all.append(centre_sirt.detach().cpu())
        digits_sirt_all.append(digit_sirt.detach().cpu())

        if i % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return images, images_SIRT, centers, digits, centers_sirt_all, digits_sirt_all, psnr_history, ssim_history



###################################################################### training part #####################################################################

gt_images = []
gt_digits = []
gt_centers = []
model_images = []
model_centers = []
sirt_centers = []
model_digit_prob_true = []
sirt_digit_prob_true = []
model_digit_nll = []
sirt_digit_nll = []
model_digit_entropy = []
sirt_digit_entropy = []
model_centre_w1 = []
sirt_centre_w1 = []
sirt_images = []
sirt_psnr = []
sirt_ssim = []
model_psnr = []
model_ssim = []
sirt_psnr_crop = []
sirt_ssim_crop = []
model_psnr_crop = []
model_ssim_crop = []

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
 
    #     # columnas centrales
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
    
 
    if accelerator.is_main_process and batch_id == 0 :
        logger.info(f"Original digit: {ground_truth_2}")
        logger.info(f"original centre: {ground_truth_3}")

    #if 1 in ground_truth_2 :
    images, images_SIRT, centers, digits, centers_sirt, digits_sirt, psnr_history, ssim_history, digit_acc_history, centre_mse_history = model(input_data, batch_id, ground_truth_1, ground_truth_3, ground_truth_2)
    
    if accelerator.is_main_process and batch_id == 0 :

        # take last image from the list
        img = ground_truth_1[0,0].detach().cpu().numpy()
        img_both = images[-1][0, 0].numpy()
        img_sirt = images_SIRT[-1][0, 0].numpy()

        all_imgs = np.stack([img, img_sirt, img_both])
        vmin = all_imgs.min()
        vmax = all_imgs.max()

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        im = axes[0].imshow(img_sirt, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title("SIRT only")
        axes[0].axis("off")

        axes[1].imshow(img_both, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title("SIRT + score")
        axes[1].axis("off")

        axes[2].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        # barra de color común
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(results_dir, f"{batch_id}_comparison_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()


        I_vals = [d["I"] for d in psnr_history]
        psnr_x_vals = [d["psnr_x"].mean().item() for d in psnr_history]
        psnr_x_sirt_vals = [d["psnr_x_sirt"].mean().item() for d in psnr_history]

        plt.figure()
        plt.plot(I_vals, psnr_x_vals, marker="o", label="Mean SCORE")
        plt.plot(I_vals, psnr_x_sirt_vals, marker="o", label="Mean SIRT")
        plt.xlabel("I")
        plt.ylabel("PSNR (dB)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"{batch_id}_psnr_curve_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        I_vals = [d["I"] for d in psnr_history]
        psnr_x_crop_vals = [d["psnr_x_crop"].mean().item() for d in psnr_history]
        psnr_x_sirt_crop_vals = [d["psnr_x_sirt_crop"].mean().item() for d in psnr_history]


        plt.figure()
        plt.plot(I_vals, psnr_x_sirt_crop_vals, marker="o", label="Mean SIRT")
        plt.plot(I_vals, psnr_x_crop_vals, marker="o", label="Mean SCORE")
        plt.xlabel("I")
        plt.ylabel("PSNR crop (dB)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"{batch_id}_psnr_crop_curve_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        I_vals = [d["I"] for d in ssim_history]
        ssim_x_vals = [d["ssim_x"].mean().item() for d in ssim_history]
        ssim_x_sirt_vals = [d["ssim_x_sirt"].mean().item() for d in ssim_history]

        plt.figure()
        plt.plot(I_vals, ssim_x_vals, marker="o", label="Mean SCORE")
        plt.plot(I_vals, ssim_x_sirt_vals, marker="o", label="Mean SIRT")
        plt.xlabel("I")
        plt.ylabel("SSIM")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"{batch_id}_ssim_curve_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        I_vals = [d["I"] for d in ssim_history]
        ssim_x_crop_vals = [d["ssim_x_crop"].mean().item() for d in ssim_history]
        ssim_x_sirt_crop_vals = [d["ssim_x_sirt_crop"].mean().item() for d in ssim_history]


        plt.figure()
        plt.plot(I_vals, ssim_x_sirt_crop_vals, marker="o", label="Mean SIRT")
        plt.plot(I_vals, ssim_x_crop_vals, marker="o", label="Mean SCORE")
        plt.xlabel("I")
        plt.ylabel("SSIM crop ")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"{batch_id}_ssim_crop_curve_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    

        I_vals = [d["I"] for d in digit_acc_history]
        digit_prob_vals = [d["digit_prob_true"].mean().item() for d in digit_acc_history]
        digit_prob_vals_sirt = [d["digit_prob_true_sirt"].mean().item() for d in digit_acc_history]

        plt.figure()
        plt.plot(I_vals, digit_prob_vals, marker="o", label="Mean Digit prob (SCORE)")
        plt.plot(I_vals, digit_prob_vals_sirt, marker="o", label="Mean Digit prob (SIRT)")
        plt.xlabel("I")
        plt.ylabel("Probability")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"{batch_id}_digit_prob_curve_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        digit_nll_vals = [d["digit_nll"].mean().item() for d in digit_acc_history]
        digit_nll_vals_sirt = [d["digit_nll_sirt"].mean().item() for d in digit_acc_history]

        plt.figure()
        plt.plot(I_vals, digit_nll_vals, marker="o", label="Mean Digit NLL (SCORE)")
        plt.plot(I_vals, digit_nll_vals_sirt, marker="o", label="Mean Digit NLL (SIRT)")
        plt.xlabel("I")
        plt.ylabel("-log p(true)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"{batch_id}_digit_nll_curve_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        digit_margin_vals = [d["digit_entropy"].mean().item() for d in digit_acc_history]
        digit_margin_vals_sirt = [d["digit_entropy_sirt"].mean().item() for d in digit_acc_history]

        plt.figure()
        plt.plot(I_vals, digit_margin_vals, marker="o", label="Mean Digit entropy (SCORE)")
        plt.plot(I_vals, digit_margin_vals_sirt, marker="o", label="Mean Digit entropy (SIRT)")
        plt.xlabel("I")
        plt.ylabel("Entropy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"{batch_id}_digit_entropy_curve_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()


        I_vals = [d["I"] for d in centre_mse_history]
        centre_w1_vals = [d["centre_w1"].mean().item() for d in centre_mse_history]
        centre_w1_vals_sirt = [d["centre_w1_sirt"].mean().item() for d in centre_mse_history]

        plt.figure()
        plt.plot(I_vals, centre_w1_vals, marker="o", label="Mean Centre W1 (SCORE)")
        plt.plot(I_vals, centre_w1_vals_sirt, marker="o", label="Mean Centre W1 (SIRT)")
        plt.xlabel("I")
        plt.ylabel("Wasserstein-1 (pixels)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"{batch_id}_centre_w1_curve_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    for b in range(B):
        gt_images.append(ground_truth_1.detach().cpu())
        gt_digits.append(ground_truth_2.detach().cpu())
        gt_centers.append(ground_truth_3.detach().cpu())
        model_images.append(images)
        model_digits.append(digits[-1][b].detach().cpu())
        sirt_digits.append(digits_sirt[-1][b].detach().cpu())
        model_centers.append(centers[-1][b].detach().cpu())
        sirt_centers.append(centers_sirt[-1][b].detach().cpu())
        model_digit_prob_true.append([d["digit_prob_true"][b].item() for d in digit_acc_history])
        sirt_digit_prob_true.append([d["digit_prob_true_sirt"][b].item() for d in digit_acc_history])
        model_digit_nll.append([d["digit_nll"][b].item() for d in digit_acc_history])
        sirt_digit_nll.append([d["digit_nll_sirt"][b].item() for d in digit_acc_history])
        model_digit_entropy.append([d["digit_entropy"][b].item() for d in digit_acc_history])
        sirt_digit_entropy.append([d["digit_entropy_sirt"][b].item() for d in digit_acc_history])
        model_centre_w1.append([d["centre_w1"][b].item() for d in centre_mse_history])
        sirt_centre_w1.append([d["centre_w1_sirt"][b].item() for d in centre_mse_history])
        sirt_images.append(images_SIRT)
        sirt_psnr.append([d["psnr_x_sirt"] for d in psnr_history])
        model_psnr.append([d["psnr_x"] for d in psnr_history])
        sirt_ssim.append([d["ssim_x_sirt"] for d in ssim_history])
        model_ssim.append([d["ssim_x"] for d in ssim_history])
        sirt_psnr_crop.append([d["psnr_x_sirt_crop"] for d in psnr_history])
        model_psnr_crop.append([d["psnr_x_crop"] for d in psnr_history])
        sirt_ssim_crop.append([d["ssim_x_sirt_crop"] for d in ssim_history])
        model_ssim_crop.append([d["ssim_x_crop"] for d in ssim_history])


results = {
    "R_ini": R_ini,
    "R" : R,
    "I" : I,
    "sigma_data": sigma_data,
    "angles": angles,
    "gt_images": gt_images,
    "gt_digits": gt_digits,
    "gt_centers": gt_centers,
    "model_images": model_images,      # [batch][iter][B,1,H,W]
    "sirt_images": sirt_images,
    "sirt_psnr": sirt_psnr,
    "model_psnr": model_psnr,
    "sirt_ssim": sirt_ssim,
    "model_ssim": model_ssim,
    "sirt_psnr_crop": sirt_psnr_crop,
    "model_psnr_crop": model_psnr_crop,
    "sirt_ssim_crop": sirt_ssim_crop,
    "model_ssim_crop": model_ssim_crop,
    "model_digits": model_digits,
    "model_centers": model_centers,
    "sirt_digits": sirt_digits,
    "sirt_centers": sirt_centers,
    "model_digit_prob_true": model_digit_prob_true,
    "sirt_digit_prob_true": sirt_digit_prob_true,
    "model_digit_nll": model_digit_nll,
    "sirt_digit_nll": sirt_digit_nll,
    "model_digit_entropy": model_digit_entropy,
    "sirt_digit_entropy": sirt_digit_entropy,
    "model_centre_w1": model_centre_w1,
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