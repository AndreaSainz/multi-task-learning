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
from code.utils.markov_process import build_transition_matrix, markov_chain_uniform_noise
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
import torch.nn as nn


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

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

# PARAMETERS
H = 28
W = 28
num_angles = 30 #60
i_0 = 10**5
sigma_data = 0.1
patch = 28
rad = patch // 2
ymin, ymax = rad, H - rad - 1
xmin, xmax = rad, W - rad - 1


# TRAINING PARAMETERS
T_ini = 20
T_max = 20
T_min = 1
decay_iters = 10
I = 10
alpha = 0.1 
beta1 =  0.3
beta2 = 2e-2
lambda_par = 0
accelerator = Accelerator()
device = accelerator.device
n_train = 5000
n_val = 128*2
batch_size = 128
eps = 1e-12
lr = 1e-4
weight_decay = 1e-4
rho = 0.090  #0.090 hace que en 10 pasos haya aprox 40% en la clase principal
C = 0.5


results_dir = "simpler_pipeline/Results/Just_Classification/fine_tunning_JAdler_cleaner_0.1"
classifier_dir = "/home/as3628/multi-task-learning/checkpoints/classifier"


# DATA
# define CT geometry
vg = ts.volume(shape=(1,H,W))                                 # Volume
angles = np.linspace(0, np.pi, num_angles, endpoint=True)     # Angles
pg = ts.parallel(angles=angles, shape=(1, H))                 # Parallel beam structure
A = ts.operator(vg,pg)


A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)

# Import data
data = MnistDataset(training = True, testing= False,  A = A,  dimension = (H,W),  num_angles = num_angles,  sigma = sigma_data, i_0 = i_0, cut = False, seed = seed, max_len = None, device = device)

# for reproducibility
g_split = torch.Generator()
g_split.manual_seed(seed)
labels = np.array([data[i]["label"] for i in range(len(data))])

# Split 
train_idx, val_idx = train_test_split(
    np.arange(len(data)),
    train_size=n_train,
    stratify=labels,
    random_state=seed
)

np.savez(f"{classifier_dir}/splits.npz", train_idx=train_idx, val_idx=val_idx)

train_data = Subset(data, train_idx)
val_data = Subset(data, val_idx[:n_val])


# create dataloaders
train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    generator=g_split)

val_subset = Subset(val_data, range(n_val))

val_dataloader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False
)



# LOGGER
# create logger for checkpoints
log_dir = "logs"
name = "30_angles_sigma_0.3_ft"

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

# MODELS

class_model = CNNClassifier()

checkpoint = torch.load(f"{classifier_dir}/cnn_classifier_bigger_best.pt", map_location="cpu")
class_model.load_state_dict(checkpoint)

class_model.train() 
for p in class_model.parameters():
    p.requires_grad_(True)

loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(class_model.parameters(), lr=lr, weight_decay= weight_decay)

den_model = ResUNetDenoiser()

checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/denoiser/dncnn_best.pt", map_location="cpu")
den_model.load_state_dict(checkpoint)

den_model.eval() 
for p in den_model.parameters():
   p.requires_grad_(False)

class_model, den_model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(class_model, den_model, train_dataloader, val_dataloader, optimizer)


# UTILS
def sirt_step(x, y, D, M, alpha):
    Ax = A_torch(x)
    r = (y - Ax) * M
    AT_r = AT_torch(r) 
    update = AT_r * D
    step_data = alpha * update
    x_new = x + step_data
    return x_new, step_data


def patch_scores_for_digits( img, digit_probs, class_model, den_model, beta_class=1.0, beta_deno=1.0, sigma_data=1.0):
    class_model.eval()

    img_req = img.clone().detach().requires_grad_(True)
    grad_patch = torch.zeros_like(img_req)

    logits = class_model(img_req)
    probs = torch.softmax(logits, dim=1)
    weights = digit_probs.clone().detach()
    # print(weights.size())
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).mean()

    objective_clf = (weights * torch.log(probs.clamp_min(1e-12))).sum() + lambda_par * entropy

    grad_clf = torch.autograd.grad(objective_clf, img_req)[0]

    grad_patch = grad_patch + beta_class * grad_clf

    # # debugging
    # grads = []

    # for k in range(10):
    #     logits = class_model(img_req)  # recomputar
    #     probs = torch.softmax(logits, dim=1)
    #     log_pk = torch.log(probs[:, k].clamp_min(1e-12)).sum()
    #     grad_k = torch.autograd.grad(log_pk, img_req, retain_graph=True)[0]
    #     grads.append(grad_k)

    # grads = torch.stack(grads, dim=1)  # [B,10,1,H,W]
    # grad_B = (weights.view(B, 10, 1, 1, 1) * grads).sum(dim=1)

    # diff = (grad_clf - grad_B).abs().mean()
    # print("Mean abs diff:", diff.item())

    # b = 0

    # for k in range(10):

    #     grad_img = grads[b, k, 0].detach().cpu().numpy()
    #     plt.figure()
    #     plt.imshow(grad_img, cmap="gray")
    #     plt.title(f"Grad digit {k}")
    #     plt.colorbar()
    #     plt.savefig(f"{results_dir}/grad_sample{b}_digit{k}.png")
    #     plt.close()


    grad_prior = (den_model(img_req) - img_req).detach()

    # print("probabilidades", probs.max(dim=1).values)
    # print("grad clf", grad_clf.abs().mean().item())
    # print("grad prior", grad_prior.abs().mean().item())

    grad_patch = grad_patch + beta_deno * grad_prior

    grad_patch = (sigma_data ** 2) * grad_patch
    grad_patch = grad_patch.detach()

    return grad_patch


def expected_task_gradient( x, i , digit_probs, class_model, den_model, beta_class=1.0, beta_deno=1.0, sigma_data=1.0):
    B, _, _, _ = x.shape
    grads = []

    for b in range(B):
        xb = x[b:b+1]                       
        dp = digit_probs[b:b+1]

        grad_patch = patch_scores_for_digits(
            img=xb,
            digit_probs=dp,
            class_model=class_model,
            den_model=den_model,
            beta_class=beta_class,
            beta_deno=beta_deno,
            sigma_data=sigma_data,
        )  # [1,1,28,28]

        grads.append(grad_patch)

    del  dp
    return torch.cat(grads, dim=0)


def update_case(x, y, D, M,  digit, alpha, i, batch_id, BOTH = False):
    x, step_data = sirt_step(x, y, D, M, alpha)

    if BOTH :
        grad_task = expected_task_gradient(
            x=x,
            i=i,
            digit_probs=digit,
            class_model=class_model,
            den_model=den_model,
            beta_class=beta1,
            beta_deno=beta2,
            sigma_data=sigma_data,
        )
        
        x = (x + grad_task).clamp(0, 1).detach()
    return x, step_data


def evaluate(model, dataloader, M_base, D_base):
    model.eval()
    total_correct = 0
    total_samples = 0


    for batch in dataloader:
        x_true = batch["ground_truth_image"].to(device)
        y = batch["label"].to(device)
        sinogram = batch["noisy_sinogram"].to(device)

        B = x_true.shape[0]
        M = M_base.expand(B, -1, -1, -1)
        D = D_base.expand(B, -1, -1, -1)

        x = torch.zeros_like(x_true)

        # SIRT init
        for _ in range(T_ini):
            x, _ = sirt_step(x, sinogram, D, M, alpha)
        x = x.clamp(0,1)

        # principal loop
        x_both = x.clone()

        for i in range(I):
            if i < decay_iters:
                T = int(T_max - (T_max - T_min) * (i / (decay_iters - 1)))
            else:
                T = T_min
                
            for t in range(T): 
                x_both, _ = update_case(x_both, sinogram, D, M, y, alpha, i, 0)

        # classifier
        with torch.no_grad():
            logits = model(x_both)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

    return total_correct / total_samples


#  MULTI-TASK FUNCTION
def model(y, D, M, batch_id, x_true, digit_true, vector_of_labels):
    images_SIRT = []
    images_class = []
    images_deno = []
    images = []
    digits = []
    digits_sirt_all = []
    psnr_history = []
    ssim_history = []   
    digit_acc_history = []
    iter_metrics = []

    digit_true_one_hot = F.one_hot(digit_true, num_classes=10).float()

    # parameters for classifier fine tunning
    total_correct = 0
    total_samples = 0
    class_model.train() 
    for p in class_model.parameters():
        p.requires_grad_(True)

    B = y.shape[0]


    # RECONSTRUCTION
    x = torch.zeros(B, 1, H, W, device=y.device)

    for t in range(T_ini):
        x_prev = x
        x , step_data = sirt_step(x, y, D, M, alpha)
        if accelerator.is_main_process and t == 0 and batch_id == 0:
            fig, ax = plt.subplots()
            im = ax.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")
            fig.colorbar(im, ax=ax)
            fig.savefig(
                os.path.join(results_dir, f"initial_sample_sirt_{name}.png"),
                dpi=300, bbox_inches="tight"
            )
            plt.close(fig)

    x = x.clamp(0,1)
    x_both = x.clone()
    x_sirt = x.clone()

    # metrics after the initialization with SIRT (equi to I = -1, with T_ini steps)
    psnr_x_init = compute_psnr_results(x.detach(), x_true.detach())
    ssim_sirt_init = compute_ssim(x.detach(), x_true.detach(), data_range=1.0)

    ssim_history.append({
        "I": -1,
        "num_inner_steps": T_ini,
        "ssim_x": ssim_sirt_init.detach().cpu(),
        "ssim_x_sirt": ssim_sirt_init.detach().cpu()
    })

    psnr_history.append({
        "I": -1,
        "num_inner_steps": T_ini,
        "psnr_x": psnr_x_init.detach().cpu(),
        "psnr_x_sirt": psnr_x_init.detach().cpu()
    })

    if accelerator.is_main_process and batch_id== 0:
        logger.info(
            f"[PSNR] I=-1 | steps={T_ini} | x={psnr_x_init.mean().item():.4f} dB"
        )
        logger.info(
            f"[SSIM] I=-1 | steps={T_ini} | x={ssim_sirt_init.mean().item():.4f}"
        )

    # CLASSIFICATOR TRAINING
    class_model.train()
    for p in class_model.parameters():
        p.requires_grad_(True)

    optimizer.zero_grad()
    logits = class_model(x)

    # log_probs = F.log_softmax(logits, dim=1)
    # loss_soft = -(vector_of_labels[I] * log_probs).sum(dim=1).mean()
    loss = (1-C) * loss_fn(logits, vector_of_labels[I].to(device)) + C * F.mse_loss(A_torch(x), y)
    # print(f"manual :{loss_soft} ||| automatica: {loss} |||| Diferencia: {loss_soft-loss}")

    loss.backward()
    optimizer.step()


    total_samples += B
    preds = logits.argmax(dim=1)
    correct = (preds == digit_true).sum().item()
    total_correct += correct

    avg_loss = loss.item()
    avg_metric = total_correct / total_samples

    iter_metrics.append({
        "I": -1,
        "loss": avg_loss,
        "acc": avg_metric})


    digit = torch.softmax(logits, dim=1)
    digit_sirt = digit.clone()

    digit_prob_true = digit.gather(1, digit_true.unsqueeze(1)).squeeze(1)
    digit_nll = -torch.log(digit_prob_true.clamp_min(eps))
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

    images.append(x.detach().cpu())
    images_SIRT.append(x.detach().cpu())
    digits.append(digit.detach().cpu())
    digits_sirt_all.append(digit_sirt.detach().cpu())



    ###### model loop
    for i in range(I):
        # parameters for classifier fine tunning
        total_correct = 0
        total_samples = 0

        if i < decay_iters:
            T = int(T_max - (T_max - T_min) * (i / (decay_iters - 1)))
        else:
            T = T_min


        # reconstruction
        for t in range(T):

            # BOTH
            x_both , step_data = update_case(x_both, y, D, M, vector_of_labels[I-i], alpha, i, batch_id, BOTH= True) 
            x_both = x_both.clamp(0,1)


            #only SIRT
            x_sirt , _ = update_case(x_sirt, y, D, M, vector_of_labels[I-i], alpha, i, batch_id) 
            x_sirt = x_sirt.clamp(0,1)
        
        # métricas 
        psnr_x_init = compute_psnr_results(x_both.detach(), x_true.detach())
        psnr_sirt_init = compute_psnr_results(x_sirt.detach(), x_true.detach())
        ssim_x_init = compute_ssim(x_both.detach(), x_true.detach(), data_range=1.0)
        ssim_sirt_init = compute_ssim(x_sirt.detach(), x_true.detach(), data_range=1.0)
        

        ssim_history.append({
            "I": i,
            "num_inner_steps": T,
            "ssim_x": ssim_x_init.detach().cpu(),
            "ssim_x_sirt": ssim_sirt_init.detach().cpu()
           
        })

        psnr_history.append({
            "I": i,
            "num_inner_steps": T,
            "psnr_x": psnr_x_init.detach().cpu(),
            "psnr_x_sirt": psnr_sirt_init.detach().cpu()
        })

        if  accelerator.is_main_process and batch_id== 0:
            logger.info(
                f"[PSNR] I={i} | steps={T} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_sirt_init.mean().item():.4f} dB"
            )
            logger.info(
                f"[SSIM] I={i} | steps={T} | x={ssim_x_init.mean().item():.4f} | x_sirt={ssim_sirt_init.mean().item():.4f}"
            )

        if accelerator.is_main_process and i % 5 == 0: 
            for b in range(min(10,B)):

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

        # digits 

        #BOTH
        class_model.train()
        for p in class_model.parameters():
            p.requires_grad_(True)
        
        optimizer.zero_grad()
        logits_both = class_model(x_both)

        loss = (1-C) * loss_fn(logits_both, vector_of_labels[I-i].to(device)) + C * F.mse_loss(A_torch(x_both), y)
        loss.backward()
        optimizer.step()

        total_samples += B
        preds = logits_both.argmax(dim=1)
        correct = (preds == digit_true).sum().item()
        total_correct += correct
        avg_loss = loss.item()
        avg_metric = total_correct / total_samples

        iter_metrics.append({
            "I": i,
            "loss": avg_loss,
            "acc": avg_metric})
        

        digit = torch.softmax(logits_both, dim=1)

        digit_prob_true = digit.gather(1, digit_true.unsqueeze(1)).squeeze(1)
        digit_nll = -torch.log(digit_prob_true.clamp_min(eps))
        digit_entropy = -(digit * torch.log(digit.clamp_min(eps))).sum(dim=1)

        #SIRT
        with torch.no_grad():
            logits_sirt = class_model(x_sirt)
            digit_sirt = torch.softmax(logits_sirt, dim=1)

        digit_prob_true_sirt = digit_sirt.gather(1, digit_true.unsqueeze(1)).squeeze(1)
        digit_nll_sirt = -torch.log(digit_prob_true_sirt.clamp_min(eps))
        digit_entropy_sirt = -(digit_sirt * torch.log(digit_sirt.clamp_min(eps))).sum(dim=1)

        digit_acc_history.append({
            "I": i,
            "digit_nll": digit_nll.detach().cpu(),
            "digit_prob_true": digit_prob_true.detach().cpu(),
            "digit_entropy": digit_entropy.detach().cpu(),
            "digit_nll_sirt": digit_nll_sirt.detach().cpu(),
            "digit_prob_true_sirt": digit_prob_true_sirt.detach().cpu(),
            "digit_entropy_sirt": digit_entropy_sirt.detach().cpu()
            })

        if accelerator.is_main_process and i % 5 == 0 and batch_id == 0:
            logger.info(f"{i}th digit: {digit}")


        #save information
        images.append(x_both.detach().cpu())
        images_SIRT.append(x_sirt.detach().cpu())
        digits.append(digit.detach().cpu())
        digits_sirt_all.append(digit_sirt.detach().cpu())

        del logits_both, logits_sirt

        

    return images, images_SIRT, digits,digits_sirt_all, psnr_history, ssim_history, digit_acc_history, iter_metrics


        

###################################################################### training part #####################################################################

gt_images = []
gt_digits = []
both_digits = []
sirt_digits = []
both_digit_prob_true = []
sirt_digit_prob_true = []
both_digit_nll = []
sirt_digit_nll = []
both_digit_entropy = []
sirt_digit_entropy = []
both_images = []
sirt_images = []
sirt_psnr = []
both_psnr = []
sirt_ssim = []
both_ssim = []
classificator_loss = []
classificator_acc = []
val_acc_history = []
steps_history = []

# for the classifier
best_val_acc = 0

# normalizimng matrix

ones_img = torch.ones(1, 1, H, W, device=device)
ones_sino = torch.ones(1, 1, num_angles, H, device=device)

row_sums = A_torch(ones_img)
col_sums = AT_torch(ones_sino)

M_base = 1.0 / torch.clamp(row_sums, min=1e-8)
D_base = 1.0 / torch.clamp(col_sums, min=1e-8)

# build transitioning matrix
P = build_transition_matrix(rho = rho, num_classes=10).to(device) 
# print(f" P: {P}")
# print(f" suma P: {P.sum(dim=1)}")

for batch_id, batch in enumerate(train_dataloader):

    ground_truth_1 = batch["ground_truth_image"].to(device)
    ground_truth_2 = batch["label"].to(device)
    input_data = batch["noisy_sinogram"].to(device)

    B = ground_truth_1.shape[0]
    M = M_base.expand(B, -1, -1, -1)
    D = D_base.expand(B, -1, -1, -1)

    
    
    if accelerator.is_main_process and batch_id== 0:
        logger.info(f"Original digit: {ground_truth_2}")

    # adding noise to the labels for each iteration
    labels_one_hot = F.one_hot(ground_truth_2, num_classes=10).float().to(device)
    vector_of_labels = markov_chain_uniform_noise(labels_one_hot, P, I+1)
    # print(f"sahpe: {vector_of_labels.shape}")
    # print(f" primera label: {vector_of_labels[0][0]}")
    # print(f" ultima label: {vector_of_labels[-1][0]}")
    # print(f" suma de cada label: {vector_of_labels.sum(dim=-1)}")

    images, images_SIRT, digits, digits_sirt, psnr_history, ssim_history, digit_acc_history, iter_metrics= model(input_data, D, M, batch_id, ground_truth_1, ground_truth_2, vector_of_labels)

    

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(class_model)
        torch.save({
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict()
                }, os.path.join(classifier_dir, "classifier_last_finetunned_noisy_labels_JAdler_clean.pt"))

        if batch_id % 10 == 0:  
        
            val_acc = evaluate(class_model, val_dataloader, M_base, D_base)
            
            logger.info(f"Validation accuracy: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                torch.save({
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, os.path.join(classifier_dir, "classifier_best_finetunned_noisy_labels_JAdler_clean.pt"))
            
            val_acc_history.append(val_acc)
            steps_history.append(batch_id)

        if batch_id== 0:
            for b in range(20):
                # take last image from the list
                img = ground_truth_1[b,0].detach().cpu().numpy()
                img_both = images[-1][b, 0].detach().cpu().numpy()
                img_sirt = images_SIRT[-1][b, 0].detach().cpu().numpy()

                all_imgs = np.stack([img, img_sirt, img_both])
                vmin = all_imgs.min()
                vmax = all_imgs.max()

                fig, axes = plt.subplots(1, 3, figsize=(20, 4))

                im = axes[0].imshow(img_sirt, cmap="gray", vmin=vmin, vmax=vmax)
                axes[0].set_title("SIRT only")
                axes[0].axis("off")

                axes[1].imshow(img_both, cmap="gray", vmin=vmin, vmax=vmax)
                axes[1].set_title("SIRT + classifier + denoiser")
                axes[1].axis("off")

                axes[2].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                axes[2].set_title("Ground Truth")
                axes[2].axis("off")


                # common color bar 
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                fig.colorbar(im, cax=cbar_ax)

                fig.subplots_adjust(right=0.9)
                plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_comparison_{name}.png"), dpi=300, bbox_inches="tight")
                plt.close(fig)

                I_vals = [d["I"] for d in psnr_history]
                psnr_x_vals = [d["psnr_x"][b].mean().item() for d in psnr_history]
                psnr_x_sirt_vals = [d["psnr_x_sirt"][b].mean().item() for d in psnr_history]

                plt.figure()
                plt.plot(I_vals, psnr_x_sirt_vals, marker="o", label="Mean SIRT")
                plt.plot(I_vals, psnr_x_vals, marker="o", label="Mean SIRT + classifier + denoiser")
                plt.xlabel("I")
                plt.ylabel("PSNR (dB)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_psnr_curve_{name}.png"), dpi=300, bbox_inches="tight")
                plt.close()

                
                ssim_x_vals = [d["ssim_x"][b].mean().item() for d in ssim_history]
                ssim_x_sirt_vals = [d["ssim_x_sirt"][b].mean().item() for d in ssim_history]

                plt.figure()
                plt.plot(I_vals, ssim_x_sirt_vals, marker="o", label="Mean SIRT")
                plt.plot(I_vals, ssim_x_vals, marker="o", label="Mean SIRT + classifier + denoiser")
                plt.xlabel("I")
                plt.ylabel("SSIM")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_ssim_curve_{name}.png"), dpi=300, bbox_inches="tight")
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

                class_loss = [d["loss"] for d in iter_metrics]
                class_acc = [d["acc"] for d in iter_metrics]

                plt.figure()
                plt.plot(I_vals, class_loss, label="Classificator loss")
                plt.xlabel("I")
                plt.ylabel("loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_classificator_loss_{name}.png"), dpi=300, bbox_inches="tight")
                plt.close()

                plt.figure()
                plt.plot(I_vals, class_acc, label="Classificator accuracy")
                plt.xlabel("I")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(results_dir, f"{batch_id}_sample{b}_classificator_acc_{name}.png"), dpi=300, bbox_inches="tight")
                plt.close()


        
    for b in range(B):
        gt_images.append(ground_truth_1[b].detach().cpu())
        gt_digits.append(ground_truth_2[b].detach().cpu())

        both_images.append(images[-1][b].detach().cpu())
        sirt_images.append(images_SIRT[-1][b].detach().cpu())

        both_digits.append(digits[-1][b].detach().cpu())
        sirt_digits.append(digits_sirt[-1][b].detach().cpu())


        both_digit_prob_true.append([d["digit_prob_true"][b].item() for d in digit_acc_history])
        sirt_digit_prob_true.append([d["digit_prob_true_sirt"][b].item() for d in digit_acc_history])

        both_digit_nll.append([d["digit_nll"][b].item() for d in digit_acc_history])
        sirt_digit_nll.append([d["digit_nll_sirt"][b].item() for d in digit_acc_history])

        both_digit_entropy.append([d["digit_entropy"][b].item() for d in digit_acc_history])
        sirt_digit_entropy.append([d["digit_entropy_sirt"][b].item() for d in digit_acc_history])


        both_psnr.append([d["psnr_x"][b].item() for d in psnr_history])
        sirt_psnr.append([d["psnr_x_sirt"][b].item() for d in psnr_history])
        

        both_ssim.append([d["ssim_x"][b].item() for d in ssim_history])
        sirt_ssim.append([d["ssim_x_sirt"][b].item() for d in ssim_history])
        
    classificator_loss.append([d["loss"] for d in iter_metrics])
    classificator_acc.append([d["acc"] for d in iter_metrics])

# Classifier accuracy    
plt.figure()
plt.plot(steps_history, val_acc_history, marker="o")
plt.xlabel("Training step")
plt.ylabel("Validation accuracy")
plt.grid(True)
plt.savefig(os.path.join(results_dir, "val_accuracy_curve.png"))
plt.close()

results = {
    "T_ini": T_ini,
    "T" : [d["num_inner_steps"] for d in psnr_history],
    "I" : I,
    "sigma_data": sigma_data,
    "angles": angles,
    "gt_images": gt_images,
    "gt_digits": gt_digits,
    "both_images": both_images,      
    "sirt_images": sirt_images,
    "sirt_psnr": sirt_psnr,
    "both_psnr": both_psnr,
    "sirt_ssim": sirt_ssim,
    "both_ssim": both_ssim,
    "both_digits": both_digits,
    "sirt_digits": sirt_digits,
    "both_digit_prob_true": both_digit_prob_true,
    "sirt_digit_prob_true": sirt_digit_prob_true,
    "both_digit_nll": both_digit_nll,
    "sirt_digit_nll": sirt_digit_nll,
    "both_digit_entropy": both_digit_entropy,
    "sirt_digit_entropy": sirt_digit_entropy
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