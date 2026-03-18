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
num_angles = 60
i_0 = 10**5
sigma_data = 0.5


### training parameters
T_ini = 40
T = 40
I = 30
alpha = 2e-1/(75.77664947509766)**2 #1/(liptchitz constant)^2
# beta1 = [3, 3]
# beta2 = [0,0] #[5e-4, 5e-3]
device = "cuda"
n_train = 10
batch_size = 1
save_batch_id = 0
k = int(0.05 * H * W)   # top 5%
results_dir = "simpler_pipeline/results/Experiments3"


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
name = "recon_much_noise_deno_class"
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

###################################################################### Multi-task function #####################################################################

def model(y, batch_id, centre, digit): #x
    images_SIRT = []
    images = []
    centers = []
    digits = []

    B = y.shape[0]


    ##### Initilizing tasks


    # RECONSTRUCTION
    x = torch.zeros(B, 1, H, W, device=y.device)
    

    #if batch_id == 0:
    dx_norm_mean = torch.zeros(T_ini, device=y.device) ################
    dx_norm_b0 = torch.zeros(T_ini, device=y.device)  # una muestra

    for t in range(T_ini):
        x_prev = x
        Ax = A_torch(x)
        r = y - Ax
        AT_r = AT_torch(r)
        step_data = alpha * AT_r
        if batch_id ==0:
            logger.info(f"Gradient descent step -1_{t}: {step_data.flatten(1).norm(dim=1).item()}")
        x = x + step_data

        # debugging || x^{k+1} -x^k||
        #if batch_id == 0:
        dx = x - x_prev ################
        dx_norm_b0[t]   = dx[0].flatten().norm()
        dx_norm_mean[t] = dx.flatten(1).norm(dim=1).mean()
            
    #if batch_id == 0:
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
    x_sirt = x.clone()

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

        #if batch_id == 0:
        dx_norm_mean = torch.zeros(T, device=y.device) ################
        dx_norm_b0 = torch.zeros(T, device=y.device)  # una muestra
        dx_norm_mean_sirt = torch.zeros(T, device=y.device)
        dx_norm_b0_sirt = torch.zeros(T, device=y.device)  # una muestra

        # reconstruction
        for t in range(T):
            x_prev = x
            Ax = A_torch(x)
            r = y - Ax
            AT_r = AT_torch(r)
            step_data = alpha * AT_r
            if batch_id ==0:
                logger.info(f"Gradient descent step {i}_{t}: {step_data.flatten(1).norm(dim=1).item()}")
            x = x + step_data

            # regularization term F
            x = x.clamp(0,1)
            x = x.detach()

            # we need to crop x as an MNIST digit 28x28
            crops = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)


            for b in range(B):
                cy = int(centre[b, 0].item())
                cx = int(centre[b, 1].item())
                crops[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

            # normalizing Flow was trained for data in range [0,1]
            crops = torch.clamp(crops, 0.0, 1.0)
            crop_x = crops.detach().requires_grad_(True)
            

            # calculate the value of mu() in this cropeed image
            logits = class_model(crop_x)   # [B,10]
            log_probs = torch.log_softmax(logits, dim=1)         # [B,10]
            classi = log_probs.gather(1, digit.unsqueeze(1)).squeeze(1)     # [B,1]
            grad_classi = torch.autograd.grad(classi.sum(),crop_x,create_graph=False)[0]
            deno = den_model(crop_x)
            grad_deno = (deno-crop_x)

            # getting the F value and its gradient
            #F =  (sigma_data**2) * log_mu.sum()
            if i == 0:
                g = grad_classi.flatten(1).norm(dim=1)  # (B,)
                beta1 = 0.01 / (g + 1e-8)
                g = grad_deno.flatten(1).norm(dim=1)  # (B,)
                beta2 = 0.01 / (g + 1e-8)
                grad_crop = -(sigma_data**2) * (beta1*grad_classi + beta2*grad_deno)
            else:
                g = grad_classi.flatten(1).norm(dim=1)  # (B,)
                beta1 = 0.08 / (g + 1e-8)
                g = grad_deno.flatten(1).norm(dim=1)  # (B,)
                beta2 = 0.08 / (g + 1e-8)
                grad_crop = -(sigma_data**2) * (beta1*grad_classi + beta2*grad_deno)
            #grad_crop = torch.autograd.grad(F, crop_x, create_graph=False)[0] 

            if  i % 2 == 0 and t==0: #batch_id == 0 and
                plt.imshow(grad_crop[0,0].detach().cpu().numpy(), cmap="gray")
                plt.colorbar()
                plt.savefig(os.path.join(results_dir, f"{batch_id}_{i}_grad_crop__{name}.png"), dpi=300, bbox_inches="tight")
                plt.close()

            x_scaled = x.clone()
            scale = grad_crop.norm() / (x_scaled.norm() + 1e-8)
            x_scaled = x_scaled * scale
            grad_F = 10 * x_scaled 

            


            for b in range(B):
                cy = int(centre[b, 0].item())
                cx = int(centre[b, 1].item())
                grad_F[b:b+1, :, cy-14:cy+14, cx-14:cx+14] = grad_crop[b:b+1]

            # grad descent on total => x <- x + beta * grad_F
            step_F =  grad_F
            # if  #batch_id ==0 and
            logger.info(f"Gradient descent step {batch_id}_{i}_{t}| classifier : {beta1*grad_classi.flatten(1).norm(dim=1).item()}, denoiser : {beta2*grad_deno.flatten(1).norm(dim=1).item()}")
            # else: #if batch_id ==0:
            #     logger.info(f"Gradient descent step {batch_id}_{i}_{t}| classifier : {beta1[1]*grad_classi.flatten(1).norm(dim=1).item()}, denoiser : {beta2[1]*grad_deno.flatten(1).norm(dim=1).item()}")
            x = x - step_F

            # debugging || x^{k+1} -x^k||
            #if batch_id == 0:
            dx = x - x_prev ################
            dx_norm_b0[t] = dx[0].flatten().norm()          # norma para el primer elemento del batch
            dx_norm_mean[t] = dx.flatten(1).norm(dim=1).mean()                  # media batch


            #only SIRT
            x_sirt_prev = x_sirt
            Ax_sirt = A_torch(x_sirt)
            r_sirt = y - Ax_sirt
            AT_r_sirt = AT_torch(r_sirt)
            x_sirt = x_sirt + alpha * AT_r_sirt
            x_sirt = x_sirt.clamp(0,1)

            # debugging || x^{k+1} -x^k||
            #if batch_id == 0:
            dx = x_sirt - x_sirt_prev ################
            dx_norm_b0_sirt[t] = dx[0].flatten().norm()          # norma por elemento del batch
            dx_norm_mean_sirt[t] = dx.flatten(1).norm(dim=1).mean()             # media batch

        #if batch_id == 0:
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

            plt.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")
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
        images.append(x.detach().cpu())
        images_SIRT.append(x_sirt.detach().cpu())
        centers.append(centre.detach().cpu())
        digits.append(digit.detach().cpu())

    return images, images_SIRT, centers, digits



###################################################################### training part #####################################################################

gt_images = []
gt_digits = []
gt_centers = []
model_images = []
model_digits = []
model_centers = []
sirt_images = []

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
    images, images_SIRT, centers, digits = model(input_data, batch_id, ground_truth_3, ground_truth_2) #x

    if batch_id == save_batch_id:
        gt_images.append(ground_truth_1)
        gt_digits.append(ground_truth_2)
        gt_centers.append(ground_truth_3)
        model_images.append(images)
        model_digits.append(digits)
        model_centers.append(centers)
        sirt_images.append(images_SIRT)

results = {
    "T" : T,
    "I" : I,
    "gt_images": gt_images,
    "gt_digits": gt_digits,
    "gt_centers": gt_centers,
    "model_images": model_images,      # [batch][iter][B,1,H,W]
    "sirt_images": sirt_images,
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