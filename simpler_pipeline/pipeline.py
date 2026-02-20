import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_loader import MnistDataset
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
T = 20
I = 5
alpha = 1e-4/(53.595951080322266)**2 #1/(liptchitz constant)^2
beta = 5e-8
device = "cuda"
n_train = 20
batch_size = 10
save_batch_id = 0
k = int(0.05 * H * W)   # top 5%


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
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "sigma_no constant_try.log")),
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


###################################################################### Trained NF Model #####################################################################

# Import trained mu(x_cropped|digit)
# Define flows
L = 2               #number of latent variables (z_l)
K = 24              #how complex the invertible function is
torch.manual_seed(29072000)

input_shape = (1, 28, 28)
n_dims = np.prod(input_shape)
channels = 1
hidden_channels = 256           #how large are CNN inside GlowBlocks, so s() and t()
split_mode = 'channel'
scale = True
num_classes = 10

# Set up flows, distributions and merge operations
q0 = []
merges = []
flows = []
for i in range(L):
    flows_ = []
    for j in range(K):
        flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                     split_mode=split_mode, scale=scale)]
    flows_ += [nf.flows.Squeeze()]
    flows += [flows_]
    if i > 0:
        merges += [nf.flows.Merge()]
        latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                        input_shape[2] // 2 ** (L - i))
    else:
        latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                        input_shape[2] // 2 ** L)
    q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

mu_model = nf.MultiscaleFlow(q0, flows, merges)
mu_model = mu_model.to(device)

checkpoint = torch.load("flow_model_1.pt", map_location=device)
mu_model.load_state_dict(checkpoint["model_state_dict"])

mu_model.eval() 
for p in mu_model.parameters():
   p.requires_grad_(False)





###################################################################### Center finder function #####################################################################
def score_centers_with_unfold(model, img, digit_cond, *, H=100, W=100, patch=28,
                              ymin=14, ymax=85, xmin=14, xmax=85,
                              chunk_size=512):
    """
    Devuelve map_x con scores (log-prob o -NLL) para cada centro (y,x) válido.
    - x: [B,1,H,W]
    - digit_cond: lo que espere tu mu_model 
    - map_x: [B,H,W] con -inf fuera de la zona válida
    """

    device = img.device
    B, C, Hx, Wx = img.shape
    assert Hx == H and Wx == W and C == 1
    assert patch % 2 == 0
    rad = patch // 2

    # 1) Extrae TODOS los patches 28x28 como vistas (sin copiar):
    # patches: [B, C, out_h, out_w, patch, patch]
    patches = img.unfold(2, patch, 1).unfold(3, patch, 1)

    out_h = H - patch + 1   # 73 si H=100, patch=28
    out_w = W - patch + 1   # 73

    # Centro (cy,cx) corresponde a esquina sup izq (ty,tx) = (cy-rad, cx-rad)
    ty_min, ty_max = ymin - rad, ymax - rad  # 0..71 si ymin=14,ymax=85,rad=14
    tx_min, tx_max = xmin - rad, xmax - rad  # 0..71

    # Recorta solo la región válida
    patches = patches[:, :, ty_min:ty_max+1, tx_min:tx_max+1, :, :]
    # Ahora patches: [B,1,72,72,28,28]  (si rango 14..85)

    gh, gw = patches.shape[2], patches.shape[3]  # 72,72
    N = gh * gw  # 5184

    # 2) Flatten a batch grande: [B*N,1,28,28]
    patches_flat = patches.contiguous().view(B * N, 1, patch, patch)

    # 3) Evalúa mu_model en chunks
    scores_flat = torch.empty(B * N, device=device, dtype=torch.float32)

    for start in range(0, B * N, chunk_size):
        end = min(start + chunk_size, B * N)
        p = patches_flat[start:end]

        # Repite el condicionamiento de dígito para cada patch
        cond = digit_cond
        cond_rep = cond.repeat_interleave(N, dim=0)[start:end]
        
        # Score de cada patch
        scores_flat[start:end] = (- model(p, cond_rep).view(-1)).float()

    # 4) Reshape a [B,72,72] y colócalo en map_x [B,H,W]
    scores_grid = scores_flat.view(B, gh, gw)

    map_x = torch.full((B, H, W), -torch.inf, device=device, dtype=scores_grid.dtype)

    # Pon scores en las posiciones de centros válidas: cy=ymin..ymax, cx=xmin..xmax
    map_x[:, ymin:ymax+1, xmin:xmax+1] = scores_grid

    return map_x

###################################################################### Multi-task function #####################################################################

def model(y, batch_id):
    images_SIRT = []
    images = []
    centers = []
    digits = []

    B = y.shape[0]


    ##### Initilizing tasks


    # RECONSTRUCTION
    x = torch.zeros(B, 1, H, W, device=y.device)
    

    if batch_id == 0:
        dx_norm_mean = torch.zeros(T, device=y.device)
        dx_norm_b0 = torch.zeros(T, device=y.device)  # una muestra

    for t in range(T):
        x_prev = x
        Ax = A_torch(x)
        r = y - Ax
        AT_r = AT_torch(r)
        step_data = alpha * AT_r
        if batch_id ==0:
            logger.info(f"Gradient descent step -1_{t}: {step_data.flatten(1).norm(dim=1)}")
        x = x + step_data

        # debugging || x^{k+1} -x^k||
        if batch_id == 0:
            dx = x - x_prev
            dx_norm_b0[t]   = dx[0].flatten().norm()
            dx_norm_mean[t] = dx.flatten(1).norm(dim=1).mean()
            
    if batch_id == 0:
        plt.figure()
        plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch")
        plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("$||x_{t+1}-x_t||$")
        plt.legend()
        plt.savefig("simpler_pipeline/results/initial_dx_norm_batch0.png", dpi=300, bbox_inches="tight")
        plt.close()


    if batch_id == 0:
        plt.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")
        plt.colorbar()
        plt.savefig("simpler_pipeline/results/first_sirt_image.png", dpi=300, bbox_inches="tight")
        plt.close()

    # x for only SIRT
    x = x.clamp(0,1)
    x_sirt = x.clone()

    # DIGIT
    digit = torch.randint(0, 10, (B,), device=y.device)
    digit = torch.nn.functional.one_hot(digit, num_classes=10).float()

    if batch_id == 0:
        logger.info(f"First digit: {digit.argmax(dim=1)}")

    # CENTER
    xmin, xmax = 14, 85
    ymin, ymax = 14, 85
    centre = torch.randint(xmin, xmax+1, (B, 2), device=y.device)
    map_x = torch.full((B, H, W), 1, device=device)
    # area = (ymax - ymin + 1) * (xmax - xmin + 1)
    # map_x[:, ymin:ymax+1, xmin:xmax+1] = 1.0 / area

    if batch_id == 0:
        logger.info(f"First centre: {centre}")


    images.append(x.detach().cpu())
    images_SIRT.append(x.detach().cpu())
    centers.append(centre.detach().cpu())
    digits.append(digit.detach().cpu())

    ###### model loop

    for i in range(I):

        if batch_id == 0:
            dx_norm_mean = torch.zeros(T, device=y.device)
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
                logger.info(f"Gradient descent step {i}_{t}: {step_data.flatten(1).norm(dim=1)}")
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
            #crops = torch.clamp(crops, 0.0, 1.0)
            crop_x = crops.detach().requires_grad_(True)
            

            # calculate the value of mu() in this cropeed image         
            log_mu = - mu_model(crop_x, digit).view(B, 1)      # es nll                               

            # getting the F value and its gradient
            F =  -(sigma_data**2) * log_mu.sum()

            grad_crop = torch.autograd.grad(F, crop_x, create_graph=False)[0] 

            if batch_id == 0 and i % 2 == 0 and t==0:
                plt.imshow(grad_crop[0,0].detach().cpu().numpy(), cmap="gray")
                plt.colorbar()
                plt.savefig(f"simpler_pipeline/results/{i}_grad_crop.png", dpi=300, bbox_inches="tight")
                plt.close()

            x_scaled = x.clone()
            scale = grad_crop.norm() / (x_scaled.norm() + 1e-8)
            x_scaled = x_scaled * scale
            map_x_ = map_x.unsqueeze(1)
            grad_F = - x_scaled #/ map_x_ #x_scale


            for b in range(B):
                cy = int(centre[b, 0].item())
                cx = int(centre[b, 1].item())
                grad_F[b:b+1, :, cy-14:cy+14, cx-14:cx+14] = grad_crop[b:b+1]

            # grad descent on total => x <- x - beta * grad_F
            step_F = beta * grad_F
            if batch_id ==0:
                logger.info(f"Gradient descent step F {i}_{t}: {step_F.flatten(1).norm(dim=1)}")

            x = x + step_F

            # debugging || x^{k+1} -x^k||
            if batch_id == 0:
                dx = x - x_prev
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
            if batch_id == 0:
                dx = x_sirt - x_sirt_prev
                dx_norm_b0_sirt[t] = dx[0].flatten().norm()          # norma por elemento del batch
                dx_norm_mean_sirt[t] = dx.flatten(1).norm(dim=1).mean()             # media batch

        if batch_id == 0:
            plt.figure()
            plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch")
            plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
            plt.plot(dx_norm_mean_sirt.detach().cpu().numpy(), label="mean over batch SIRT")
            plt.plot(dx_norm_b0_sirt.detach().cpu().numpy(), label="sample 0 SIRT")
            plt.yscale("log")
            plt.xlabel("t")
            plt.ylabel("$||x_{t+1}-x_t||$")
            plt.legend()
            plt.savefig(f"simpler_pipeline/results/{i}_dx_norm_batch0.png", dpi=300, bbox_inches="tight")
            plt.close()

        if batch_id == 0 and i % 2 == 0:
            plt.imshow(x_sirt[0,0].detach().cpu().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(f"simpler_pipeline/results/{i}_sirt.png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(f"simpler_pipeline/results/{i}_flow.png", dpi=300, bbox_inches="tight")
            plt.close()

        

        #center 
        with torch.no_grad():
            # map_x[b, y, x] will store a "score" for placing the 28x28 crop center at (y, x).
            # We only fill valid centers (14..85) so the crop stays inside the 100x100 image.
            map_x = map_x = score_centers_with_unfold(
                            model =mu_model,
                            img = x,
                            digit_cond=digit,          
                            H=H, W=W,
                            patch=28,
                            ymin=14, ymax=85, xmin=14, xmax=85,
                            chunk_size=512  # prueba 256/512/1024 según memoria
                        )

        flat = map_x.view(B, -1)          # [B, H*W]
        best_idx = flat.argmax(dim=1)     # [B]

        cy = best_idx // W
        cx = best_idx %  W

        centre_new = torch.stack([cy, cx], dim=1)   # [B,2]
        centre_new = torch.stack([cy, cx], dim=1)                     # [B, 2]
        centre_new[:,0] = centre_new[:,0].clamp(14, H-14)
        centre_new[:,1] = centre_new[:,1].clamp(14, W-14)

        # digit 
        crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        for b in range(B):
            cy = int(centre[b, 0].item())
            cx = int(centre[b, 1].item())
            crop_x[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

        with torch.no_grad():
            map_digit = torch.full((B, 10), -torch.inf, device=y.device)
            for d in range(0,10):
                d_vec = torch.full((B,), d, dtype=torch.long, device=y.device) #it could be one hot or one digit for this model
                map_digit[:, d] =  - mu_model(crop_x, d_vec).view(B)

        digit = torch.softmax(map_digit, dim=1)
        centre = centre_new


        if batch_id == 0 and i % 2 == 0:
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

    if batch_id == 0 :
        logger.info(f"Original digit: {ground_truth_2}")
        logger.info(f"original centre: {ground_truth_3}")

    images, images_SIRT, centers, digits = model(input_data, batch_id)
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


torch.save(results, "simpler_pipeline/results/results_mnist_ct.pt")


for h in list(logger.handlers):
    try:
        h.flush()
        h.close()
    finally:
        logger.removeHandler(h)

logging.shutdown()