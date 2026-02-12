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

# Parameters

### data
seed = 29072000
H = 100
W = 100
num_angles = 30
i_0 = 10
sigma_data = 0.1
accelerator = Accelerator()
device = "cuda"
n_train = 100
batch_size = 10
save_batch_id = 0
k = int(0.05 * H * W)   # top 5%

### training
T = 30
I = 30
alpha = 1e-3
sigma_value = 1e-6  ## tengo que probar comomafecta este sigma al entrnamiento y a las imagenes reconstruidas
gamma = 0.5

# Import data

# define CT geometry
vg = ts.volume(shape=(1,H,W))                            # Volume
angles = np.linspace(0, np.pi, num_angles, endpoint=True)     # Angles
pg = ts.parallel(angles=angles, shape=(1, H))            # Parallel beam structure
A = ts.operator(vg,pg)

# this is for normalizing the operator
x_ref = torch.ones(1,H,W, device=device)
y_ref = A(x_ref)
scale_fact = y_ref.abs().max()


A_raw  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_raw = to_autograd(A.T, is_2d=True, num_extra_dims=2)

# define normalized operators
def A_torch(x):
    return A_raw(x) / scale_fact

def AT_torch(y):
    return AT_raw(y) / scale_fact

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
splits = random_split(data,[n_train, len(data)-n_train] , generator=g_split)

train_data = splits[0]

# be sure all digits are equally represented
labels = torch.tensor([train_data[i]["label"] for i in range(len(train_data))])

idx_by_class = [torch.where(labels == k)[0] for k in range(10)]
min_count = min(len(ix) for ix in idx_by_class)

balanced_idx = torch.cat([ix[:min_count] for ix in idx_by_class])
# mix 
balanced_idx = balanced_idx[torch.randperm(len(balanced_idx), generator=g_split)]

balanced_train = Subset(train_data, balanced_idx.tolist())

# create dataloader
train_dataloader = DataLoader(
    balanced_train,
    batch_size=batch_size,
    shuffle=True,
    generator=g_split)


# create logger for checkpoints
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "second_try.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


#Check if the dataset is balanced
labels_bal = torch.tensor([balanced_train[i]["label"] for i in range(len(balanced_train))])
counts = torch.bincount(labels_bal, minlength=10)

logger.info("Balanced dataset label counts:")
for d in range(10):
    logger.info(f"Digit {d}: {counts[d].item()}")


# Import trained mu(x_cropped|digit)
# Define flows
L = 2 #number of latent variables (z_l)
K = 24 #how complex the invertible function is
torch.manual_seed(29072000)

input_shape = (1, 28, 28)
n_dims = np.prod(input_shape)
channels = 1
hidden_channels = 512 #how large are CNN inside GlowBlocks, so s() and t()
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

checkpoint = torch.load("flow_model_3.pt", map_location=device)
mu_model.load_state_dict(checkpoint["model_state_dict"])

mu_model.eval() 
for p in mu_model.parameters():
   p.requires_grad_(False)




# defining the whole idea of the inverse problem
def model(y, batch_id):
    images_SIRT = []
    images = []
    centers = []
    digits = []

    B = y.shape[0]
    sigma = torch.full((B,1), sigma_value, device=y.device) 


    # Initilizing tasks

    # RECONSTRUCTION
    x = torch.zeros(B, 1, H, W, device=y.device)


    for t in range(T):
        Ax = A_torch(x)
        r = y - Ax
        AT_r = AT_torch(r)
        x = x + alpha * AT_r

    if batch_id == 0:
        plt.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")
        plt.colorbar()
        plt.savefig("simpler_pipeline/results/first_sirt_image_second.png", dpi=300, bbox_inches="tight")
        plt.close()

    # x for only SIRT
    x_sirt = x.clone()

    # DIGIT
    digit = torch.randint(0, 10, (B,), device=y.device)
    digit = torch.nn.functional.one_hot(digit, num_classes=10).float()

    if batch_id == 0:
        logger.info(f"First digit: {digit.argmax(dim=1)}")

    # CENTER
    centre = torch.randint(14, 86, (B, 2), device=y.device)

    if batch_id == 0:
        logger.info(f"First centre: {centre}")


    images.append(x.detach().cpu())
    images_SIRT.append(x.detach().cpu())
    centers.append(centre.detach().cpu())
    digits.append(digit.detach().cpu())

    # model loop
    var_prev = None

    for i in range(I):

        # reconstruction
        for t in range(T):
            Ax = A_torch(x)
            r = y - Ax
            AT_r = AT_torch(r)
            x = x + alpha * AT_r

            # regularization term F
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
            

            if batch_id == 0 and i == 5:
                plt.imshow(crop_x[0,0].detach().cpu().numpy(), cmap="gray")
                plt.colorbar()
                plt.savefig("simpler_pipeline/results/crop_image_second.png", dpi=300, bbox_inches="tight")
                plt.close()

            # calculate the value of mu() in this cropeed image         
            log_mu = mu_model(crop_x, digit).view(B, 1)                                    

            # getting the F value and its gradient
            F = - (sigma**2) * log_mu
            loss_reg = F.sum()

            grad_crop = torch.autograd.grad(loss_reg, crop_x, create_graph=False)[0] # REVISAR ESTA PARTE (ver si genera no 0)
            grad_F = torch.zeros_like(x)

            for b in range(B):
                cy = int(centre[b, 0].item())
                cx = int(centre[b, 1].item())
                grad_F[b:b+1, :, cy-14:cy+14, cx-14:cx+14] = grad_crop[b:b+1]

            if batch_id == 0 and i % 5== 0:
                plt.imshow(grad_F[0,0].detach().cpu().numpy(), cmap="gray")
                plt.colorbar()
                plt.savefig(f"simpler_pipeline/results/{i}_grad_F_second.png", dpi=300, bbox_inches="tight")
                plt.close()

            # grad descent on total => x <- x - alpha * grad_F
            x = x -  alpha * grad_F

            
            #only SIRT
            Ax_sirt = A_torch(x_sirt)
            r_sirt = y - Ax_sirt
            AT_r_sirt = AT_torch(r_sirt)
            x_sirt = x_sirt + alpha * AT_r_sirt


        if batch_id == 0 and i % 5== 0:
            plt.imshow(x_sirt[0,0].detach().cpu().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(f"simpler_pipeline/results/{i}_sirt_image_second.png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(f"simpler_pipeline/results/{i}_sirt_flow_image_second.png", dpi=300, bbox_inches="tight")
            plt.close()


        
        

        #center 
        with torch.no_grad():
             # map_x[b, y, x] will store a "score" for placing the 28x28 crop center at (y, x).
            # We only fill valid centers (14..85) so the crop stays inside the 100x100 image.
            map_x = torch.full((B, H, W), -torch.inf, device=y.device)
            for i_x in range(14,86):
                for i_y in range(14,86):
                    # Extract the 28x28 patch if the center were (i_y, i_x)
                    crop_x = x[:, :, i_y-14:i_y+14, i_x-14:i_x+14]

                    # Score the patch under the digit-conditional flow
                    # NOTE: the '-' assumes mu_model returns NLL; if it returns log_prob, remove the minus.
                    map_x[:, i_y,i_x] =  mu_model(crop_x, digit).view(B)

        # Flatten the (H,W) grid into a vector of length H*W so we can do softmax/topk per batch element
        flat = map_x.view(B, -1)

        # Stabilize numerically by subtracting the per-sample maximum score (softmax invariance)
        flat = flat - flat.max(dim=1, keepdim=True).values
        
        # Convert scores into a probability distribution over all centers
        w = torch.softmax(flat, dim=1)  # shape: [B, H*W]

        # --- TOP-K SELECTION (keep only the k most probable centers) ---
        # Get indices of the k largest scores (equivalently, highest probabilities after softmax)
        topk_vals, topk_idx = w.topk(k, dim=1)  # shapes: [B,k], [B,k]

        # The MAP center within the top-k is simply the best of those k (the first one from topk)
        best_idx = topk_idx[:, 0]  # shape: [B]

        # Build a binary mask that is 1 at the top-k indices, 0 elsewhere
        mask = torch.zeros_like(w)           # [B, H*W]
        mask.scatter_(1, topk_idx, 1.0)      # set top-k positions to 1

        # Truncate the distribution: keep probability mass only on top-k, zero out the rest
        w_thr = w * mask

        # Renormalize so w_thr sums to 1 (avoid division by zero with a small clamp)
        w_thr = w_thr / w_thr.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # --- MEAN / VARIANCE COMPUTATION ON THE TRUNCATED (TOP-K) DISTRIBUTION ---
        # Build a coordinate grid for all pixels, flattened to match the H*W indexing
        ys = torch.arange(H, device=map_x.device)
        xs = torch.arange(W, device=map_x.device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([Y.flatten(), X.flatten()], dim=0)  # [2, H*W], coords[0]=y, coords[1]=x

        # Compute the (y,x) expectation under the truncated distribution
        mean = (w_thr.unsqueeze(1) * coords.unsqueeze(0)).sum(dim=2)  # [B, 2]

        # Compute weighted variance in y and x, then average to get a scalar "spread"
        diff = coords.unsqueeze(0) - mean.unsqueeze(2)                # [B, 2, H*W]
        var_xy = (w_thr.unsqueeze(1) * diff.pow(2)).sum(dim=2)        # [B, 2]
        var_s = var_xy.mean(dim=1, keepdim=True) / ((H/2)**2)  # [B,1]

        sigma0 = sigma_value
        if var_prev is None:
            var_prev = var_s.detach()
        else:
            improve = (var_s < var_prev)   # mejora si var baja

            ratio = (var_prev / var_s).clamp(1.0, 2.0)  # solo deja aumentar
            sigma_up = (sigma * ratio.pow(gamma)).clamp(sigma0, 1e-2)

            sigma = torch.where(improve, sigma_up, torch.full_like(sigma, sigma0))
            var_prev = var_s.detach()


        # --- MAP CENTER (argmax) AS THE NEW DISCRETE CENTER ---
        # Convert best_idx (flattened index) back to (y, x)
        cy = best_idx // W
        cx = best_idx %  W
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
                map_digit[:, d] =  mu_model(crop_x, d_vec).view(B)

        digit = torch.softmax(map_digit, dim=1)
        centre = centre_new


        if batch_id == 0 and i % 5 == 0:
            logger.info(f"{i}th digit: {map_digit.argmax(dim=1)}")
            logger.info(f"{i}th centre: {centre}")

        #save information
        images.append(x.detach().cpu())
        images_SIRT.append(x_sirt.detach().cpu())
        centers.append(centre.detach().cpu())
        digits.append(digit.detach().cpu())

    return images, images_SIRT, centers, digits


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
    input_data = batch["noisy_sinogram"]/scale_fact

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

torch.save(results, "simpler_pipeline/results/results_mnist_ct_second.pt")

for h in list(logger.handlers):
    try:
        h.flush()
        h.close()
    finally:
        logger.removeHandler(h)

logging.shutdown()