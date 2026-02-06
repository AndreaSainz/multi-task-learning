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
n_train = 500
batch_size = 10
save_batch_id = 0

### training
T = 50
I = 50
alpha = 1e-8
sigma = 1e-4             ## tengo que probar comomafecta este sigma al entrnamiento y a las imagenes reconstruidas

# Import data

# define CT geometry
vg = ts.volume(shape=(1,H,W))                            # Volume
angles = np.linspace(0, np.pi, num_angles, endpoint=True)     # Angles
pg = ts.parallel(angles=angles, shape=(1, H))            # Parallel beam structure
A = ts.operator(vg,pg)

A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)



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
#mu_model = nf.MultiscaleFlow(q0, flows, merges)
#mu_model = mu_model.to(device)

#checkpoint = torch.load("flow_model.pt", map_location=device)
#mu_model.load_state_dict(checkpoint["model_state_dict"])

#mu_model.eval() 
#for p in mu_model.parameters():
#    p.requires_grad_(False)


#function tu get the term that we expect
#def logpx_digit(x, digit):
#    """
#    Compute exact log p(x | digit) for a conditional normalizing flow (cINN).
#
#    x:     [B, 1, 28, 28] float tensor
#    digit: [B] int64 tensor with values in {0,...,9}
#
#    Returns:
#        log_px: [B] tensor with log p(x | digit)
#    """
#    # the mu_model expects the digit as 1-hot vector
#    digit = digit.long()
#    cond = F.one_hot(digit, num_classes=10).float().to(x.device)

#    #the model gives you the original z and the logarithm of the determinan of the jacobian
#    z, log_jac = mu_model(x, cond, rev=False)
#    z = z.view(z.shape[0], -1)

#    # using the equation from normalizing flow we can calculate the log p(x|cond) = log p(z) + log det(Jacobian)
#    D = z.shape[1]
#    log_pz = -0.5 * (z**2).sum(dim=1) - 0.5 * D * math.log(2*math.pi)

#    return log_pz + log_jac

def compute_prototypes(train_dataloader, device="cpu"):
    """
    train_dataset: dataset que devuelve dict con clave 'label' y
                   una imagen MNIST original (28x28) o accesible.
    Returns:
        proto: [10, 1, 28, 28]
    """
    proto_sum = torch.zeros(10, 1, 28, 28, device=device)
    counts = torch.zeros(10, device=device)

    for i in range(len(train_dataloader)):
        sample = train_dataloader[i]

        digit = int(sample["label"])
        
        # OJO: aquí necesitamos la imagen MNIST ORIGINAL (28x28),
        # no el canvas 100x100 ni el sinograma.
        # Ajusta la key si en tu dataset se llama distinto.
        img = sample["mnist_image"]      # [1,28,28]
        
        proto_sum[digit] += img
        counts[digit] += 1

    # evitar división por cero
    for d in range(10):
        if counts[d] > 0:
            proto_sum[d] /= counts[d]

    return proto_sum

proto1 = compute_prototypes(balanced_train, device=device)


fig, axs = plt.subplots(2, 5, figsize=(10,4))
for d in range(10):
    ax = axs[d//5, d%5]
    ax.imshow(proto1[d,0].cpu(), cmap="gray")
    ax.set_title(str(d))
    ax.axis("off")
plt.savefig("simpler_pipeline/results/proto_second.png")
plt.close()


def logpx_digit(crop_x, digit, proto, sigma_p=0.2):
    """
    crop_x: [B,1,28,28]
    digit:  [B] long
    proto:  [10,1,28,28]
    """
    digit = digit.long()
    mu_d = proto[digit]                 # [B,1,28,28]
    diff = crop_x - mu_d
    sq = (diff**2).sum(dim=(1,2,3))     # [B]
    return -0.5 * sq / (sigma_p**2)     # constante omitida



# defining the whole idea of the inverse problem
def model(y, batch_id):
    images_SIRT = []
    images = []
    centers = []
    digits = []

    B = y.shape[0]


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

    if batch_id == 0:
        logger.info(f"First digit: {digit}")

    # CENTER
    centre = torch.randint(14, 86, (B, 2), device=y.device)

    if batch_id == 0:
        logger.info(f"First centre: {centre}")


    images.append(x.detach().cpu())
    images_SIRT.append(x.detach().cpu())
    centers.append(centre.detach().cpu())
    digits.append(digit.detach().cpu())

    # model loop

    for i in range(I):

        # reconstruction
        for t in range(T):
            Ax = A_torch(x)
            r = y - Ax
            AT_r = AT_torch(r)
            x = x + alpha * AT_r

            # regularization term F
            x = x.detach().requires_grad_(True)

            # we need to crop x as an MNIST digit 28x28
            crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
            for b in range(B):
                cy = int(centre[b, 0].item())
                cx = int(centre[b, 1].item())
                crop_x[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

            if batch_id == 0 and i == 5:
                plt.imshow(crop_x[0,0].detach().cpu().numpy(), cmap="gray")
                plt.colorbar()
                plt.savefig("simpler_pipeline/results/crop_image_second.png", dpi=300, bbox_inches="tight")
                plt.close()

            # calculate the value of mu() in this cropeed image         
            log_mu = logpx_digit(crop_x, digit, proto1)            
            log_mu = log_mu.sum()                         

            # getting the F value and its gradient
            F = - sigma**2 * log_mu
            grad_F = torch.autograd.grad(F, x, create_graph=False)[0] # REVISAR ESTA PARTE
            if batch_id == 0 and i % 5== 0:
                plt.imshow(grad_F[0,0].detach().cpu().numpy(), cmap="gray")
                plt.colorbar()
                plt.savefig(f"simpler_pipeline/results/{i}_grad_Fsecond.png", dpi=300, bbox_inches="tight")
                plt.close()

            # grad descent on total => x <- x - alpha * grad_F
            x = x - grad_F

            
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
            map_x = torch.full((B, H, W), -torch.inf, device=y.device) #this is because we are only going to evaluate in possible centers, not in all
            for i_x in range(14,86):
                for i_y in range(14,86):
                    crop_x = x[:, :, i_y-14:i_y+14, i_x-14:i_x+14]
                    map_x[:, i_y,i_x] =  logpx_digit(crop_x, digit, proto1).view(B)

        flat_idx = map_x.view(B, -1).argmax(dim=1)  # [B]

        ys = flat_idx // W
        xs = flat_idx % W
        centre_new = torch.stack([ys, xs], dim=1)


        # digit 
        crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        for b in range(B):
            cy = int(centre[b, 0].item())
            cx = int(centre[b, 1].item())
            crop_x[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

        with torch.no_grad():
            map_digit = torch.full((B, 10), -torch.inf, device=y.device)
            for d in range(0,10):
                d_vec = torch.full((B,), d, dtype=torch.long, device=y.device)
                map_digit[:, d] =  logpx_digit(crop_x, digit, proto1).view(B)

        digit = map_digit.argmax(dim=1)
        centre = centre_new


        if batch_id == 0 and i % 5 == 0:
            logger.info(f"{i}th digit: {digit}")
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

torch.save(results, "simpler_pipeline/results/results_mnist_ct_second.pt")

for h in list(logger.handlers):
    try:
        h.flush()
        h.close()
    finally:
        logger.removeHandler(h)

logging.shutdown()