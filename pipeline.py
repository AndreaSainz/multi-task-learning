import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_loader import MnistDataset
from code.utils.metrics import compute_psnr_results, compute_ssim, compute_mse, update_confusion_matrix, metrics_from_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Parameters

### data
seed = 29072000
H,W = 100
num_angles = 30
i_0 = 10
sigma_data = 0.1
accelerator = Accelerator()
device = accelerator.device
n_train = 100000
batch_size = 10

### training
T = 10
I = 10
alpha = 1e-6
sigma = 0.1                   ## tengo que probar comomafecta este sigma al entrnamiento y a las imagenes reconstruidas

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

# create dataloader for both 
train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    generator=g_split)

    

# Import trained mu(x_cropped|digit)

mu_model = 

mu_model.eval()
for p in mu_model.parameters():
    p.requires_grad_(False)





# defining the whole idea of the inverse problem
def model(y):


    B = y.shape[0]


    # Initilizing tasks

    # RECONSTRUCTION
    x = torch.zeros(B, 1, H, W, device=y.device)


    for t in range(T):
        Ax = A_torch(x)
        r = y - Ax
        AT_r = AT_torch(r)
        x = x + alpha * AT_r


    # DIGIT
    digit = torch.randint(0, 10, (B,), device=y.device)

    # CENTER
    centre = torch.randint(14, 86, (B, 2), device=y.device)


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

            # calculate the value of mu() in this cropeed image         
            mu = mu_model(crop_x, digit)            
            mu = mu.sum()                         

            # getting the F value and its gradient
            F = - sigma**2 * mu
            grad_F = torch.autograd.grad(F, x, create_graph=False)[0]

            # grad descent on total => x <- x - alpha * grad_F
            x = x - alpha * grad_F

        #center 
        with torch.no_grad():
            map_x = torch.full((B, H, W), -torch.inf, device=y.device) #this is because we are only going to evaluate in possible centers, not in all
            for i_x in range(14,86):
                for i_y in range(14,86):
                    crop_x = x[:, :, i_y-14:i_y+14, i_x-14:i_x+14]
                    map_x[:, i_y,i_x] = mu_model(crop_x, digit).view(B)

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
                map_digit[:, d] = mu_model(crop_x, d_vec).view(B)

        digit = map_digit.view(B, -1).argmax(dim=1)
        centre = centre_new

