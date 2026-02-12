# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import logging
import os
from datetime import datetime

# ---- logging setup ----
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir, 
    f"nf_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()   # sigue imprimiendo en consola
    ]
)

logger = logging.getLogger(__name__)
logger.info("Training started")



def check_params_finite(model):
    for p in model.parameters():
        if p is not None and p.data is not None and not torch.isfinite(p.data).all():
            return False
    return True

def grad_finite(model):
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True

def evaluate_nll(model, loader):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            logp = model(x.to(device), y.to(device))
            total += (-logp).sum().item()
            n += x.size(0)
    model.train()
    return total / n


# Set up model

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
eval_every = 800
ckpt_every = 2000

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


# Construct flow model with the multiscale architecture
model = nf.MultiscaleFlow(q0, flows, merges)


# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)

# # Prepare training data
batch_size = 128

transform = tv.transforms.Compose([tv.transforms.ToTensor(), nf.utils.Scale(255. / 256.), nf.utils.Jitter(1 / 256.)])
train_data = MNIST(root = 'data/', train = True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                           drop_last=True)


test_data = MNIST(root = 'data/', train = False, transform = transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

train_iter = iter(train_loader)

x,y = next(iter(test_loader))
x,y = x.to(device), y.to(device)
print("forward_kld:", model.forward_kld(x,y).item())
print("-mean logp:", (-model(x,y).mean()).item())
print("mean logp:", model(x,y).mean().item())

# Train model
max_iter = 50000

loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9,0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.5)

for i in tqdm(range(max_iter)):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    optimizer.zero_grad()
    loss = model.forward_kld(x.to(device), y.to(device))
    
    # 1) loss finita
    if not torch.isfinite(loss).item():
        logger.error(f"[iter {i}] loss no-finita: {loss.item()}")
        break

    # backward
    loss.backward()

    total_norm = torch.norm(
    torch.stack([torch.norm(p.grad.detach(), 2) 
                 for p in model.parameters() 
                 if p.grad is not None]), 
    2
    )

    #print("Global grad norm:", total_norm.item())

    # 2) gradientes finitos (opcional pero recomendable)
    if not grad_finite(model):
        logger.error(f"[iter {i}] gradientes no-finitos -> STOP")
        break


    # clip + step
    optimizer.step()
    if i % eval_every == 0:
        val_loss = evaluate_nll(model, test_loader)  
        scheduler.step(val_loss)

    # 3) parámetros finitos tras el step
    if not check_params_finite(model):
        logger.error(f"[iter {i}] parámetros no-finitos tras optimizer.step() -> STOP")
        break

    loss_hist = np.append(loss_hist, loss.detach().cpu().numpy())

    
    if i % ckpt_every == 0 and i > 0:
        torch.save({"iter": i, "model_state_dict": model.state_dict(),
                    "opt_state_dict": optimizer.state_dict()}, 
                os.path.join(log_dir, f"ckpt_{i}.pt"))






plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig("loss_hist_normalizing_flows_1.png")

#saving model
save_path = "flow_model_1.pt"

torch.save({
    "model_state_dict": model.state_dict(),
}, save_path)

 

# Model samples
num_sample = 10
model.eval()



with torch.no_grad():
    y = torch.arange(num_classes).repeat(num_sample).to(device)
    x, _ = model.sample(y=y)
    x_ = torch.clamp(x, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0)))
    plt.savefig("sample_normalizing_flows_1.png")


# Get bits per dim
n = 0
bpd_cum = 0
with torch.no_grad():
    for x, y in iter(test_loader):
        nll = model(x.to(device), y.to(device))

        print("finite ratio:", torch.isfinite(nll).float().mean().item())
        print("min/max:", nll.min().item(), nll.max().item())

        bpd = (nll / (np.log(2) * n_dims) + 8)  # [B]

        bpd_cum += torch.nan_to_num(bpd, nan=0.0).sum().item()
        n += torch.isfinite(bpd).sum().item()

final_bpd = bpd_cum / n
logger.info(f"Final Bits per dim: {final_bpd}")
print("Bits per dim:", final_bpd)

for h in list(logger.handlers):
    try:
        h.flush()
        h.close()
    finally:
        logger.removeHandler(h)

logging.shutdown()