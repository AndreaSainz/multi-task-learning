from code.models.model_base import ModelBase
from code.models.task2_model import CenterRegressor
from accelerate import Accelerator
import os
import random
import numpy as np
import torch


accelerator = Accelerator()


#parameter values
batch_size = 10
epochs = 20
H = 100
W = 100
dimension = (H, W)
cut = False
seed = 29072000
optimizer_type = "AdamW"
learning_rate = 1e-3
scheduler = True

save_path = "results/centre finder completed"

# Deterministic
# Python
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)

# NumPy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Deterministic behaviour
torch.use_deterministic_algorithms(True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Needed for full CUDA determinism (must be set BEFORE CUDA ops)
if torch.cuda.is_available():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


exp2 = ModelBase(
    model_path=f"{save_path}/mnist_centre_finder_complete",
    dimension=dimension,
    cut=cut,
    seed=seed,
    batch_size=batch_size,
    epochs=epochs,
    optimizer_type=optimizer_type,
    learning_rate=learning_rate,
    debug=True,
    accelerator=accelerator,
    task = "centre_finder",
    scheduler= scheduler)


exp2.model = CenterRegressor(H,W)        # centro (x, y)

# Train + test
history = exp2.train()
results = exp2.test()

# Plots (si ya las tienes)
exp2.plot_training(history, out_path=f"{save_path}/{exp2.model_path}_train.png")
exp2.plot_validation_metrics(history, out_path=f"{save_path}/{exp2.model_path}_valmetrics.png")
exp2.plot_test(out_path=save_path, normalize_cm=False)
exp2.plot_test(out_path=save_path, normalize_cm=True)