from code.models.project_base import ProjectBase
from code.models.inverse_model import ReconNet
from code.models.task1_model import ClassifierNet
from code.models.task2_model import CenterRegressor
from accelerate import Accelerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tomosipo as ts
import numpy as np
import torch
import os
import random

accelerator = Accelerator()


#parameter values
batch_size = 10
epochs = 20
H = 100
W = 100
dimension = (H, W)
num_angles = 30
sigma = 0.01
i_0 = 10
cut = False
seed = 29072000
optimizer_type = ("AdamW", "AdamW")
learning_rate = (1e-3, 1e-3)
scheduler = (True, True) 

vg = ts.volume(shape=(1,H,W))                            # Volumen
angles = np.linspace(0, np.pi, num_angles, endpoint=True)     # Angles
pg = ts.parallel(angles=angles, shape=(1, H))            # Parallel beam structure
A = ts.operator(vg,pg) 

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

save_path = "/home/as3628/multi-task-learning/results/experiment_10_30"     

exp = ProjectBase(
    model_path=f"{save_path}/mnist_ct_experiment_10_30",
    dimension=dimension,
    num_angles=num_angles,
    sigma=sigma,
    i_0=i_0,
    cut=cut,
    seed=seed,
    batch_size=batch_size,
    epochs=epochs,
    optimizer_type=optimizer_type,
    learning_rate=learning_rate,
    debug=True,
    accelerator=accelerator,
    scheduler=scheduler,
)


exp.model1 = ReconNet(A, H, W, 10)    # sinograma → imagen
exp.model2 = ClassifierNet(10)        # clasificador MNIST
exp.model3 = CenterRegressor(H,W)     # centro (x, y)


history = exp.train(max_len_train = None, max_len_val = None)
results = exp.test(max_len_test = None)

exp.plot_per_model_metrics(history, out_dir=save_path)
exp.plot_test_per_model(out_dir=save_path)
