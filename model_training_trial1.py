from code.models.model_base import ModelBase
from code.models.task1_model import ClassifierNet
from accelerate import Accelerator
import os
import random
import numpy as np
import torch


accelerator = Accelerator()


#parameter values
batch_size = 10
epochs = 15
H = 100
W = 100
dimension = (H, W)
cut = False
seed = 29072000
optimizer_type = "AdamW"
learning_rate = 1e-3
scheduler = True

save_path = "results/classifier completed"

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


exp1 = ModelBase(
    model_path=f"{save_path}/mnist_classifier_complete",
    dimension=dimension,
    cut=cut,
    seed=seed,
    batch_size=batch_size,
    epochs=epochs,
    optimizer_type=optimizer_type,
    learning_rate=learning_rate,
    debug=True,
    accelerator=accelerator,
    task = "classify",
    scheduler= scheduler)

exp1.model = ClassifierNet(10)        # clasificador MNIST


# Train + test
history = exp1.train()
results = exp1.test()

# Plots (si ya las tienes)
exp1.plot_training(history, out_path=f"{save_path}/{exp1.model_path}_train.png")
exp1.plot_validation_metrics(history, out_path=f"{save_path}/{exp1.model_path}_valmetrics.png")
exp1.plot_test(out_path=save_path, normalize_cm=False)
exp1.plot_test(out_path=save_path, normalize_cm=True)


