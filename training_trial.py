from model_base import ModelBase
from inverse_model import ReconNet
from task1_model import ClassifierNet
from task2_model import CenterRegressor
from accelerate import Accelerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tomosipo as ts
import numpy as np
import torch

accelerator = Accelerator()


#parameter values
batch_size = 1
epochs = 3
H = 100
W = 100
dimension = (H, W)
num_angles = 30
sigma = 0.01
standard_dose = True
cut = False
seed = 29072000
device = torch.device("cuda:0")
optimizer_type = ("SGD", "SGD", "SGD")
learning_rate = (1e-3, 1e-3, 1e-3)
scheduler = (True, True, True) 

save_path = "/home/as3628/multi-task-learning/results"

# Create tomosipo volume and projection geometry
vg = ts.volume(shape=(1,H,W))                                                       # Volumen
angles = np.linspace(0, np.pi, num_angles, endpoint=True)                           # Angles
pg = ts.parallel(angles=angles, shape=(1, H))                                       # Parallel beam structure
A = ts.operator(vg,pg)     

exp = ModelBase(
    model_path="mnist_ct_experiment",
    dimension=dimension,
    num_angles=num_angles,
    sigma=sigma,
    standard_dose=standard_dose,
    cut=cut,
    seed=seed,
    batch_size=batch_size,
    epochs=epochs,
    optimizer_type=optimizer_type,
    learning_rate=learning_rate,
    debug=False,
    accelerator=accelerator,
    scheduler=scheduler,
)


exp.model1 = ReconNet(A, H, W)          # sinograma → imagen
exp.model2 = ClassifierNet(10)   # clasificador MNIST
exp.model3 = CenterRegressor()   # centro (x, y)


history = exp.train(
    training_path=save_path,      
    validation_path=save_path,      
    max_len_train=2,               
    max_len_val=1,                 
    epochs=None,                   
)

plt.figure(figsize=(7,5))
plt.plot(history["train_loss_1"], label="Train Loss 1")
plt.plot(history["val_loss_1"], label="Val Loss 1")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 1 — Reconstruction Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{save_path}_model1_loss.png", dpi=200)

# --- Modelo 2: Clasificador ---
plt.figure(figsize=(7,5))
plt.plot(history["train_loss_2"], label="Train Loss 2")
plt.plot(history["val_loss_2"], label="Val Loss 2")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 2 — Classification Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{save_path}_model2_loss.png", dpi=200)


# --- Modelo 3: Centro ---
plt.figure(figsize=(7,5))
plt.plot(history["train_loss_3"], label="Train Loss 3")
plt.plot(history["val_loss_3"], label="Val Loss 3")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 3 — Center Regression Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{save_path}_model3_loss.png", dpi=200)


results = exp.test(
    test_path=save_path,   
    max_len_test=1)        

print(results)

