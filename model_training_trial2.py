from code.models.model_base import ModelBase
from code.models.task2_model import CenterRegressor
from accelerate import Accelerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tomosipo as ts
import numpy as np
import torch

accelerator = Accelerator()


#parameter values
batch_size = 10
epochs = 40
H = 100
W = 100
dimension = (H, W)
cut = False
seed = 29072000
optimizer_type = "SGD"
learning_rate = 1e-3
scheduler = True

save_path = "/home/as3628/multi-task-learning/results/centre finder completed"


exp2 = ModelBase(
    model_path="mnist_classifier_complete",
    dimension=dimension,
    cut=cut,
    seed=seed,
    batch_size=batch_size,
    epochs=epochs,
    optimizer_type=optimizer_type,
    learning_rate=learning_rate,
    debug=False,
    accelerator=accelerator,
    task = "centre_finder",
    scheduler= scheduler)


exp2.model = CenterRegressor()        # centro (x, y)




