import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tomosipo.torch_support import to_autograd
import numpy as np
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from code.dataloader.data_loader import MnistDataset
from code.models.classifier import CNNClassifier
import matplotlib.pyplot as plt

class_model = CNNClassifier()

checkpoint = torch.load("/home/as3628/multi-task-learning/checkpoints/classifier/classifier_best_finetunned.pt", map_location="cpu")
class_model.load_state_dict(checkpoint["model"])

class_model.eval() 
for p in class_model.parameters():
   p.requires_grad_(False)

class_model = class_model.to("cuda")
for i in range(10):
   x = torch.rand(1, 1, 28, 28, device="cuda")

   fig, axes = plt.subplots(1, 1, figsize=(20, 4))

   im = axes.imshow(x[0,0].detach().to("cpu"), cmap="gray", vmin=0, vmax=1)
   axes.set_title("SIRT only")
   axes.axis("off")

   cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
   fig.colorbar(im, cax=cbar_ax)

   fig.subplots_adjust(right=0.9)
   plt.savefig(f"/home/as3628/multi-task-learning/simpler_pipeline/Results/{i}_uniform", dpi=300, bbox_inches="tight")
   plt.close(fig)

   logits = class_model(x)
   prob = torch.softmax(logits, dim=1)

   print(prob)

