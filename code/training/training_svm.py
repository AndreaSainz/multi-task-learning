import os
from dataclasses import dataclass
import json
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from code.models.classifier import CNNClassifier
from code.models.denoiser import ResUNetDenoiser
import numpy as np
import random 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.metrics import accuracy_score

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

# parameters
model_name = "svm"
batch_size = 128
epochs = 30
lr= 1e-3
weight_decay = 1e-4
num_workers = 2
noise_std_max = 0.4
device= "cuda" 
save_dir = "checkpoints/svm"
os.makedirs(save_dir, exist_ok=True)

# Data
print("Strating: data")
transform = transforms.ToTensor()

train_dataset = datasets.MNIST( root="./data", train=True, download=True, transform=transform )
test_dataset = datasets.MNIST( root="./data", train=False, download=True, transform=transform )

X_train = train_dataset.data.view(-1, 28*28).numpy()  / 255.0
y_train = train_dataset.targets.numpy()

X_test = test_dataset.data.view(-1, 28*28).numpy() / 255.0
y_test = test_dataset.targets.numpy()

print("Finish: data")
# fig, axes = plt.subplots(1, 1, figsize=(4, 4))

# im = axes.imshow(X_train[0].reshape(28, 28), cmap="gray")
# axes.set_title("SIRT only")
# axes.axis("off")

# # common color bar 
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# fig.colorbar(im, cax=cbar_ax)

# fig.subplots_adjust(right=0.9)
# plt.savefig(os.path.join(save_dir,"comprobacion.png"), dpi=300, bbox_inches="tight")
# plt.close(fig)

# TRAINING
model = SVC(kernel="rbf", C=1.0, gamma="scale")

print("Starting: training")
model.fit(X_train, y_train)
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Finish: training")
# EVALUATION
print("Starting: evaluation")
y_pred = model.predict(X_test)

print("Finish: evaluation")
print("Starting: matrix")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm)
plt.title("Confusion Matrix ")
plt.colorbar()
classes = list(range(10))
plt.xticks(classes)
plt.yticks(classes)
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, int(cm[i, j]),
                 ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Finish: matrix")
print("Accuracy:", accuracy_score(y_test, y_pred))