import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """
    ClassifierNet architecture MNIST

    Input:
        x ∈ R^{B×1×28x28}      # grayscale image

    Feature extraction:
        Conv2d(1 → 16, 3×3, pad=1)   → (B, 16, 28, 28)
        ReLU

        Conv2d(16 → 32, 3×3, pad=1)  → (B, 32, 28, 28)
        ReLU
        MaxPool2d(2)                → (B, 32, 14, 14)

        Conv2d(32 → 64, 3×3, pad=1)  → (B, 64, 14, 14)
        ReLU
        MaxPool2d(2)                → (B, 64, 7, 7)

    Spatial aggregation:
        Conv2d(64 → 64, 3×3, pad=1)
        ReLU
        AdaptiveAvgPool2d(1)        → (B, 64, 1, 1)

    Classification:
        Flatten                     → (B, 64)
        Linear(64 → 10)              → logits

    Key idea:
        - Convolutions learn local patterns and shapes
        - Pooling provides translation invariance
        - AdaptiveAvgPool removes spatial dependence (digit can be anywhere)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)