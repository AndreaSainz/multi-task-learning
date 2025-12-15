import torch.nn as nn
import torch.nn.functional as F

class ClassifierNet(nn.Module):
    """
    ClassifierNet architecture (MNIST on 100x100 canvas)

    Input:
        x ∈ R^{B×1×100×100}      # grayscale image, digit randomly translated

    Feature extraction:
        Conv2d(1 → 16, 3×3, pad=1)   → (B, 16, 100, 100)
        ReLU

        Conv2d(16 → 32, 3×3, pad=1)  → (B, 32, 100, 100)
        ReLU
        MaxPool2d(2)                → (B, 32, 50, 50)

        Conv2d(32 → 64, 3×3, pad=1)  → (B, 64, 50, 50)
        ReLU
        MaxPool2d(2)                → (B, 64, 25, 25)

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
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 100 -> 50
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 50 -> 25
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)
