import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """
    ClassifierNet architecture MNIST or FashionMNIST

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

    Parameters:
        num_classes (int):
            Number of output classes. Default: 10.

        noise_aware (bool):
            Enables noise-aware behaviour if implemented
            in the network's forward pass or architecture.

    Output:
        Logits ∈ R^{B×num_classes}
        Raw class scores; softmax is not applied.

    Key idea:
        - Convolutions learn local patterns and shapes
        - Pooling provides translation invariance
        - AdaptiveAvgPool removes spatial dependence (digit can be anywhere)

    Notes:
        - Input images are expected to have one channel
          and a resolution of 28×28.
        - The model supports batches of images.
        - CrossEntropyLoss can be applied directly to logits.
        - The actual effect of noise_aware depends on its
          implementation in __init__ and forward.
    """
    
    
    def __init__(self, num_classes=10, noise_aware=False):
        super().__init__()

        self.noise_aware = noise_aware

        # 1 channel for gay images or 2 channels (image + noise)
        in_channels = 2 if noise_aware else 1

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
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

    def forward(self, x, noise_level=None):

        # Adding noise if Noise Aware if True
        if self.noise_aware:
            if noise_level is None:
                raise ValueError(
                    "noise_level is required when noise_aware=True"
                )

            # One noise per image in the batch
            if noise_level.ndim == 1:
                noise_level = noise_level.unsqueeze(1)

            # noise is a constant channel
            noise_channel = noise_level.reshape(-1, 1, 1, 1)
            noise_channel = noise_channel.expand(
                -1, 1, x.shape[2], x.shape[3]
            )

            x = torch.cat([x, noise_channel], dim=1)

        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)