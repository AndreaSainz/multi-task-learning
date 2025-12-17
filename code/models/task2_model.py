import torch.nn as nn
import torch

class CenterRegressor(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 100->50
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 50->25
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        # 64 * 25 * 25 -> 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*25*25, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feat(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        scale = x.new_tensor([self.W - 1, self.H - 1])
        return x * scale