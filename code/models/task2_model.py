import torch.nn as nn
import torch

class CenterRegressor(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 100->50
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 50->25
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),         # -> (B,64,1,1)
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        x = self.feat(x).flatten(1)     # (B,64)
        x = self.head(x)                # (B,2)   libre
        x = torch.sigmoid(x) 
        scale = x.new_tensor([self.W - 1, self.H - 1])  # cx in [0,W-1], cy in [0,H-1]
        return x * scale