import torch.nn as nn
import torch.nn.functional as F

class CenterRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # por ejemplo 2 coords (x,y) del centro
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)