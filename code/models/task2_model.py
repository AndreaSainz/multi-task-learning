import torch.nn as nn
import torch.nn.functional as F

class CenterRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.down = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 100->50
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 50->25
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )

        # upsample de 25->100
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 25->50
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 50->100
            nn.Conv2d(16, 1, 1)  # logits del heatmap (B,1,100,100)
        )

    def forward(self, x):
        x = self.down(x)
        heat_logits = self.up(x)
        return heat_logits