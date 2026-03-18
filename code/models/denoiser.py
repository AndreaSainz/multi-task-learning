import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.block(x)
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # por seguridad, si hay desajuste de tamaño
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class ResUNetDenoiser(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        self.in_block = ResidualBlock(in_channels, base_channels)

        self.down1 = DownBlock(base_channels, base_channels * 2)      # 28 -> 14
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 14 -> 7

        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4)

        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)  # 7 -> 14
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels)      # 14 -> 28

        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.in_block(x)          # [B, 32, 28, 28]
        s1, x1 = self.down1(x0)        # s1: [B, 64, 28, 28], x1: [B, 64, 14, 14]
        s2, x2 = self.down2(x1)        # s2: [B,128,14,14], x2: [B,128, 7, 7]

        x3 = self.bottleneck(x2)       # [B,128, 7, 7]

        x4 = self.up1(x3, s2)          # [B, 64, 14, 14]
        x5 = self.up2(x4, s1)          # [B, 32, 28, 28]

        residual = self.out_conv(x5)

        # residual learning: predice la corrección, no la imagen completa
        out = x + residual
        return out