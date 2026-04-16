import torch
import torch.nn as nn

class AFF(nn.Module):
    def __init__(self, c1, r=4):
        super().__init__()
        channels = c1
        inter_channels = max(1, channels // r)

        norm = lambda c: nn.GroupNorm(1, c)  # 1个组，相当于LayerNorm

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            norm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            norm(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            norm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            norm(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x
        xa = x1 + x2
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x1 * wei + 2 * x2 * (1 - wei)
        return xo
