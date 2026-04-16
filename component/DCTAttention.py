import torch
import torch.nn as nn
import numpy as np


def dct_1d(x):
    """
    计算 DCT-II（离散余弦变换）
    输入 x: [B, L]，输出 [B, L]
    """
    N = x.size(-1)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v, dim=-1)
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i
    return 2 * V


class DCTAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1)  # [B, C, H*W]

        # 对每个通道做 DCT（批量计算）
        x_dct = dct_1d(x_flat.mean(dim=-1))  # [B, C]

        weights = self.fc(x_dct)  # [B, C]
        weights = weights.view(b, c, 1, 1)  # 广播回原 shape

        return x * weights  # 通道注意力加权
