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

def dct_2d(x):
    """
    输入 x: [B, C, H, W]，输出 x_dct: [B, C, H, W]
    用法：先在 W 方向做 dct_1d，再在 H 方向做 dct_1d
    """
    B, C, H, W = x.shape

    # Step 1: 对每一行（宽度维）做 DCT —— 最后一个维度 W
    x = x.view(B * C * H, W)               # [B*C*H, W]
    x = dct_1d(x)                          # 对 W 做 DCT
    x = x.view(B, C, H, W)                 # 还原形状

    # Step 2: 对每一列（高度维）做 DCT —— 倒数第二个维度 H
    x = x.transpose(2, 3)                  # [B, C, W, H]
    x = x.contiguous().view(B * C * W, H)  # [B*C*W, H]
    x = dct_1d(x)                          # 对 H 做 DCT
    x = x.view(B, C, W, H).transpose(2, 3) # 还原成 [B, C, H, W]

    return x

class dct_channel_block_2d(nn.Module):
    def __init__(self, channel, height, width):
        super().__init__()
        self.height = height
        self.width = width

        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )
        # LayerNorm 作用于 (H, W) 展平后的特征长度
        self.dct_norm = nn.LayerNorm(height * width, eps=1e-6)


    def forward(self, x):
        b, c, h, w = x.size()  # 输入尺寸 [B, C, H, W]

        x_dct = dct_2d(x)  # DCT 变换

        x_dct_flat = x_dct.reshape(b, c, -1)  # [B, C, H*W]

        # 动态创建 LayerNorm（每次都新建，不依赖 self.height）
        x_norm = nn.LayerNorm(h * w, eps=1e-6).to(x.device)(x_dct_flat)  # [B, C, H*W]

        x_avg = x_norm.mean(dim=2)  # [B, C]

        weights = self.fc(x_avg).view(b, c, 1, 1)  # [B, C, 1, 1]

        return x * weights

