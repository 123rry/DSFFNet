# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class EnhancerBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()

#         # 模拟中值滤波 → 用 depthwise conv
#         self.median_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

#         # CLAHE替代（用可学习的对比度增强）
#         self.contrast = nn.Sequential(
#             nn.Conv2d(channels, channels, 1),
#             nn.Sigmoid()
#         )

#         # Laplacian kernel
#         laplacian_kernel = torch.tensor([
#             [0, -1, 0],
#             [-1, 4, -1],
#             [0, -1, 0]
#         ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#         self.laplacian = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

#         self.laplacian.weight.data = laplacian_kernel.repeat(channels, 1, 1, 1)
#         self.laplacian.weight.requires_grad = False  # 可改为True做学习

#     def forward(self, x):
#         # 1. “中值滤波”（近似）
#         x_smooth = self.median_conv(x)

#         # 2. 对比度增强（替代CLAHE）
#         x_contrast = x * self.contrast(x)

#         # 3. 高频增强（Laplacian）
#         lap = self.laplacian(x_contrast)

#         # 4. 融合（对应你 addWeighted）
#         out = 1.1 * x_contrast - 0.3 * lap

#         return outimport torch
import torch
import torch.nn as nn

class EnhancerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()

        # ❗不再写死 3 → c
        self.median_conv = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)

        self.contrast = nn.Sequential(
            nn.Conv2d(c, c, 1),
            nn.Sigmoid()
        )

        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.laplacian = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.laplacian.weight.data = laplacian_kernel.repeat(c, 1, 1, 1)
        self.laplacian.weight.requires_grad = False

    def forward(self, x):
        identity = x

        x_smooth = self.median_conv(x)
        x_contrast = x * self.contrast(x)
        lap = self.laplacian(x_contrast)
        x_sharp = x_contrast - 0.3 * lap

        return identity + 0.5 * x_smooth + 0.5 * x_sharp




# class EnhancerBlock(nn.Module):
#     def __init__(self, c):
#         super().__init__()

#         # 平滑
#         self.smooth = nn.AvgPool2d(3, stride=1, padding=1)

#         # 对比度（SE）
#         self.contrast = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c, c, 1),
#             nn.ReLU(),
#             nn.Conv2d(c, c, 1),
#             nn.Sigmoid()
#         )

#         # Laplacian
#         laplacian_kernel = torch.tensor([
#             [0, -1, 0],
#             [-1, 4, -1],
#             [0, -1, 0]
#         ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#         self.laplacian = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
#         self.laplacian.weight.data = laplacian_kernel.repeat(c, 1, 1, 1)
#         self.laplacian.weight.requires_grad = False

#         # 自适应融合
#         self.fuse = nn.Sequential(
#             nn.Conv2d(c * 3, c, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         identity = x

#         x_smooth = self.smooth(x)

#         x_contrast = x * self.contrast(x)

#         lap = self.laplacian(x_contrast)
#         x_sharp = x_contrast + 0.8 * lap

#         fusion = torch.cat([identity, x_smooth, x_sharp], dim=1)
#         w = self.fuse(fusion)

#         out = identity * (1 - w) + (x_smooth + x_sharp) * w

#         return out