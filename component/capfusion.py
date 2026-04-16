import torch
import torch.nn as nn
import torch.nn.functional as F



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=1, padding=dilation, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class cwFModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3x3 = DepthwiseSeparableConv(in_channels * 2, in_channels * 2, kernel_size=3)
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convout = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU6(inplace=True)


    def forward(self, x):
        x1, x2 = x
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.gap(x)
        x = self.convout(x)
        x = self.sigmoid(x)
        x = x1 * x + x2 * x
        x = self.relu(x)
        return x



# class cwFModule(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()

#         # ========== 关键1：分支对齐 ==========
#         self.norm1 = nn.BatchNorm2d(in_channels)
#         self.norm2 = nn.BatchNorm2d(in_channels)

#         # ========== 关键2：权重生成 ==========
#         self.weight_net = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 4, 2, kernel_size=1, bias=False)
#         )

#         # ========== 输出增强 ==========
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):

#         # ======================
#         # YOLO兼容
#         # ======================
#         if isinstance(x, (list, tuple)):
#             x1, x2 = x
#         else:
#             x1 = x
#             x2 = x

#         # ======================
#         # 1️⃣ 分布对齐（关键）
#         # ======================
#         x1 = self.norm1(x1)
#         x2 = self.norm2(x2)

#         # ======================
#         # 2️⃣ 差异建模
#         # ======================
#         diff = x1 - x2

#         # ======================
#         # 3️⃣ 生成竞争权重
#         # ======================
#         w = self.weight_net(diff)  # [B,2,H,W]
#         w = torch.softmax(w, dim=1)

#         w1 = w[:, 0:1, :, :]
#         w2 = w[:, 1:2, :, :]

#         # ======================
#         # 4️⃣ 融合
#         # ======================
#         out = x1 * w1 + x2 * w2

#         # ======================
#         # 5️⃣ 残差（稳）
#         # ======================
#         out = out + x1

#         # ======================
#         # 6️⃣ 输出增强
#         # ======================
#         out = self.out_conv(out)

#         return out





