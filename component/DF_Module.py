import torch
import torch.nn as nn
from .batchnorm import SynchronizedBatchNorm2d

bn_mom = 0.0003

class DenseCatAdd(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, padding=0),
            SynchronizedBatchNorm2d(c2, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class DenseCatDiff(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, padding=0),
            SynchronizedBatchNorm2d(c2, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)

        return self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))


class DF_Module(nn.Module):
    def __init__(self, c1, c2, reduction=True):
        super().__init__()
        if reduction:
            self.reduction = nn.Sequential(
                nn.Conv2d(c1, c1 // 2, 1, 1, padding=0),
                SynchronizedBatchNorm2d(c1 // 2, momentum=bn_mom),
                nn.ReLU(inplace=True)
            )
            c1 = c1 // 2
        else:
            self.reduction = nn.Identity()

        # 你原来用的三个卷积的结构，改成concat后用两个卷积处理
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 * 2, 3, 1, padding=1, bias=False),
            SynchronizedBatchNorm2d(c1 * 2, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(c1, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convout = nn.Conv2d(c1, c1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1, x2 = x  # 输入是元组，拆成两个张量
        if not isinstance(self.reduction, nn.Identity):
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)

        x_cat = torch.cat([x1, x2], dim=1)  # 通道维concat
        x = self.conv3x3(x_cat)
        x = self.conv1x1(x)
        x_gap = self.gap(x)
        x_att = self.convout(x_gap)
        x_att = self.sigmoid(x_att)

        out = x1 * x_att + x2 * x_att
        out = self.relu(out)

        return out
