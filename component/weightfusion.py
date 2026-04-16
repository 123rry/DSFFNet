import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1(nn.Module):
    # 标准 YOLO Conv + BN + SiLU
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class weightfusion(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, fusion_type='add', *args):

        """
        fusion_type: 'add', 'concat', 'weighted'
        """
        super().__init__()
        self.fusion_type = fusion_type
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        if fusion_type == 'add':
            fusion_in_channels = in_channels
        elif fusion_type == 'concat':
            fusion_in_channels = in_channels * 2
        elif fusion_type == 'weighted':
            fusion_in_channels = in_channels
            self.w = nn.Parameter(torch.tensor([0.5, 0.5]))  # learnable weights
        else:
            raise ValueError("fusion_type must be 'add', 'concat', or 'weighted'")

        # Conv after fusion
        self.conv_blocks = nn.Sequential(
            Conv1(fusion_in_channels, mid_channels),
            Conv1(mid_channels, mid_channels),
            Conv1(mid_channels, out_channels, act=False),
        )

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 2, \
            "FusionModule expects a list/tuple of two inputs"
        f1, f2 = inputs

        # 确保大小一致（如果不一致自动上采样 f2）
        if f1.shape[2:] != f2.shape[2:]:
            f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)

        if self.fusion_type == 'add':
            x = f1 + f2

        elif self.fusion_type == 'concat':
            x = torch.cat([f1, f2], dim=1)

        elif self.fusion_type == 'weighted':
            w = torch.softmax(self.w, dim=0)  # 保证 w1 + w2 = 1
            x = w[0] * f1 + w[1] * f2

        return self.conv_blocks(x)


# - [10, 18, FusionModule, [0, 128, 128, 'weighted']]  # 注意 args[0]=0 只是占位，实际会被推导替换