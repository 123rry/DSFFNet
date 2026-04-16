import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, fusion_type='add'):
        """
        融合两个特征图
        Args:
            in_channels (int): 每个输入特征图的通道数（两个输入需相同）
            fusion_type (str): 'add' 或 'concat'
        """
        super().__init__()
        assert fusion_type in ['add', 'concat'], "fusion_type 必须是 'add' 或 'concat'"
        self.fusion_type = fusion_type
        self.in_channels = in_channels

        if fusion_type == 'concat':
            # 拼接后通道数翻倍，用1x1卷积压缩回 in_channels
            self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(in_channels)
            self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        """
        Args:
            inputs: 长度为2的列表，包含两个Tensor (B, C, H, W)
        Returns:
            Tensor: 融合后的Tensor (B, C, H, W)
        """
        x1, x2 = inputs
        assert x1.shape == x2.shape, "两个输入特征图的形状必须一致"
        if self.fusion_type == 'add':
            return x1 + x2
        else:
            x = torch.cat([x1, x2], dim=1)
            x = self.conv1x1(x)
            x = self.bn(x)
            x = self.act(x)
            return x
