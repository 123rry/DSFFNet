import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same'
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv1(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class FusionModule(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, fusion_type='add'):
        """
        in_channels: 输入单个特征图的通道数
        mid_channels: 中间层通道数（默认同in_channels）
        out_channels: 输出通道数（默认同in_channels）
        fusion_type: 'add'或'concat'
        """
        super().__init__()
        self.fusion_type = fusion_type
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        
        # 计算融合后的输入通道数
        if fusion_type == 'add':
            fusion_in_channels = in_channels  # 两个特征图相加，通道不变
        elif fusion_type == 'concat':
            fusion_in_channels = in_channels * 2  # 拼接，通道翻倍
        else:
            raise ValueError("fusion_type must be 'add' or 'concat'")

        # 定义融合后的卷积层序列
        self.conv_blocks = nn.Sequential(
            Conv1(fusion_in_channels, mid_channels, k=3, s=1),
            Conv1(mid_channels, mid_channels, k=3, s=1),
            Conv1(mid_channels, out_channels, k=3, s=1, act=False),
        )
        
    def forward(self, inputs):
        """
        inputs: list or tuple with two tensors: [feature_cnn, feature_transformer]
        """
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 2, \
            "FusionModule expects input as a list or tuple of length 2"
        
        f1, f2 = inputs
        
        if self.fusion_type == 'add':
            x = f1 + f2
        else:  # concat
            x = torch.cat([f1, f2], dim=1)
        return self.conv_blocks(x)
