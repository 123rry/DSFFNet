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
    B, C, H, W = x.shape

    # Step 1: DCT on width
    x = x.reshape(B * C * H, W)  # ⚠️ 改为 reshape
    x = dct_1d(x)
    x = x.reshape(B, C, H, W)

    # Step 2: DCT on height
    x = x.transpose(2, 3)
    x = x.contiguous().reshape(B * C * W, H)  # ⚠️ reshape
    x = dct_1d(x)
    x = x.reshape(B, C, W, H).transpose(2, 3)  # ⚠️ reshape

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


class DCTBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dct = dct_channel_block_2d(c, height=32, width=32)  # height/width 是占位符，实际 forward 时替换

    def forward(self, x):
        b, c, h, w = x.shape
        # 动态设置 height/width
        self.dct.height = h
        self.dct.width = w
        return self.dct(x)


class DCTBottleneck(nn.Module):
    """DCT-enhanced Bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        """
        用 DCTBlock 替代原来的两个卷积。
        - c1: 输入通道数
        - c2: 输出通道数
        - e: 中间隐藏层通道比例
        """
        super().__init__()
        c_ = int(c2 * e)
        self.dct1 = DCTBlock(c1)  # 第一个DCT block：输入 -> hidden
        self.conv1x1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1)  # 保证形状映射到c_
        self.dct2 = DCTBlock(c_)  # 第二个DCT block：hidden -> 输出
        self.conv1x1_out = nn.Conv2d(c_, c2, kernel_size=1, stride=1)  # 映射回c2
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x1 = self.dct1(x)
        x1 = self.conv1x1(x1)
        x2 = self.dct2(x1)
        x2 = self.conv1x1_out(x2)
        return x + x2 if self.add else x2


class DCTcatch(attention):
    """C2f with DCT-based Bottlenecks."""

    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        """
        - c1: 输入通道数
        - c2: 输出通道数
        - n: 堆叠的 DCTBottleneck 个数
        - shortcut: 是否使用残差连接
        """
        super(attention, self).__init__()
        self.triplet_attention = TripletAttention()
        # self.triplet_attention = ECA(channel=c1)
        # self.triplet_attention = CBAM(c1=c1)
        # self.triplet_attention = CoordAtt(inp=c1)

        c_ = int(c2 * e)
        self.cv1 = DPConv1(c1, 2 * c_, 1, 1)  # 输入通道映射
        self.cv2 = DPConv1((2 + n) * c_, c2, 1)  # 输出通道映射
        self.m = nn.ModuleList([DCTBottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)])