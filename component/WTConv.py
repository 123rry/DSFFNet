import pywt
import pywt.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv(nn.Module):
    def __init__(self, c1, c2, k=5, s=1, p=None, g=1, act=True, wt_levels=1, wt_type='db1'):
        super().__init__()
        self.stride = s
        self.wt_levels = wt_levels
        self.in_channels = c1
        self.out_channels = c2

        wt_f, iwt_f = create_wavelet_filter(wt_type, c1, c1, torch.float)
        self.register_buffer('wt_filter', wt_f)
        self.register_buffer('iwt_filter', iwt_f)


        self.base_conv = nn.Conv2d(c1, c1, k, padding=k // 2, stride=1, groups=c1, bias=False)
        self.base_scale = _ScaleModule([1, c1, 1, 1])

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(c1 * 4, c1 * 4, k, padding=k // 2, stride=1, groups=c1 * 4, bias=False)
            for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, c1 * 4, 1, 1], init_scale=0.1) for _ in range(wt_levels)
        ])

        self.channel_adapter = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(c2)

        self.do_stride = nn.AvgPool2d(kernel_size=1, stride=s) if s > 1 else None
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # 主分支和小波分支
        x_main = self.base_scale(self.base_conv(x))
        x_tag = next_x_ll

        # 通道适配
        x_main = self.channel_adapter(x_main)
        x_tag = self.channel_adapter(x_tag)

        x = x_main + x_tag
        x = self.out_bn(x)
        x = self.act(x)

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x



# if __name__ == '__main__':
#     x = torch.randn(2, 64, 128, 128)

#     model = WTConv(64, 128)
#     output = model(x)
#     print(output.shape)


# class Bottleneck_WT(nn.Module):
#     def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=0.5, wt_type='db1'):
#         super().__init__()
#         c_ = int(c2 * e)
#         k1 = k[0][0] if isinstance(k, tuple) else k  # 取第一个核大小

#         self.cv1 = WTConv(c1, c_, k=k1, s=1, act=True, wt_type=wt_type)
#         self.cv2 = WTConv(c_, c2, k=k1, s=1, act=True, wt_type=wt_type)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         y = self.cv2(self.cv1(x))
#         return x + y if self.add else y


# class C2f_WT(nn.Module):
#     """基于WTConv替换的C2f模块"""

#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, wt_type='db1', k=((3, 3), (3, 3))):
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = WTConv(c1, 2 * self.c, k=k[0][0], s=1, act=True, wt_type=wt_type)
#         self.cv2 = WTConv((2 + n) * self.c, c2, k=1, s=1, act=True, wt_type=wt_type)
#         self.m = nn.ModuleList(
#             Bottleneck_WT(self.c, self.c, shortcut, g, k=k, e=1.0, wt_type=wt_type) for _ in range(n)
#         )

#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
