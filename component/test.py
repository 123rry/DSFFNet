import torch
import torch.nn as nn
from ultralytics.nn.addmoudel.swimtransform import *

# 假设你已经定义了 SwinStage 和 SwinTransformerBlock

if __name__ == "__main__":
    B, C, H, W = 2, 64, 16, 16  # 64是patch_embed的输出通道，16x16是空间尺寸
    x = torch.randn(B, C, H, W)

    swin_stage = SwinStage(
        dim=C, c2=C, depth=2, num_heads=4, window_size=4,
        mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., norm_layer=nn.LayerNorm
    )

    out = swin_stage(x)
    print(out.shape)  # 应该是 [2, 64, 16, 16]
