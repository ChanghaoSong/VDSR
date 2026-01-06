import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VDSR(nn.Module):
    def __init__(self, upscale_factor=4):
        super(VDSR, self).__init__()
        self.upscale_factor = upscale_factor 
        
        # 1. 第一层：卷积层
        # 输入不再是 LR 小图，而是经过插值放大的 ILR 大图 (3通道 RGB)
        # 所以输入通道是 3，输出 64
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 2. 中间层：18 层 (64 -> 64)
        layers = []
        for _ in range(18):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        self.mid_layers = nn.Sequential(*layers)
        
        # 3. 最后一层：64 -> 3 (RGB 残差)
        self.last_layer = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化 [cite: 391]
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # x: 输入是 LR 小图 [B, 3, H_lr, W_lr]
        
        # --- 步骤 1: 刚性插值 (Bicubic Interpolation) ---
        # 这一步代替了之前的转置卷积，也代替了 datasets.py 里的预处理
        # 直接在 GPU 上把小图拉伸成大图 (ILR)
        # 这张图颜色是绝对正确的，只是模糊
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        
        # 此时 x 的形状变成了 [B, 3, H_hr, W_hr]
        identity = x # 保存这张模糊的大图，用于最后的残差连接 [cite: 131]
        
        # --- 步骤 2: 预测残差 ---
        out = self.first_layer(x)
        out = self.mid_layers(out)
        residual = self.last_layer(out)
        
        # --- 步骤 3: 全局残差相加 ---
        # Output = Bicubic_Image + Residual
        out = torch.add(residual, identity)
        
        return out