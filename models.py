import torch
import torch.nn as nn
import math

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        
        # 1. 定义网络层
        # 第一层：输入通道为 1 (亮度 Y 通道)，输出通道为 64 [cite: 90, 91]
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 中间层：第 2 到第 19 层，全是 64 通道进，64 通道出 
        # 我们用 nn.ModuleList 来存放这些重复的层
        layers = []
        for _ in range(18):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        self.mid_layers = nn.Sequential(*layers)
        
        # 最后一层：将 64 通道映射回 1 个通道，预测残差 [cite: 91, 103]
        self.last_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        # 2. 权重初始化（He 初始化） [cite: 391, 392]
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 论文中使用的是 He 初始化 (He et al. [10]) 
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # x 是输入的 ILR 图像 [cite: 107]
        identity = x 
        
        # 前向传播经过 20 层权重层
        out = self.first_layer(x)
        out = self.mid_layers(out)
        residual = self.last_layer(out)
        
        # 核心：全局残差学习 y = f(x) + x [cite: 108, 131, 132]
        out = torch.add(residual, identity)
        
        return out