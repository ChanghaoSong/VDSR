import torch
import torch.utils.data as data
import h5py
import numpy as np
from PIL import Image
import os

class TrainDataset(data.Dataset):
    """
    训练集：专心读取预处理好的 .h5 文件
    """
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, index):
        # 训练数据已在 prepare.py 中完成了 Y 通道提取、缩放、切片和增强
        with h5py.File(self.h5_file, 'r') as f:
            input_patch = torch.from_numpy(f['input'][index]).float()
            label_patch = torch.from_numpy(f['label'][index]).float()
            return input_patch, label_patch

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['input'])

class TestDataset(data.Dataset):
    """
    测试集：读取指定文件夹下的所有 PNG 图片，并实时生成 ILR 图像
    """
    def __init__(self, image_dir, upscale_factor):
        super(TestDataset, self).__init__()
        # 获取目录下所有 png 文件的完整路径
        self.image_filenames = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.png')])
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        # 1. 加载 HR 图像并转为 YCbCr
        hr_img = Image.open(self.image_filenames[index]).convert('YCbCr')
        hr_y, _, _ = hr_img.split() # 只取 Y 通道
        
        # 2. 模拟下采样并 Bicubic 放大回原样 (生成 ILR)
        w, h = hr_y.size
        # 为了能整除缩放因子，通常需要微调尺寸
        new_w, new_h = w - (w % self.upscale_factor), h - (h % self.upscale_factor)
        hr_y = hr_y.resize((new_w, new_h), resample=Image.BICUBIC)
        
        # 下采样得到 LR
        lr_y = hr_y.resize((new_w // self.upscale_factor, new_h // self.upscale_factor), resample=Image.BICUBIC)
        # 重新放大得到 ILR (网络的输入)
        ilr_y = lr_y.resize((new_w, new_h), resample=Image.BICUBIC)

        # 3. 转为 Tensor (归一化到 [0, 1])
        input_tensor = np.array(ilr_y).astype(np.float32) / 255.0
        label_tensor = np.array(hr_y).astype(np.float32) / 255.0
        
        # PyTorch 要求输入维度为 [C, H, W]，此处 C=1
        return torch.from_numpy(input_tensor).unsqueeze(0), torch.from_numpy(label_tensor).unsqueeze(0)

    def __len__(self):
        return len(self.image_filenames)