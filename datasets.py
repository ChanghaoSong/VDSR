import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import glob
import random

class TrainDataset(data.Dataset):
    """
    训练集：
    1. 扫描文件夹，预先计算好所有可能的 Patch 坐标 (Grid Crop)。
    2. 读取 RGB 图片。
    3. 实时裁剪 -> 随机增强 -> 下采样 -> 返回 (LR, HR)。
    """
    def __init__(self, image_dir, upscale_factor, patch_size=96):
        """
        :param patch_size: 这里指的是 HR 图像的 Patch 大小（Label的大小）。
                           注意：VDSR 原文用的 patch_size 41 是指 ILR 的大小。
                           现在我们输入 LR，建议 HR patch_size 设为 96 或 128 (必须是 upscale_factor 的倍数)。
        """
        super(TrainDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        
        # 1. 获取所有图片路径
        self.image_filenames = glob.glob(os.path.join(image_dir, "*.png")) + \
                               glob.glob(os.path.join(image_dir, "*.jpg"))
        
        # 2. 构建 Patch 索引表 (Mimic prepare.py logic)
        # 我们不存图片数据，只存 "去哪张图(path)的哪个位置(x,y)切图"
        self.patches = [] 
        self._build_patch_index()

    def _build_patch_index(self):
        print("正在扫描图片并构建 Patch 索引...")
        for img_path in self.image_filenames:
            # 只读尺寸，不读内容，速度很快
            with Image.open(img_path) as img:
                w, h = img.size
            
            # 计算可以切多少块 (参考 prepare.py 的逻辑，步长 = patch_size，无重叠)
            # 如果想让数据更多，可以减小 step (e.g., step = patch_size // 2)
            step = self.patch_size 
            
            for y in range(0, h - self.patch_size + 1, step):
                for x in range(0, w - self.patch_size + 1, step):
                    self.patches.append((img_path, x, y))
        
        print(f"索引构建完成：共找到 {len(self.image_filenames)} 张原图，生成了 {len(self.patches)} 个训练 Patch。")

    def __getitem__(self, index):
        # 1. 获取当前 Patch 的信息
        img_path, x, y = self.patches[index]
        
        # 2. 读取图片并转为 RGB
        full_img = Image.open(img_path).convert('RGB')
        
        # 3. 裁剪出 HR Patch
        hr_patch = full_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        # 4. 数据增强 (随机水平翻转 / 旋转)
        # 替代 prepare.py 中生成 8 份文件的做法，这里随机选一种变化，效果等效且省空间
        if random.random() < 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
        # 随机旋转 0, 90, 180, 270
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            hr_patch = hr_patch.rotate(90 * rot_k)

        # 5. 生成 LR Patch (输入)
        # 计算 LR 尺寸
        lr_w = self.patch_size // self.upscale_factor
        lr_h = self.patch_size // self.upscale_factor
        
        # 下采样 (Bicubic) -> 得到 LR
        # 注意：这里不再上采样回去了！
        lr_patch = hr_patch.resize((lr_w, lr_h), resample=Image.BICUBIC)

        # 6. 转 Tensor 并归一化 [0, 1]
        input_tensor = np.array(lr_patch).astype(np.float32) / 255.0
        label_tensor = np.array(hr_patch).astype(np.float32) / 255.0
        
        # HWC -> CHW
        input_tensor = input_tensor.transpose(2, 0, 1)
        label_tensor = label_tensor.transpose(2, 0, 1)

        return torch.from_numpy(input_tensor), torch.from_numpy(label_tensor)

    def __len__(self):
        return len(self.patches)

class TestDataset(data.Dataset):
    """
    测试集：
    1. 读取 RGB。
    2. 下采样得到 LR。
    3. 不再上采样，直接返回 LR 给网络。
    """
    def __init__(self, image_dir, upscale_factor):
        super(TestDataset, self).__init__()
        self.image_filenames = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        # 1. 加载 HR 图像并转为 RGB
        hr_img = Image.open(self.image_filenames[index]).convert('RGB')
        
        # 2. 预处理尺寸：确保能被缩放因子整除
        w, h = hr_img.size
        new_w = w - (w % self.upscale_factor)
        new_h = h - (h % self.upscale_factor)
        hr_img = hr_img.resize((new_w, new_h), resample=Image.BICUBIC)
        
        # 3. 下采样得到 LR (Input)
        lr_img = hr_img.resize((new_w // self.upscale_factor, new_h // self.upscale_factor), resample=Image.BICUBIC)

        # 4. 转为 Tensor
        input_tensor = np.array(lr_img).astype(np.float32) / 255.0
        label_tensor = np.array(hr_img).astype(np.float32) / 255.0
        
        # HWC -> CHW
        input_tensor = input_tensor.transpose(2, 0, 1)
        label_tensor = label_tensor.transpose(2, 0, 1)
        
        return torch.from_numpy(input_tensor), torch.from_numpy(label_tensor)

    def __len__(self):
        return len(self.image_filenames)