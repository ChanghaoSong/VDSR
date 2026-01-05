import numpy as np
import torch
import math

def calc_psnr(img1, img2):
    """
    计算峰值信噪比 (PSNR)
    img1, img2: [0, 1] 范围的张量或数组
    """
    # 论文中使用均方误差 (MSE) 作为优化目标 
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    
    # 标准 PSNR 公式 [cite: 191]
    return 10 * math.log10(1.0 / mse.item())

def convert_rgb_to_y(img):
    """
    将 RGB 图像转换为 YCbCr 空间并只返回 Y 通道
    img: PIL Image 或 numpy 数组
    """
    if type(img) == np.ndarray:
        # 标准转换公式 (BT.601)
        y = 16.0 + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.0
        return y.astype(np.float32)
    else:
        # PIL 直接转换
        ycbcr = img.convert('YCbCr')
        y, cb, cr = ycbcr.split()
        return np.array(y).astype(np.float32)

def convert_y_to_rgb(y, cb, cr):
    """
    将预测出的 Y 通道与插值后的 Cb, Cr 重新组合回 RGB
    """
    # 确保 Y 值的范围在 [0, 255]
    y = np.array(y).clip(0, 255).astype(np.uint8)
    cb = np.array(cb).clip(0, 255).astype(np.uint8)
    cr = np.array(cr).clip(0, 255).astype(np.uint8)
    
    from PIL import Image
    y_img = Image.fromarray(y, mode='L')
    cb_img = Image.fromarray(cb, mode='L')
    cr_img = Image.fromarray(cr, mode='L')
    
    return Image.merge('YCbCr', [y_img, cb_img, cr_img]).convert('RGB')

def normalize(img):
    """将 [0, 255] 的像素归一化到 [0, 1] """
    return img / 255.0

def denormalize(img):
    """将 [0, 1] 的像素映射回 [0, 255]"""
    return img * 255.0