import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import time
import os
import math

# 导入自定义模块
from models import VDSR
from datasets import TestDataset
# 注意：之前的 utils 可能只支持单通道 PSNR，这里我们直接内置一个 RGB PSNR 计算函数
# from utils import calc_psnr, denormalize 

def calc_rgb_psnr(img1, img2):
    """
    计算 RGB 图像的 PSNR
    img1, img2: [B, 3, H, W] or [3, H, W], range [0, 1]
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * torch.log10(1. / mse)

def tensor_to_img(tensor):
    """
    Tensor [C, H, W] (0-1) -> PIL Image (0-255)
    """
    array = tensor.squeeze().cpu().detach().numpy()
    # 限制范围并转为 uint8
    array = np.clip(array, 0, 1) * 255.0
    array = array.astype(np.uint8)
    # CHW -> HWC
    array = array.transpose(1, 2, 0)
    return Image.fromarray(array)

def run_test(test_dir, results_dir, model_path, upscale_factor=4, visualize=False):
    """
    RGB 版 VDSR 测试主函数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 实例化模型 (修改点：必须传入 upscale_factor)
    model = VDSR(upscale_factor=upscale_factor).to(device)
    
    print(f"加载模型权重: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    # 2. 加载测试集 (TestDataset 已经改为返回 LR 和 HR RGB)
    test_set = TestDataset(test_dir, upscale_factor)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    os.makedirs(results_dir, exist_ok=True)

    total_psnr_bicubic = 0
    total_psnr_vdsr = 0
    total_time = 0

    print(f"{'Image':<30} | {'Bicubic (RGB)':<15} | {'VDSR (RGB)':<15} | {'Time':<10}")
    print("-" * 80)

    with torch.no_grad():
        for i, (lr_tensor, hr_tensor) in enumerate(test_loader):
            # 获取文件名
            img_path = test_set.image_filenames[i]
            img_name = os.path.basename(img_path)
            
            lr_tensor = lr_tensor.to(device) # [1, 3, h, w]
            hr_tensor = hr_tensor.to(device) # [1, 3, H, W]

            # 3. 模拟 Bicubic 基准结果 (为了对比 PSNR)
            # 因为现在输入是小图，我们需要手动上采样来计算 Bicubic 的分数
            bicubic_tensor = F.interpolate(lr_tensor, scale_factor=upscale_factor, mode='bicubic', align_corners=False)
            
            # 截断到 [0, 1] 范围，防止插值溢出
            bicubic_tensor = torch.clamp(bicubic_tensor, 0, 1)

            # 4. 模型推理
            start_time = time.time()
            # 模型内部会先做转置卷积上采样，再加残差
            vdsr_tensor = model(lr_tensor)
            # 截断输出，保证 RGB 合法
            vdsr_tensor = torch.clamp(vdsr_tensor, 0, 1)
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # 5. 计算 RGB PSNR (领导的要求)
            psnr_bicubic = calc_rgb_psnr(bicubic_tensor, hr_tensor)
            psnr_vdsr = calc_rgb_psnr(vdsr_tensor, hr_tensor)
            
            total_psnr_bicubic += psnr_bicubic
            total_psnr_vdsr += psnr_vdsr

            print(f"{img_name:<30} | {psnr_bicubic:14.2f} dB | {psnr_vdsr:14.2f} dB | {elapsed_time*1000:6.1f} ms")

            # 6. 保存图片 (直接保存 RGB)
            vdsr_img = tensor_to_img(vdsr_tensor)

            if visualize:
                # 拼接: Bicubic | VDSR | GT
                bicubic_img = tensor_to_img(bicubic_tensor)
                gt_img = tensor_to_img(hr_tensor)
                
                # 创建画布
                w, h = gt_img.size
                canvas = Image.new('RGB', (w * 3, h))
                canvas.paste(bicubic_img, (0, 0))
                canvas.paste(vdsr_img, (w, 0))
                canvas.paste(gt_img, (w * 2, 0))
                
                canvas.save(os.path.join(results_dir, f"compare_{img_name}"))
            else:
                vdsr_img.save(os.path.join(results_dir, f"sr_{img_name}"))

    avg_psnr_vdsr = total_psnr_vdsr / len(test_loader)
    avg_psnr_bicubic = total_psnr_bicubic / len(test_loader)
    
    print("-" * 80)
    print(f"平均 PSNR (Bicubic): {avg_psnr_bicubic:.2f} dB")
    print(f"平均 PSNR (VDSR):    {avg_psnr_vdsr:.2f} dB")
    print(f"平均推理时间: {(total_time / len(test_loader))*1000:.2f} ms")

if __name__ == "__main__":
    # 配置
    TEST_DIR = "/data/output/garden/GT/HR_Inputs/view/"
    RESULTS_DIR = '/data/output/garden/VDSR_Output/'
    
    # 这里的 upscale_factor 必须和训练时模型的一致！
    UPSCALE_FACTOR = 4 
    
    # 这里的路径记得改成你生成的 RGB 模型路径
    MODEL_PATH = f"./vdsr_x4_rgb.pth" 
    
    if os.path.exists(MODEL_PATH):
        run_test(TEST_DIR, RESULTS_DIR, MODEL_PATH, upscale_factor=UPSCALE_FACTOR, visualize=False)
    else:
        print(f"找不到模型文件: {MODEL_PATH}")