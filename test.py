import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import time
import os

# 导入自定义模块
from models import VDSR
from datasets import TestDataset
from utils import calc_psnr, convert_y_to_rgb, denormalize

def run_test(test_dir, results_dir,model_path, upscale_factor=4, visualize=False):
    """
    VDSR 测试主函数
    :param visualize: 是否生成横向拼接对比图 (Bicubic | VDSR | GT)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 优雅地实例化模型并加载权重
    model = VDSR().to(device)
    # VDSR 使用 20 层权重层 [cite: 10] 并预测残差细节 [cite: 92, 103]
    model.load_state_dict(torch.load(model_path))
    model.eval() 

    # 2. 优雅地使用 datasets.py 中的 TestDataset
    # TestDataset 内部已经处理了 RGB->YCbCr 和下采样逻辑
    test_set = TestDataset(test_dir, upscale_factor)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    #results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    total_psnr_bicubic = 0
    total_psnr_vdsr = 0
    total_time = 0

    print(f"{'Image':<20} | {'Bicubic PSNR':<15} | {'VDSR PSNR':<15} | {'Inference Time':<15}")
    print("-" * 75)

    with torch.no_grad(): # 推理不计算梯度，节省 4090 显存
        for i, (input_tensor, label_tensor) in enumerate(test_loader):
            # 获取图片名称（假设 TestDataset 返回了路径，若没返回可按索引排序）
            img_name = test_set.image_filenames[i].split("/")[-1]
            
            input_tensor = input_tensor.to(device)
            label_tensor = label_tensor.to(device)

            # 3. 推理并统计耗时
            start_time = time.time()
            # VDSR 将插值后的低分辨率图 (ILR) 加回预测残差得到输出 [cite: 108]
            output_tensor = model(input_tensor)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # 4. 计算 PSNR (基于 Y 通道 [cite: 38, 400])
            psnr_bicubic = calc_psnr(input_tensor, label_tensor)
            psnr_vdsr = calc_psnr(output_tensor, label_tensor)
            total_psnr_bicubic += psnr_bicubic
            total_psnr_vdsr += psnr_vdsr

            print(f"{img_name:<20} | {psnr_bicubic:14.2f} dB | {psnr_vdsr:14.2f} dB | {elapsed_time*1000:13.2f} ms")

            # 5. 图像后处理与保存
            # 获取原始的 Cb, Cr 分量（为了还原彩色图）
            # 注意：此处为演示简洁，假设 TestDataset 重新加载了颜色分量
            hr_img = Image.open(test_set.image_filenames[i]).convert('YCbCr')
            _, cb, cr = hr_img.split()
            w, h = hr_img.size
            # 缩放对齐
            cb = cb.resize((w - w % upscale_factor, h - h % upscale_factor), resample=Image.BICUBIC)
            cr = cr.resize((w - w % upscale_factor, h - h % upscale_factor), resample=Image.BICUBIC)

            # 将 Tensor 转回 Y 图像数据
            vdsr_y = denormalize(output_tensor.squeeze().cpu().numpy())
            vdsr_res = convert_y_to_rgb(vdsr_y, cb, cr)

            if visualize:
                # 拼接对比图: Bicubic | VDSR | GT
                bicubic_y = denormalize(input_tensor.squeeze().cpu().numpy())
                bicubic_res = convert_y_to_rgb(bicubic_y, cb, cr)
                gt_res = hr_img.convert('RGB').crop((0, 0, w - w % upscale_factor, h - h % upscale_factor))
                
                # 横向拼接
                canvas = Image.new('RGB', (gt_res.width * 3, gt_res.height))
                canvas.paste(bicubic_res, (0, 0))
                canvas.paste(vdsr_res, (gt_res.width, 0))
                canvas.paste(gt_res, (gt_res.width * 2, 0))
                canvas.save(os.path.join(results_dir, f"compare_{img_name}"))
            else:
                # 仅保存超分结果
                vdsr_res.save(os.path.join(results_dir, f"sr_{img_name}"))

    avg_psnr_vdsr = total_psnr_vdsr / len(test_loader)
    print("-" * 75)
    print(f"平均 PSNR (VDSR): {avg_psnr_vdsr:.2f} dB")
    print(f"平均推理时间: {(total_time / len(test_loader))*1000:.2f} ms")

if __name__ == "__main__":
    # 配置你的测试参数
    TEST_DIR = "/data/output/garden/GT/HR_Inputs/view/" # 测试集路径
    RESULTS_DIR='/data/output/garden/VDSR_Output/'
    MODEL_PATH = "./checkpoints/vdsr_epoch_80.pth"
    
    run_test(TEST_DIR, RESULTS_DIR,MODEL_PATH, upscale_factor=4, visualize=False)