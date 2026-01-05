import os
import h5py
import numpy as np
from PIL import Image
import glob
#from utils import convert_rgb_to_y, normalize
def prepare_image2h5(image_dir, save_path, upscale_factor=4, patch_size=41):
    """
    将 Berkeley 200 张图片处理成训练用的 .h5 文件
    """
    # 存储所有切好的小块
    input_patches = []
    label_patches = []

    # 获取所有图片路径
    png_paths = glob.glob(os.path.join(image_dir, "*.png"))
    jpg_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_paths=jpg_paths+png_paths
    print(f"开始处理 {len(image_paths)} 张图片...")

    for path in image_paths:
        # 1. 加载图片并转为 YCbCr 提取 Y 通道
        hr_img = Image.open(path).convert('YCbCr')
        hr_y, _, _ = hr_img.split()
        
        # 2. 数据增强：旋转和翻转 (论文提到的)
        # 我们定义一个子流程来处理单张图的所有增强版本
        augmented_imgs = [
            hr_y,
            hr_y.transpose(Image.FLIP_LEFT_RIGHT), # 水平翻转
            hr_y.transpose(Image.ROTATE_90),
            hr_y.rotate(Image.ROTATE_180),
            hr_y.rotate(Image.ROTATE_270)
        ]

        for img in augmented_imgs:
            # 3. 预处理尺寸：确保能被缩放因子整除
            w, h = img.size
            img = img.crop((0, 0, w - w % upscale_factor, h - h % upscale_factor))
            w, h = img.size
            
            # 4. 生成 ILR (输入) 和 HR (标签)
            # HR 归一化
            hr_data = np.array(img).astype(np.float32) / 255.0
            
            # 下采样得到 LR -> 再上采样得到 ILR
            lr_img = img.resize((w // upscale_factor, h // upscale_factor), resample=Image.BICUBIC)
            ilr_img = lr_img.resize((w, h), resample=Image.BICUBIC)
            ilr_data = np.array(ilr_img).astype(np.float32) / 255.0

            # 5. 切片 (Patch Extraction) - 论文要求无重叠，步长 = patch_size
            for i in range(0, h - patch_size + 1, patch_size):
                for j in range(0, w - patch_size + 1, patch_size):
                    input_patch = ilr_data[i:i + patch_size, j:j + patch_size]
                    label_patch = hr_data[i:i + patch_size, j:j + patch_size]
                    
                    # 增加 C 维度 [H, W] -> [1, H, W]
                    input_patches.append(input_patch[np.newaxis, :, :])
                    label_patches.append(label_patch[np.newaxis, :, :])

    # 6. 转换为 numpy 数组并保存到 H5
    input_patches = np.array(input_patches)
    label_patches = np.array(label_patches)

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('input', data=input_patches)
        f.create_dataset('label', data=label_patches)

    print(f"处理完成！文件已保存至: {save_path}")
    print(f"总计提取了 {len(input_patches)} 个 patch。")

if __name__ == "__main__":
    # 使用示例
    prepare_image2h5(
        image_dir="/workspace/VDSR/train_data", 
        save_path="train_data.h5", 
        upscale_factor=4
    )