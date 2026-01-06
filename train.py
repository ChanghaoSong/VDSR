import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# 从我们写的其他文件中导入
from models import VDSR
from datasets import TrainDataset

# --- 第一部分：配置超参数 ---
# 论文中提到的核心参数
BATCH_SIZE = 64              
INITIAL_LR = 1e-4            
MOMENTUM = 0.9              
WEIGHT_DECAY = 0.0001        
EPOCHS = 80                  
GRAD_CLIP_FACTOR = 0.4       # 用于梯度裁剪的阈值 theta

# --- 新增/修改的参数 ---
UPSCALE_FACTOR = 4           # 放大倍数 (必须与 models.py 和 datasets.py 匹配)
PATCH_SIZE = 96              # 训练时 HR Patch 的大小 (建议是 UPSCALE_FACTOR 的倍数)
TRAIN_DIR = "train_data"     # 训练图片文件夹路径 (请改为你的实际路径，如 /workspace/VDSR/train_data)

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # --- 第二部分：数据加载 (调用 datasets.py) ---
    print(f"正在加载训练数据 (Grid Crop) 从: {TRAIN_DIR} ...")
    
    # 修改点 1: 实例化 TrainDataset (不再读取 H5)
    # 这里会自动扫描文件夹并构建 Patch 索引
    full_dataset = TrainDataset(image_dir=TRAIN_DIR, upscale_factor=UPSCALE_FACTOR, patch_size=PATCH_SIZE)
    
    # DataLoader 负责批量读取、打乱和多线程并行加载
    # num_workers=8 比较稳妥，太高在某些环境下可能会卡顿
    train_loader = DataLoader(dataset=full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # --- 第三部分：初始化模型与优化 (调用 models.py) ---
    # 修改点 2: 实例化模型时传入 upscale_factor
    # 这样模型内部才知道转置卷积该设置多大的 stride
    model = VDSR(upscale_factor=UPSCALE_FACTOR).to(device)
    
    # [cite_start]损失函数：均方误差 [cite: 125, 126]
    # 注意：现在的输入输出都是 RGB 三通道，MSELoss 会计算 (R+G+B) 所有像素的平均误差
    criterion = nn.MSELoss() 
    
    # [cite_start]优化器：带动量的随机梯度下降 [cite: 136, 137]
    #optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    # --- 第四部分：训练循环 ---
    print(f"开始训练 VDSR (RGB版, x{UPSCALE_FACTOR})...")
    
    for epoch in range(1, EPOCHS + 1):

        model.train() # 设置为训练模式
        epoch_loss = 0
        
        for iteration, batch in enumerate(train_loader):
            # batch[0] 是输入 LR (B, 3, H/s, W/s)
            # batch[1] 是标签 HR (B, 3, H, W)
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # A. 前向传播 (LR -> Upsample -> BaseHR + Residual -> Output)
            preds = model(inputs)
            
            # 计算 Loss
            loss = criterion(preds, labels)

            # B. 反向传播
            optimizer.zero_grad() 
            loss.backward()

            # [cite_start]C. 梯度裁剪 (核心创新点) [cite: 13, 153]
            # 论文要求裁剪到 [-theta/lr, theta/lr]
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_FACTOR )

            # D. 权重更新
            optimizer.step()

            epoch_loss += loss.item()

        # 打印每个 Epoch 的平均 Loss
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.6f}, LR: {current_lr:.1e}")

        # 2. 定期保存模型
        if epoch % 10 == 0:
            # 文件名加上 scale 信息，避免混淆
            save_name = f"vdsr_x{UPSCALE_FACTOR}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"模型已保存: {save_name}")

if __name__ == "__main__":
    if not os.path.exists(TRAIN_DIR):
        print(f"错误: 找不到训练文件夹 '{TRAIN_DIR}'，请修改代码中的 TRAIN_DIR 变量。")
    else:
        train()