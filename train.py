import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os

# 从我们写的其他文件中导入
from models import VDSR
from datasets import TrainDataset

# --- 第一部分：配置超参数 ---
# 论文中提到的核心参数
BATCH_SIZE = 64              
INITIAL_LR = 0.1            
MOMENTUM = 0.9              
WEIGHT_DECAY = 0.0001        
EPOCHS = 80                  
GRAD_CLIP_FACTOR = 0.4       # 用于梯度裁剪的阈值 theta

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # --- 第二部分：数据加载 (调用 datasets.py) ---
    # 我们将 91-image 和 Berkeley 的数据集合并 [cite: 160]
    #dataset_91 = TrainDataset("91-image_x4.h5")
    #dataset_berkeley = TrainDataset("berkeley_x4.h5")
    #full_dataset = ConcatDataset([dataset_91, dataset_berkeley])
    full_dataset=TrainDataset('train_data.h5')
    # DataLoader 负责批量读取、打乱和多线程并行加载
    train_loader = DataLoader(dataset=full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    # --- 第三部分：初始化模型与优化 (调用 models.py) ---
    model = VDSR().to(device)
    criterion = nn.MSELoss() # 损失函数：均方误差 [cite: 125, 126]
    
    # 优化器：带动量的随机梯度下降 [cite: 136, 137]
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # --- 第四部分：训练循环 ---
    print("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        # 1. 调整学习率：每 20 个 epoch 减小 10 倍 [cite: 394]
        current_lr = INITIAL_LR * (0.1 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.train() # 设置为训练模式
        epoch_loss = 0
        
        for iteration, batch in enumerate(train_loader):
            # batch[0] 是输入 ILR，batch[1] 是标签 HR
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # A. 前向传播
            preds = model(inputs)
            loss = criterion(preds, labels)

            # B. 反向传播
            optimizer.zero_grad() # 清空之前的梯度堆积
            loss.backward()

            # C. 梯度裁剪 (核心创新点) [cite: 13, 153]
            # 论文要求裁剪到 [-theta/lr, theta/lr]
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_FACTOR / current_lr)

            # D. 权重更新
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.6f}, LR: {current_lr}")

        # 2. 定期保存模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"vdsr_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()