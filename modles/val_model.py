import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
# 损失函数定义
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

def combined_loss(pred, target):
    l1 = l1_loss(pred, target)
    mse = mse_loss(pred, target)
    return 0 * l1 + 1.0 * mse

def evaluate_model(model, val_loader, device='cpu'):
    model.eval()  # 设置模型为评估模式

    total_loss = 0
    total_l1_loss = 0
    total_mse_loss = 0

    l1_losses = []
    mse_losses = []
    combined_losses = []

    # 禁用梯度计算
    with torch.no_grad():
        # 进度条
        pbar = tqdm(total=len(val_loader), desc="Evaluating", unit="batch")

        for batch_idx, (neighbor_ligand_data, receptor, tf, tg, neighbor_distance_data, posi_samp_mat_data) in enumerate(val_loader):
            # 将数据移动到指定的设备
            neighbor_ligand_data, receptor, tf, tg, neighbor_distance_data, posi_samp_mat_data = neighbor_ligand_data.to(device), receptor.to(device), tf.to(device), tg.to(device), neighbor_distance_data.to(device), posi_samp_mat_data.to(device)

            # 调用模型的前向传播方法
            output = model(neighbor_ligand_data, receptor, tf, neighbor_distance_data)

            # 计算损失
            loss = combined_loss(output, tg)
            l1, mse = l1_loss(output, tg), mse_loss(output, tg)

            total_l1_loss += l1.item()
            total_mse_loss += mse.item()
            total_loss += loss.item()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(CombinedLoss=total_loss / (batch_idx + 1), L1Loss=total_l1_loss / (batch_idx + 1), MSELoss=total_mse_loss / (batch_idx + 1))

        # 计算并存储每个epoch的平均损失
        avg_l1_loss = total_l1_loss / len(val_loader)
        avg_mse_loss = total_mse_loss / len(val_loader)
        avg_combined_loss = total_loss / len(val_loader)

        l1_losses.append(avg_l1_loss)
        mse_losses.append(avg_mse_loss)
        combined_losses.append(avg_combined_loss)

        # 打印结果
        print(f"Evaluation Results - Combined Loss: {avg_combined_loss:.4f}, L1 Loss: {avg_l1_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}")

        # 绘制损失曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(l1_losses, label='L1 Loss')
        plt.title('L1 Loss per Batch')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(mse_losses, label='MSE Loss')
        plt.title('MSE Loss per Batch')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    return avg_combined_loss, avg_l1_loss, avg_mse_loss