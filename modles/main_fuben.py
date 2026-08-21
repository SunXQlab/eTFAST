import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

class TGDataset(Dataset):
    def __init__(self, neighbor_ligand, Rec_df, TF_df, TG_df, neighbor_distance, receiver_distance, batch_size, split_ratio=0.8):
        self.neighbor_ligand = neighbor_ligand
        self.Rec_df = Rec_df
        self.TF_df = TF_df
        self.TG_df = TG_df
        self.neighbor_distance = neighbor_distance
        self.receiver_distance = receiver_distance
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def __len__(self):
        return len(self.Rec_df)

    def __getitem__(self, idx):
        # 提取每个样本的邻居特征
        neighbor_ligand = torch.tensor(self.neighbor_ligand[idx], dtype=torch.float32)
        # 提取每个样本的受体特征
        receptor = torch.tensor(self.Rec_df.iloc[idx].values, dtype=torch.float32)
        # 提取每个样本的TF特征
        tf = torch.tensor(self.TF_df.iloc[idx].values, dtype=torch.float32)
        # 提取每个样本的TG特征
        tg = torch.tensor(self.TG_df.iloc[idx].values, dtype=torch.float32)
        # 提取每个样本的邻居距离
        neighbor_distance = torch.tensor(self.neighbor_distance[idx], dtype=torch.float32)
        # 提取每个样本的正样本矩阵
        receiver_distance = torch.tensor(self.receiver_distance[idx], dtype=torch.float32)

        return neighbor_ligand, receptor, tf, tg, neighbor_distance, receiver_distance,idx

    def split(self, train_size, val_size):
        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        return train_dataset, val_dataset

class TGPredictionModel(nn.Module):
    def __init__(self, lig_rec_dim, tf_dim, hidden_dim_1, hidden_dim_2, tg_dim):
        super(TGPredictionModel, self).__init__()

        # 配体-受体对的参数B，初始化在0.2到0.8之间
        self.B = nn.Parameter(torch.rand(lig_rec_dim) * (0.8 - 0.2) + 0.2)

        # 生成TF信号（TFsigal），由配体和受体的交互产生
        self.fc_tfsigal_1 = nn.Linear(lig_rec_dim, hidden_dim_1)  # 隐藏层1
        self.fc_tfsigal_2 = nn.Linear(hidden_dim_1, tf_dim)  # tf信号的线性变换

        # 计算TF矩阵
        self.fc_tfsigal = nn.Linear(tf_dim, tf_dim)  # 对TF信号进行线性变换
        self.fc_tfexpr = nn.Linear(tf_dim, tf_dim)  # TF表达式的线性变换

        # 最终的输出层，生成TG矩阵
        self.fc_tfact = nn.Linear(tf_dim, hidden_dim_2)  # 隐藏层，用于转换TF信号
        self.fc_tg = nn.Linear(hidden_dim_2, tg_dim)  # 输出TG矩阵

    def forward(self, neighbor_ligand, receptor, tf, neighbor_distance):
        # Step 2: 计算配体-受体对的LR得分
        expanded_receptor = receptor.unsqueeze(-2)  # 扩展受体矩阵，使其维度为 batch受体结点* 1 * lr
        distance_factor = 1 / neighbor_distance  # 距离的倒数 维度为 batch受体结点 * 邻居配体结点
        expand_distance_factor = distance_factor.unsqueeze(-1)  # 扩展距离的倒数，使其维度为 batch受体结点 * 邻居配体结点 * 1
        exp_factor = (1 - neighbor_distance).unsqueeze(-1)  # 距离的倒数 维度为 batch受体结点 * 邻居配体结点 * 1
        exp_factor = torch.exp(exp_factor * self.B.unsqueeze(0).unsqueeze(0))  # e^((1-distance)*B)

        # 计算LR得分，注意维度匹配和广播
        #print('LR_score1 :', expanded_receptor.shape, neighbor_ligand.shape, expand_distance_factor.shape, exp_factor.shape)
        LR_score = expanded_receptor * neighbor_ligand * expand_distance_factor * exp_factor
        LR_score = torch.relu(LR_score.sum(dim=1))  # 对邻居配体结点求和，得到一个 batch * 受体结点 的矩阵
        #print('LR_score :',LR_score.shape)
        # Step 3: 生成 TF 信号（TFsigal）
        tf_sigal = torch.relu(self.fc_tfsigal_1(LR_score))  # 转到隐藏层1
        #print('tf_sigal1 :',tf_sigal.shape)
        tf_sigal = torch.relu(self.fc_tfsigal_2(tf_sigal))  # 变成tf信号
        #print('tf_sigal2 :',tf_sigal.shape)
        # Step 4: 对TF矩阵进行进一步处理
        tf_sigal = self.fc_tfsigal(tf_sigal)  # 对TF信号进行线性变换
        tf_expr = self.fc_tfexpr(tf)  # 对输入的TF矩阵进行线性变换

        # Step 5: 结合TF信号生成最终的TF矩阵
        tf_act = torch.relu(tf_sigal * tf_expr)  # TF矩阵由TFsigal和TFexpr的逐元素相乘得到

        # Step 6: 计算TG矩阵并输出
        tg_output = torch.relu(self.fc_tfact(tf_act))  # TF信号通过隐藏层得到TG矩阵
        tg_output = torch.relu(self.fc_tg(tg_output))  # 最后的线性变换得到最终的TG矩阵

        return tf_act, tg_output  # 返回TG矩阵作为最终输出
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, num_positive=30, num_negative=30,posi_dist=30,neg_dist=100):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.posi_dist = posi_dist
        self.neg_dist = neg_dist

    def forward(self, tf_act, receiver_distance):
        # 计算tf_act的相似度矩阵
        sim_matrix = F.cosine_similarity(tf_act.unsqueeze(1), tf_act.unsqueeze(0), dim=2)
        
        # 将对角线元素设置为负无穷，避免自己与自己相似
        receiver_distance.fill_diagonal_(-float('inf'))
        
        # 为每个受体选择正样本和负样本
        positive_indices = torch.topk(-receiver_distance, k=self.num_positive, dim=1)[1]
        negative_indices = torch.topk(receiver_distance, k=self.num_negative, dim=1)[1]
        
        
        # 计算正样本和负样本的相似度
        int_sim = sim_matrix
        int_sim[int_sim > self.posi_dist] = 0
        print(int_sim[0:5,0:5])
        #pos_sim = int_sim.gather(1, positive_indices).squeeze(1)
        pos_sim = sim_matrix.gather(1, positive_indices).squeeze(1)
        int_sim = sim_matrix
        int_sim[int_sim < self.neg_dist] = 0
        neg_sim = sim.gather(1, negative_indices).squeeze(1)
        #neg_sim = int_sim.gather(1, negative_indices).squeeze(1)
        
        # 计算正样本的损失
        pos_loss = -torch.log(torch.sum(torch.exp(pos_sim / self.temperature), dim=1) / 
                              (torch.sum(torch.exp(neg_sim / self.temperature), dim=1) + torch.sum(torch.exp(pos_sim / self.temperature), dim=1)))
        
        # 计算平均损失
        loss = pos_loss.mean()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, lambda_mre=1, lambda_l1=0, lambda_contra=0, lambda_prior=0):
        super(CombinedLoss, self).__init__()
        self.lambda_mre = lambda_mre
        self.lambda_l1 = lambda_l1
        self.lambda_contra = lambda_contra
        self.lambda_prior = lambda_prior
        self.contrastive_loss_fn = ContrastiveLoss(temperature=0.07, num_positive=30, num_negative=30,posi_dist=30,neg_dist=30)  # 实例化ContrastiveLoss
        self.prior_loss_fn = nn.L1Loss()  # 假设你有一个计算先验损失的函数

    def mre_loss(self, tg, tg_output):
        # 避免除以零的情况，可以添加一个小的正数epsilon
        epsilon = 1e-8
        # 计算MRE
        relative_errors = torch.abs((tg - tg_output) / (tg + epsilon))
        mre = torch.mean(relative_errors)
        return mre

    def forward(self, tg_output, tg, tf_act, receiver_distance, model):
        mre_loss = self.mre_loss(tg, tg_output)
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        contrast_loss = self.contrastive_loss_fn(tf_act, receiver_distance)  # 调用ContrastiveLoss的forward方法
        prior_loss = self.prior_loss_fn(tg_output, tg)  # 使用L1损失作为先验损失的示例

        total_loss = (self.lambda_mre * mre_loss +
                      self.lambda_l1 * l1_loss +
                      self.lambda_contra * contrast_loss +
                      self.lambda_prior * prior_loss)
        return total_loss, mre_loss, l1_loss, contrast_loss, prior_loss
    
def train_model(model, train_loader, epochs=10, lr=1e-3, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
    model.train()

    # 初始化损失列表
    l1_losses = []
    mre_losses = []
    contrast_losses = []
    prior_losses = []
    combined_losses = []

    pbar = tqdm(total=epochs, desc="Training Progress", unit="epoch")

    for epoch in range(epochs):
        torch.autograd.set_detect_anomaly(True)
        epoch_l1_loss = 0
        epoch_mre_loss = 0
        epoch_contrast_loss = 0
        epoch_prior_loss = 0
        epoch_combined_loss = 0

        for batch_idx, (neighbor_ligand, receptor, tf, tg, neighbor_distance, receiver_distance,idx) in enumerate(train_loader):
            optimizer.zero_grad()
            receiver_distance = receiver_distance[:,idx]

            # 将数据移动到指定的设备
            neighbor_ligand, receptor, tf, tg, neighbor_distance, receiver_distance = [
                item.to(device) for item in [neighbor_ligand, receptor, tf, tg, neighbor_distance, receiver_distance]
            ]

            # 调用模型的前向传播方法
            tf_act, tg_output = model(neighbor_ligand, receptor, tf, neighbor_distance)

            # 计算损失
            combined_loss_fn = CombinedLoss(lambda_mre=1, lambda_l1=0, lambda_contra=0, lambda_prior=0)
            loss, mre_loss, l1_loss, contrast_loss, prior_loss = combined_loss_fn(tg_output, tg, tf_act, receiver_distance, model)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 累加损失
            epoch_l1_loss += l1_loss.item()
            epoch_mre_loss += mre_loss.item()
            epoch_contrast_loss += contrast_loss.item()
            epoch_prior_loss += prior_loss.item()
            epoch_combined_loss += loss.item()

        # 计算并存储每个epoch的平均损失
        avg_l1_loss = epoch_l1_loss / len(train_loader)
        avg_mre_loss = epoch_mre_loss / len(train_loader)
        avg_contrast_loss = epoch_contrast_loss / len(train_loader)
        avg_prior_loss = epoch_prior_loss / len(train_loader)
        avg_combined_loss = epoch_combined_loss / len(train_loader)

        # 记录损失
        l1_losses.append(avg_l1_loss)
        mre_losses.append(avg_mre_loss)
        contrast_losses.append(avg_contrast_loss)
        prior_losses.append(avg_prior_loss)
        combined_losses.append(avg_combined_loss)

        # 更新进度条，显示当前epoch的损失
        pbar.set_postfix(CombinedLoss=avg_combined_loss, L1Loss=avg_l1_loss, MSELoss=avg_mre_loss, ContrastLoss=avg_contrast_loss, PriorLoss=avg_prior_loss)
        pbar.update(1)

        scheduler.step()

    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(combined_losses[4:], label='Combined Loss')
    plt.title('Combined Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(l1_losses[4:], label='L1 Loss')
    plt.title('L1 Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(mre_losses[4:], label='MRE Loss')
    plt.title('MSE Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(contrast_losses[4], label='Contrastive Loss')
    plt.title('Contrastive Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()






