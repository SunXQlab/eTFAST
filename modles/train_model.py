import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
# from models.evalueate_model import compute_morans_i
#from models.input_prepare import min_max_normalize
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler

def min_max_normalize(df):
    """
    Normalize gene expression matrix using Min-Max scaling.

    Parameters:
        df (DataFrame): Gene expression DataFrame.

    Returns:
        DataFrame: Normalized gene expression DataFrame.
    """
    #对df转置
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns, index=df.index)

@dataclass
class ModelParams:
    lig_rec_dim: int
    diff_lig_dim:int
    tf_dim: int
    hidden_dim_1: int
    tg_dim: int


# 定义训练参数的数据类
@dataclass
class TrainingParams:
    pre_train_epochs: int
    train_epochs: int
    lr: float
    lambda_ewc: float
    batch_size: int


class TGDataset(Dataset):
    def __init__(self, diff_neighbor_ligand, cont_neighbor_ligand, diff_Rec_df,cont_Rec_df, TF_df, diff_neighbor_distance, cont_neighbor_distance,receiver_distance,TG_df, coords,receiver_type,split_ratio=0.8):
        self.diff_neighbor_ligand = diff_neighbor_ligand
        self.cont_neighbor_ligand = cont_neighbor_ligand
        self.diff_Rec_df = diff_Rec_df
        self.cont_Rec_df = cont_Rec_df
        self.TF_df = TF_df
        self.diff_neighbor_distance = diff_neighbor_distance
        self.cont_neighbor_distance = cont_neighbor_distance
        self.receiver_distance = receiver_distance
        self.TG_df = TG_df
        self.coords = coords
        self.split_ratio = split_ratio
        self.receiver_type = receiver_type

    def __len__(self):
        return len(self.diff_Rec_df)

    def __getitem__(self, idx):
        # 提取每个样本的邻居特征
        diff_neighbor_ligand = torch.tensor(self.diff_neighbor_ligand[idx], dtype=torch.float32)
        cont_neighbor_ligand = torch.tensor(self.cont_neighbor_ligand[idx], dtype=torch.float32)
        # 提取每个样本的受体特征
        diff_receptor = torch.tensor(self.diff_Rec_df.iloc[idx].values, dtype=torch.float32)
        cont_receptor = torch.tensor(self.cont_Rec_df.iloc[idx].values, dtype=torch.float32)
        # 提取每个样本的TF特征
        tf = torch.tensor(self.TF_df.iloc[idx].values, dtype=torch.float32)
        # 提取每个样本的邻居距离
        diff_neighbor_distance = torch.tensor(self.diff_neighbor_distance[idx], dtype=torch.float32)
        cont_neighbor_distance = torch.tensor(self.cont_neighbor_distance[idx], dtype=torch.float32)
        # 提取每个样本的TG特征
        tg = torch.tensor(self.TG_df.iloc[idx].values, dtype=torch.float32)
        coords = torch.tensor(self.coords.iloc[idx], dtype=torch.float32)
        # 提取每个样本的正样本矩阵
        receiver_distance = torch.tensor(self.receiver_distance[idx], dtype=torch.float32)


        return diff_neighbor_ligand, cont_neighbor_ligand, diff_receptor, cont_receptor, tf,  diff_neighbor_distance, cont_neighbor_distance, receiver_distance,tg,coords,idx
        
    def split(self, pretrain_size, train_size):
        # 确保分割的大小加起来等于总大小
        assert pretrain_size + train_size == len(self), "The sum of sizes must equal the total dataset size."
        # 使用random_split进行分割
        pretrain_dataset, train_dataset = random_split(self, [pretrain_size, train_size])
        return pretrain_dataset, train_dataset
    
def shuffle_cell_types(cell_types,receiver):
    """
    打乱细胞类型标签，但保持 receiver 类型的细胞位置和标签不变。

    参数:
        cell_types: 一个包含细胞类型信息的 Pandas Series，索引为细胞 ID，值为细胞类型。
    返回:
        打乱后的细胞类型 Series。
    """
    # 找到 receiver 类型的细胞
    receiver_cells = cell_types[cell_types == receiver]
    # 找到非 receiver 类型的细胞
    non_receiver_cells = cell_types[cell_types['cell_type']!=receiver]
    
    # 打乱非 receiver 类型的细胞标签
    shuffled_non_receiver_cells = non_receiver_cells.sample(frac=1).reset_index(drop=True)
    
    # 重新组合，保持 receiver 类型细胞的位置和标签不变
    cell_types_shuffled = cell_types.copy()
    cell_types_shuffled[cell_types_shuffled['cell_type']!=receiver] = shuffled_non_receiver_cells.values

    return cell_types_shuffled

class TGPredictionModel(nn.Module):
    def __init__(self, lig_rec_dim, diff_lig_dim, hidden_dim_1, tf_dim_dict,  tg_dim_dict, receivers,TFTG_matrix_dict):
        #先定义要用的参数，也就是模型的参数
        super(TGPredictionModel, self).__init__()

        # 配体-受体对的参数B，初始化在0.2到0.8之间
        self.B = nn.Parameter(torch.rand(diff_lig_dim) * (0.8 - 0.2) + 0.2)

        # 生成TF信号（TFsigal），由配体和受体的交互产生
        self.lr_hidden1 = nn.Linear(lig_rec_dim, hidden_dim_1)  # 隐藏层1

        # 为每种 receiver 单独设计 hidden1_tfsigal 层
        self.hidden1_tfsigal_dict = nn.ModuleDict({
            receiver: nn.Linear(hidden_dim_1, tf_dim_dict[receiver]) for receiver in receivers
        })

        # 为每种 receiver 单独设计 TF 信号处理层
        self.tfsigal_tfsigal_dict = nn.ModuleDict({
            receiver: nn.Linear(tf_dim_dict[receiver], tf_dim_dict[receiver]) for receiver in receivers
        })

        # 为每种 receiver 单独设计 TF 表达式处理层
        self.tfexpr_tfexpr_dict = nn.ModuleDict({
            receiver: nn.Linear(tf_dim_dict[receiver], tf_dim_dict[receiver]) for receiver in receivers
        })

        # 为每种 receiver 单独设计最终的输出层
        self.tfact_tg_dict = nn.ModuleDict()
        for receiver in receivers:
            TFTG_matrix_tensor = torch.tensor(TFTG_matrix_dict[receiver].values).float()
            #TFTG_matrix_tensor = (TFTG_matrix_tensor - TFTG_matrix_tensor.mean()) / TFTG_matrix_tensor.std()
            self.tfact_tg_dict[receiver] = nn.Linear(tf_dim_dict[receiver], tg_dim_dict[receiver])
            self.tfact_tg_dict[receiver].weight.data.copy_(TFTG_matrix_tensor.T)
            self.tfact_tg_dict[receiver].bias.data.fill_(0)

    def _compute_diff_lr_score(self, diff_lig, diff_rec, diff_dist):
        """计算扩散配体-受体得分"""
        rec_expanded = diff_rec.unsqueeze(-2)  # [batch, rec_nodes, 1, lr_dim]
        inv_dist = 1 / diff_dist.unsqueeze(-1)  # [batch, rec_nodes, lig_nodes, 1]
        exp_term = torch.exp((1 - diff_dist).unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0))
        return rec_expanded * diff_lig * inv_dist * exp_term

    def _compute_cont_lr_score(self, cont_lig, cont_rec, cont_dist):
        """计算接触配体-受体得分"""
        cont_rec_exp = cont_rec.unsqueeze(-2)
        return cont_rec_exp * cont_lig * cont_dist.unsqueeze(-1)

    def _compute_tf_signal(self, receiver, lr_score):
        """计算 TF 信号"""
        hidden = torch.relu(self.lr_hidden1(lr_score))
        tf_signal = torch.relu(self.hidden1_tfsigal_dict[receiver](hidden))
        return self.tfsigal_tfsigal_dict[receiver](tf_signal)

    def _compute_tf_expr(self, receiver, tf):
        """计算 TF 表达"""
        return self.tfexpr_tfexpr_dict[receiver](tf)
    def forward_single_receiver(self, receiver, lr_score, tf):
        """
        计算单个 receiver 的 TF 激活和 TG 输出。

        参数:
            receiver: 接收细胞类型。
            lr_score: 配体-受体对的 LR 得分。
            tf: TF 表达。
        """
        # 1. 计算 TF 信号和表达
        tf_signal = self._compute_tf_signal(receiver, lr_score)
        tf_expr = self._compute_tf_expr(receiver, tf)
        
        # 2. 计算 TF 激活和 TG 输出
        tf_act = torch.relu(tf_signal * tf_expr)
        tg_output = torch.relu(self.tfact_tg_dict[receiver](tf_act))
        
        return tf_act, tg_output
    
    def forward(self, receiver, diff_lig, cont_lig, diff_rec, cont_rec, tf, diff_dist, cont_dist):
        # 1. 计算 LR 得分
        diff_score = self._compute_diff_lr_score(diff_lig, diff_rec, diff_dist)
        cont_score = self._compute_cont_lr_score(cont_lig, cont_rec, cont_dist)
        combined_lr_score = torch.sigmoid(torch.cat((diff_score, cont_score), dim=2).sum(1))
        tf_act, tg_output = self.forward_single_receiver(receiver, combined_lr_score, tf)
        return combined_lr_score, tf_act, tg_output

    # def forward(self,receiver,diff_neighbor_ligand, cont_neighbor_ligand, diff_receptor, cont_receptor, tf, diff_neighbor_distance, cont_neighbor_distance):
        
    #     # Step 2: 计算配体-受体对的diff_LR得分
    #     expanded_receptor = diff_receptor.unsqueeze(-2)  # 扩展受体矩阵，使其维度为 batch受体结点* 1 * lr
    #     distance_factor = 1 / diff_neighbor_distance  # 距离的倒数 维度为 batch受体结点 * 邻居配体结点
    #     expand_distance_factor = distance_factor.unsqueeze(-1)  # 扩展距离的倒数，使其维度为 batch受体结点 * 邻居配体结点 * 1
    #     exp_factor = (1 - diff_neighbor_distance).unsqueeze(-1)  # 距离的倒数 维度为 batch受体结点 * 邻居配体结点 * 1

    #     exp_factor = torch.exp(exp_factor * self.B.unsqueeze(0).unsqueeze(0))  # e^((1-distance)*B)
    #     # 计算LR得分，注意维度匹配和广播
    #     #print('LR_score1 :', expanded_receptor.shape, neighbor_ligand.shape, expand_distance_factor.shape, exp_factor.shape)
    #     LR_score = expanded_receptor * diff_neighbor_ligand * expand_distance_factor * exp_factor
        
    #     #计算cont_LR得分
    #     cont_expanded_receptor = cont_receptor.unsqueeze(-2)
    #     cont_distance_factor = cont_neighbor_distance.unsqueeze(-1)
    #     cont_LRscore = cont_expanded_receptor * cont_neighbor_ligand * cont_distance_factor 
    #     combine_LR_score = torch.cat((LR_score, cont_LRscore), dim=2)
    #     combine_LR_score = torch.sigmoid(combine_LR_score.sum(dim=1))  # 对邻居配体结点求和，得到一个 batch * 受体结点 的矩阵
    #     #print('LR_score :',LR_score.shape)
    #     # Step 3: 生成 TF 信号（TFsigal）
    #     tf_sigal = torch.relu(self.lr_hidden1(combine_LR_score))  # 转到隐藏层1
    #     # Step 4: 对 TF 矩阵进行进一步处理
    #     tf_sigal = torch.relu(self.hidden1_tfsigal_dict[receiver](tf_sigal))  # 变成 tf 信号
    #     tf_sigal = self.tfsigal_tfsigal_dict[receiver](tf_sigal)  # 对 TF 信号进行线性变换
    #     tf_expr = self.tfexpr_tfexpr_dict[receiver](tf)  # 对输入的 TF 矩阵进行线性变换

    #     # Step 5: 结合 TF 信号生成最终的 TF 矩阵
    #     tf_act = torch.relu(tf_sigal * tf_expr)  # TF 矩阵由 TFsigal 和 TFexpr 的逐元素相乘得到

    #     # Step 6: 计算 TG 矩阵并输出
    #     tg_output = torch.relu(self.tfact_tg_dict[receiver](tf_act))  # TFact 信号得到 TG 矩阵
        
    #     return tf_act, tg_output  # 返回TG矩阵作为最终输出
    def calculate_LRscore_celltypes(self, adata, cell_types, distance_mat, senders, receivers, LR_pairs_diff, LR_pairs_cont,shuffle=False):
        """
        计算配体-受体对的得分，包括差异表达和连续表达的配体-受体对。

        参数:
            adata: AnnData对象,包含单细胞数据。
            cell_types: 包含细胞类型信息的DataFrame。
            distance_mat: 距离矩阵。
            senders: 发送细胞类型列表。
            receivers: 接收细胞类型列表。
            LR_pairs_diff: 差异表达的配体-受体对列表。
            LR_pairs_cont: 连续表达的配体-受体对列表。
        """

        # 初始化结果列表
        combine_LR_scores = {sender+'_'+ receiver:[] for sender in senders for receiver in receivers if sender!= receiver}
        for receiver in receivers:
            # 打乱细胞类型标签，但保持 receiver 类型的细胞位置和标签不变
            if shuffle:
                cell_types_shuffled = shuffle_cell_types(cell_types, receiver)
            else:
                cell_types_shuffled = cell_types.copy()

            for sender in senders:
                if sender == receiver:
                    continue
                
                # 提取配体和受体基因
                diff_lig_genes = [col.split('_')[0] for col in LR_pairs_diff]
                diff_rec_genes = [col.split('_')[1] for col in LR_pairs_diff]
                cont_lig_genes = [col.split('_')[0] for col in LR_pairs_cont]
                cont_rec_genes = [col.split('_')[1] for col in LR_pairs_cont]

                # 提取配体和受体表达数据
                diff_lig_df = adata[cell_types['cell_type'] == sender][:, diff_lig_genes].to_df()
                diff_rec_df = adata[cell_types['cell_type'] == receiver][:, diff_rec_genes].to_df()
                cont_lig_df = adata[cell_types['cell_type'] == sender][:, cont_lig_genes].to_df()
                cont_rec_df = adata[cell_types['cell_type'] == receiver][:, cont_rec_genes].to_df()
                #normalize
                diff_lig_df = min_max_normalize(diff_lig_df)
                diff_rec_df = min_max_normalize(diff_rec_df)
                cont_lig_df = min_max_normalize(cont_lig_df)                
                cont_rec_df = min_max_normalize(cont_rec_df)


                # 重新设置列名为配体-受体对
                # diff_lig_df.columns = LR_pairs_diff
                # diff_rec_df.columns = LR_pairs_diff
                # cont_lig_df.columns = LR_pairs_cont
                # cont_rec_df.columns = LR_pairs_cont

                #print(f'Extracting neighbor ligands and distances for receiver: {receiver}')

                sender_cells = cell_types_shuffled['cell_type'] == sender
                receiver_cells = cell_types['cell_type'] == receiver
                # 提取邻居距离矩阵
                neighbors_distance_mat = distance_mat[receiver_cells, :][:, sender_cells]  

                # 获取每个接收细胞的前100个邻居
                num_receivers = neighbors_distance_mat.shape[0]
                num_neighbors = min(neighbors_distance_mat.shape[1], 100)
                neighbor_indices = np.argsort(neighbors_distance_mat, axis=1)[:, :num_neighbors]

                # 提取邻居配体表达和距离
                diff_neighbor_ligand = diff_lig_df.values[neighbor_indices]
                cont_neighbor_ligand = cont_lig_df.values[neighbor_indices]
                diff_neighbor_distance = neighbors_distance_mat[np.arange(num_receivers)[:, None], neighbor_indices]
        

                # 转换为张量
                diff_neighbor_ligand = torch.tensor(diff_neighbor_ligand, dtype=torch.float32)
                cont_neighbor_ligand = torch.tensor(cont_neighbor_ligand, dtype=torch.float32)
                diff_neighbor_distance = torch.tensor(diff_neighbor_distance, dtype=torch.float32)

                # 计算连续配体的邻居距离掩码
                cont_neighbor_distance = (diff_neighbor_distance < np.sqrt(3)).float()

                # 提取受体表达
                diff_receptor = torch.tensor(diff_rec_df.values, dtype=torch.float32)
                cont_receptor = torch.tensor(cont_rec_df.values, dtype=torch.float32)

                # 计算差异配体的LR得分
                distance_factor = 1 / diff_neighbor_distance  # 距离的倒数
                exp_factor = torch.exp((1 - diff_neighbor_distance).unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0))  # e^((1-distance)*B)
                diff_LR_score = diff_receptor.unsqueeze(-2) * diff_neighbor_ligand * distance_factor.unsqueeze(-1) * exp_factor

                # 计算连续配体的LR得分
                cont_LRscore = cont_receptor.unsqueeze(-2) * cont_neighbor_ligand * cont_neighbor_distance.unsqueeze(-1)

                # 合并两种得分
                combine_LR_score = torch.cat((diff_LR_score, cont_LRscore), dim=2)
                combine_LR_score = combine_LR_score.sum(dim=1)
                combine_LR_score = combine_LR_score.sum(dim=0)

                # 保存结果
                combine_LR_scores[sender + '_' + receiver] = pd.DataFrame(
                [combine_LR_score.detach().numpy()],
                columns=LR_pairs_diff.tolist() + LR_pairs_cont.tolist()
    )
                self.LRscore = combine_LR_scores


        return combine_LR_scores


    def calculate_LRscore_coords(self, adata, cell_types, distance_mat, senders, receivers, LR_pairs_diff, LR_pairs_cont, shuffle=False):
        """计算配体-受体相互作用得分（优化版）"""
        
        # 预处理基因列表
        diff_lig = [x.split('_')[0] for x in LR_pairs_diff]
        diff_rec = [x.split('_')[1] for x in LR_pairs_diff]
        cont_lig = [x.split('_')[0] for x in LR_pairs_cont] 
        cont_rec = [x.split('_')[1] for x in LR_pairs_cont]
        
        # 预归一化所有表达数据
        norm_expr = min_max_normalize(adata.to_df())
        results = {}
        
        for receiver in receivers:
            # print(f"Calculating LR score for receiver: {receiver}")
            #归一化受体数据
            norm_expr.loc[cell_types['cell_type'] == receiver] = min_max_normalize(adata[cell_types['cell_type'] == receiver].to_df())
            # 处理细胞类型标签（是否打乱）

            ct_shuffle = shuffle_cell_types(cell_types, receiver) if shuffle else cell_types
            rec_mask = (cell_types['cell_type'] == receiver).values
            rec_cells = np.where(rec_mask)[0]
            
            for sender in senders:
                if sender == receiver: continue
                
                # 获取发送细胞索引
                snd_mask = (ct_shuffle['cell_type'] == sender).values
                snd_cells = np.where(snd_mask)[0]
                if len(snd_cells) == 0 or len(rec_cells) == 0: continue
                
                # 1. 距离矩阵处理（优化关键步骤）
                dist_mat = distance_mat[rec_mask][:, snd_mask]
                neighbors = np.argpartition(dist_mat,min(dist_mat.shape[1],100)-1, axis=1)[:, :min(dist_mat.shape[1],100)]
                sorted_idx = np.argsort(np.take_along_axis(dist_mat, neighbors, axis=1), axis=1)
                neighbors = np.take_along_axis(neighbors, sorted_idx, axis=1)
                
                # 2. 表达数据提取（向量化操作）
                lig_diff = np.take(norm_expr.iloc[snd_cells][diff_lig].values, neighbors, axis=0)
                lig_cont = np.take(norm_expr.iloc[snd_cells][cont_lig].values, neighbors, axis=0)
                dist_diff = np.take_along_axis(dist_mat, neighbors, axis=1)
                
                # 3. 转换为张量计算
                lig_diff = torch.tensor(lig_diff, dtype=torch.float32)
                lig_cont = torch.tensor(lig_cont, dtype=torch.float32)
                dist_diff = torch.tensor(dist_diff, dtype=torch.float32)
                dist_cont = (dist_diff < np.sqrt(3)).float()
                
                # 4. 受体表达数据
                rec_diff = torch.tensor(norm_expr.iloc[rec_cells][diff_rec].values, dtype=torch.float32)
                rec_cont = torch.tensor(norm_expr.iloc[rec_cells][cont_rec].values, dtype=torch.float32)
                
                # 5. 计算得分（保持原公式）
                # 差异表达得分
                dist_factor = 1 / (dist_diff + 1e-8)
                exp_factor = torch.exp((1 - dist_diff).unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0))
                score_diff = rec_diff.unsqueeze(-2) * lig_diff * dist_factor.unsqueeze(-1) * exp_factor
                
                # 连续表达得分
                score_cont = rec_cont.unsqueeze(-2) * lig_cont * dist_cont.unsqueeze(-1)
                
                # 合并结果
                combined = torch.cat((score_diff, score_cont), dim=2).sum(dim=(0,1))
                
                # 存储结果
                key = f"{sender}_{receiver}"
                results[key] = pd.DataFrame([combined.detach().numpy()], 
                                        columns=LR_pairs_diff.tolist()+LR_pairs_cont.tolist())
        
        return results       
################################################################################################ 



class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, num_positive=10, num_negative=30, pos_threshold=30, neg_threshold=100):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.pos_threshold = pos_threshold  # 正样本的最大距离阈值
        self.neg_threshold = neg_threshold  # 负样本的最小距离阈值

    def forward(self, tf_act, receiver_distance):
        # 计算tf_act的相似度矩阵
        sim_matrix = F.cosine_similarity(tf_act.unsqueeze(1), tf_act.unsqueeze(0), dim=2)
        
        # 将对角线元素设置为负无穷，避免自己与自己相似
        receiver_distance.fill_diagonal_(-float('inf'))
        
        # 获取正样本和负样本的相似度矩阵
        # 正样本: 获取与每个样本距离最小的 num_positive 个样本
        _, pos_indices = torch.topk(-receiver_distance, self.num_positive, dim=1)  # 每行取出最小的num_positive个距离
        pos_distances = torch.gather(receiver_distance, 1, pos_indices)  # 获取相应的距离
        pos_sim = torch.gather(sim_matrix, 1, pos_indices)  # 获取正样本的相似度

        # 对于正样本，如果距离大于 pos_threshold，将其相似度设置为0
        valid_pos_mask = pos_distances <= self.pos_threshold
        pos_sim = pos_sim * valid_pos_mask.float()  # 不满足条件的相似度设为0

        # 检查正样本数量是否足够
        num_valid_pos = valid_pos_mask.sum(dim=1)
        valid_samples_mask = num_valid_pos == self.num_positive  # 只保留正样本数量至少为self.num_positive的样本

        # 负样本: 获取与每个样本距离最大的 num_negative 个样本
        _, neg_indices = torch.topk(receiver_distance, self.num_negative, dim=1, largest=True)  # 每行取出最大的num_negative个距离
        neg_distances = torch.gather(receiver_distance, 1, neg_indices)  # 获取相应的距离
        neg_sim = torch.gather(sim_matrix, 1, neg_indices)  # 获取负样本的相似度

        # 对于负样本，距离小于 neg_threshold 的要排除
        valid_neg_mask = neg_distances > self.neg_threshold
        neg_sim = neg_sim * valid_neg_mask.float()  # 不满足条件的相似度设为0

        # 检查负样本数量是否足够
        num_valid_neg = valid_neg_mask.sum(dim=1)
        valid_samples_mask = valid_samples_mask & (num_valid_neg >= self.num_negative)  # 只保留负样本数量至少为self.num_negative的样本

        if valid_samples_mask.sum() == 0:
            # 如果没有有效的样本，则直接返回损失为0
            return torch.tensor(0.0, device=tf_act.device)

        # 只对有效的样本进行损失计算
        pos_sim = pos_sim[valid_samples_mask]
        neg_sim = neg_sim[valid_samples_mask]

        # 计算正样本的损失
        pos_loss = -torch.log(torch.sum(torch.exp(pos_sim / self.temperature), dim=1) / 
                              (torch.sum(torch.exp(neg_sim / self.temperature), dim=1) + torch.sum(torch.exp(pos_sim / self.temperature), dim=1)))
        
        # 计算平均损失
        loss = pos_loss.mean()
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, lambda_mse=1, lambda_l1=0, lambda_contra=0):
        super(CombinedLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.lambda_contra = lambda_contra
        self.contrastive_loss_fn = ContrastiveLoss(num_positive=5, num_negative=30, pos_threshold=20, neg_threshold=20)
        # # 假设你有一个计算先验损失的函数

    def forward(self, tg_output, tg, tf_act, receiver_distance, model):
        #MSE Loss
        criterion = nn.MSELoss()
        mse_loss = criterion(tg, tg_output)
        #L1 Loss
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())

        
        if tf_act.shape[0] < 30:
            contrast_loss = criterion(tg, tg)  # 若样本数小于10，则不计算Contrastive Loss
        else:
            contrast_loss = self.contrastive_loss_fn(tf_act, receiver_distance)  # 调用ContrastiveLoss的forward方法
        
        # 计算总损失
        total_loss = (self.lambda_mse * mse_loss +
                      self.lambda_l1 * l1_loss +
                      self.lambda_contra * contrast_loss )       
        return total_loss, mse_loss, l1_loss, contrast_loss
    

def pretrain_model(model, train_loader, receivers, epochs=10, lr=1e-3, 
                  device='cpu', save_fig_path=None):
    """
    预训练模型
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器(字典，按接收器分类)
        receivers: 接收器列表
        epochs: 训练轮数(默认10)
        lr: 学习率(默认1e-3)
        device: 训练设备(默认'cpu')
        save_fig_path: 图表保存路径(可选)
    """
    # 初始化优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    combined_loss_fn = CombinedLoss(lambda_mse=1, lambda_l1=0, lambda_contra=1e-1)
    
    # 初始化损失跟踪器
    train_losses = {"combined": [], "mse": [], "l1": [], "contrast": []}
    
    # 准备混合训练数据
    combined_data = prepare_mixed_data(train_loader, receivers)
    
    # 训练循环
    train_text = "Pre-training combined data :" if len(receivers) > 1 else f"Pre-training {receivers[0]}:"
    with tqdm(total=epochs, desc=train_text, unit="epoch") as pbar:
        for epoch in range(epochs):
            model.train()
            epoch_losses = {"combined": 0, "mse": 0, "l1": 0, "contrast": 0}
            batch_count = 0
            
            for receiver, data in combined_data:
                # 数据准备
                inputs = prepare_batch_data(data, device)
                
                # 前向传播和损失计算
                loss, losses = train_batch(
                    model, optimizer, combined_loss_fn, 
                    receiver, inputs, data, device
                )
                
                # 记录损失
                for key in epoch_losses:
                    epoch_losses[key] += losses[key]
                batch_count += 1
            
            # 计算平均损失并更新记录
            update_loss_records(train_losses, epoch_losses, batch_count)
            
            # 更新进度条和学习率
            pbar.set_postfix({
                'Combined Loss': epoch_losses["combined"] / batch_count,
                'MSE Loss': epoch_losses["mse"] / batch_count
            })
            pbar.update(1)
            scheduler.step()
    
    # 保存训练曲线
    receiver_text = "all receivers" if len(receivers) > 1 else receivers[0]
    print(f"Pre-training completed for {receiver_text}.")
    if save_fig_path:
        plot_pretraining_curves(train_losses, save_fig_path,receiver_text)
    
    return {
        'model': model,
        'train_losses': train_losses
    }

# 辅助函数
def prepare_mixed_data(train_loader, receivers):
    """准备混合训练数据"""
    combined_data = []
    if len(receivers) > 1:
        print("混合多个receiver的数据进行训练...")
        for receiver in receivers:
            combined_data.extend([(receiver, data) for data in train_loader[receiver]])
    else:
        print("使用单个receiver数据进行训练...")
        receiver = receivers[0]
        combined_data = [(receiver, data) for data in train_loader[receiver]]
    return combined_data

def prepare_batch_data(data, device):
    """准备批次数据"""
    inputs = [item.to(device) for item in data[:-4]]
    receiver_distance = data[-4].to(device)
    tg = data[-3].to(device)
    idx = data[-1].to(device)
    receiver_distance = receiver_distance[:, idx]
    return inputs, receiver_distance, tg, idx

def train_batch(model, optimizer, loss_fn, receiver, inputs, data, device):
    """训练单个批次"""
    optimizer.zero_grad()
    
    # 前向传播
    lr_scores,tf_act, tg_output = model(receiver, *inputs[0])
    
    # 计算损失
    loss, mse_loss, l1_loss, contrast_loss = loss_fn(
        tg_output, inputs[2], tf_act, inputs[1], model
    )
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    return loss, {
        "combined": loss.item(),
        "mse": mse_loss.item(),
        "l1": l1_loss.item(),
        "contrast": contrast_loss.item()
    }

def update_loss_records(train_losses, epoch_losses, batch_count):
    """更新损失记录"""
    for key in epoch_losses:
        epoch_losses[key] /= batch_count
        train_losses[key].append(epoch_losses[key])

def plot_pretraining_curves(train_losses, save_path, receiver_text=""):
    """绘制预训练曲线"""
    plt.figure(figsize=(12, 8))
    
    # 定义子图和对应的损失类型
    plot_configs = [
        (1, "combined", "Combined Loss"),
        (2, "mse", "MSE Loss"),
        (3, "l1", "L1 Loss"),
        (4, "contrast", "Contrastive Loss")
    ]
    
    # 绘制每个损失曲线
    for pos, loss_key, title in plot_configs:
        plt.subplot(2, 2, pos)
        plt.plot(train_losses[loss_key], label=f"Train {title}")
        plt.title(f"{title} per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"pretraining_curves_{receiver_text}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()        

def estimate_ewc_params(model, train_loader, receivers, num_batch=100, device='cpu'):
    estimated_mean = {}
    estimated_fisher = {}
    
    # 初始化 estimated_mean 和 estimated_fisher
    for param_name, param in model.named_parameters():
        if param_name.startswith('tfact_tg_dict.') and param_name.endswith('.weight'):
            receiver = param_name.split('.')[1]
            estimated_mean[receiver] = param.data.clone()
            estimated_fisher[receiver] = torch.zeros_like(param)

    model = model.to(device)
    model.eval()

    for receiver in receivers:
        loss_total = 0

        for batch_idx, data in enumerate(train_loader[receiver]):
            if batch_idx >= num_batch:
                break

            # 数据准备
            inputs = [item.to(device) for item in data[:-4]]
            receiver_distance = data[-4].to(device)
            tg = data[-3].to(device)
            idx = data[-1].to(device)
            receiver_distance = receiver_distance[:, idx]

            # 调用模型的前向传播方法
            lr_scores,tf_act, tg_output = model(receiver, *inputs)

            # 计算损失
            combined_loss_fn = CombinedLoss(lambda_mse=1, lambda_l1=0, lambda_contra=1e-1)
            loss, mse_loss, l1_loss, contrast_loss = combined_loss_fn(tg_output, tg, tf_act, receiver_distance, model)
            loss_total += loss

            # 累积梯度
            loss.backward()

        # 累积 Fisher 信息矩阵
        for param_name, param in model.named_parameters():
            if param_name == f'tfact_tg_dict.{receiver}.weight':
                if param.grad is not None:
                    estimated_fisher[receiver].data += param.grad.data ** 2
                else:
                    print(f"Warning: Gradient for {param_name} is None. Skipping Fisher accumulation for this parameter.")

        # 清空梯度
        model.zero_grad()

    # 归一化 Fisher 信息矩阵
    for receiver in receivers:
        estimated_fisher[receiver].data /= num_batch

    return estimated_mean, estimated_fisher

def calculate_ewc_loss(model, lambda_ewc, estimated_fishers, estimated_means):
    losses = []
    for param_name, param in model.named_parameters():
        if param_name.startswith('tfact_tg_dict.') and param_name.endswith('.weight'):
            receiver = param_name.split('.')[1]
            estimated_mean = estimated_means[receiver]
            estimated_fisher = estimated_fishers[receiver]
            losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
        
    return (lambda_ewc / 2) * sum(losses)   
    
def train_model_with_ewc(model, train_loader, receivers, estimated_mean, 
                        estimated_fisher, epochs=10, lr=1e-3, device='cpu', 
                        lambda_ewc=1e-3, save_fig_path=None):
    """
    使用弹性权重固化(EWC)训练模型（仅训练阶段）
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器(字典，按接收器分类)
        receivers: 接收器列表
        estimated_mean: EWC参数均值估计
        estimated_fisher: EWC Fisher矩阵估计
        epochs: 训练轮数(默认10)
        lr: 学习率(默认1e-3)
        device: 训练设备(默认'cpu')
        lambda_ewc: EWC损失权重(默认1e-3)
        save_fig_path: 图表保存路径(可选)
    """
    # 初始化优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    combined_loss_fn = CombinedLoss(lambda_mse=1, lambda_l1=0, lambda_contra=1e-1)
    
    # 初始化损失和指标跟踪器
    train_losses = {"combined": [], "mse": [], "l1": [], "contrast": [], "ewc": []}
    R2s = {receiver: [] for receiver in receivers}
    metrics = {'mean_r2': []}
    
    # 准备混合训练数据
    combined_data = []
    if len(receivers) > 1:
        print("混合多个receiver的数据进行训练...")
        for receiver in receivers:
            for data in train_loader[receiver]:
                combined_data.append((receiver, data))
    else:
        print("使用单个receiver数据进行训练...")
        receiver = receivers[0]
        combined_data = [(receiver, data) for data in train_loader[receiver]]
    
    # 训练循环
    train_text = "EWC for combined data :" if len(receivers) > 1 else f"EWC for {receivers[0]}:"
    with tqdm(total=epochs, desc=train_text, unit="epoch") as pbar:
        for epoch in range(epochs):
            model.train()
            epoch_losses = {"combined": 0, "mse": 0, "l1": 0, "contrast": 0, "ewc": 0}
            batch_count = 0
            
            # 存储所有预测和真实值用于计算R2
            all_tg = {receiver: [] for receiver in receivers}
            all_tg_output = {receiver: [] for receiver in receivers}
            
            # 训练批次
            for receiver, data in combined_data:
                # 数据准备
                inputs = [item.to(device) for item in data[:-4]]
                receiver_distance = data[-4].to(device)
                tg = data[-3].to(device)
                idx = data[-1].to(device)
                receiver_distance = receiver_distance[:, idx]
                
                # 前向传播
                optimizer.zero_grad()
                lr_scores, tf_act, tg_output = model(receiver, *inputs)
                
                # 计算损失
                loss, mse_loss, l1_loss, contrast_loss = combined_loss_fn(
                    tg_output, tg, tf_act, receiver_distance, model
                )
                ewc_loss = calculate_ewc_loss(model, lambda_ewc, estimated_fisher, estimated_mean)
                total_loss = ewc_loss + loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                # 记录损失
                epoch_losses["combined"] += total_loss.item()
                epoch_losses["mse"] += mse_loss.item()
                epoch_losses["l1"] += l1_loss.item()
                epoch_losses["contrast"] += contrast_loss.item()
                epoch_losses["ewc"] += ewc_loss.item()
                
                # 存储预测和真实值
                all_tg[receiver].append(tg.detach().cpu().numpy())
                all_tg_output[receiver].append(tg_output.detach().cpu().numpy())
                
                batch_count += 1
            
            # 计算平均损失
            for key in epoch_losses:
                epoch_losses[key] /= batch_count
                train_losses[key].append(epoch_losses[key])
            
            # 计算R2分数（整个epoch）
            epoch_R2 = {}
            for receiver in receivers:
                if len(all_tg[receiver]) > 0:  # 确保有数据
                    tg_np = np.concatenate(all_tg[receiver])
                    tg_output_np = np.concatenate(all_tg_output[receiver])
                    r2_scores = [np.corrcoef(tg_np[:, i], tg_output_np[:, i])[0, 1] 
                                for i in range(tg_np.shape[1])]
                    epoch_R2[receiver] = np.nanmean(r2_scores)
                    R2s[receiver].append(epoch_R2[receiver])
                else:
                    epoch_R2[receiver] = 0
                    R2s[receiver].append(0)
            
            mean_r2 = np.mean(list(epoch_R2.values()))
            metrics['mean_r2'].append(mean_r2)
            
            # 更新进度条
            pbar.set_postfix({
                'Train Loss': epoch_losses["combined"],
                'MSE Loss': epoch_losses["mse"],
                'EWC Loss': epoch_losses["ewc"],
                'R2 Score': mean_r2
            })
            pbar.update(1)
            
            # 更新学习率
            scheduler.step()
    
    # 绘制训练曲线
    if save_fig_path:
        receiver_text = "all receivers" if len(receivers) > 1 else receivers[0]
        print(f"EWC training completed for {receiver_text}.")
        plot_training_curves(train_losses, metrics, save_fig_path, receiver_text)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'R2_scores': R2s,
        'metrics': metrics
    }
    
def plot_training_curves(train_losses, metrics, save_path, receiver_text=""):
    """绘制训练曲线"""
    plt.figure(figsize=(18, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(train_losses["combined"], label="Combined Loss")
    plt.title("Combined Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(train_losses["mse"], label="MSE Loss")
    plt.title("MSE Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(train_losses["l1"], label="L1 Loss")
    plt.title("L1 Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot(train_losses["contrast"], label="Contrastive Loss")
    plt.title("Contrastive Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(train_losses["ewc"], label="EWC Loss")
    plt.title("EWC Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(metrics["mean_r2"], label="R2 Score")
    plt.title("R2 Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"training_curves_{receiver_text}.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()


def predict_model(model, dataset, receivers, batch_size=64, device='cpu'):
    """
    增强版预测函数，确保LRscore带梯度，并计算：
    1. 基础预测结果 (lr_score, tf_act, tg_output)
    2. LR对TF的重要性
    3. TF对TG的重要性
    4. TF-TG权重矩阵

    参数:
    - model: 训练好的模型 (forward返回lr_score, tf_act, tg_output)
    - dataset: 数据集
    - receivers: 接收者列表
    - batch_size: 批大小
    - device: 设备 ('cuda' 或 'cpu')

    返回:
        lr_scores: {receiver: np.array} 形状 [n_cells, n_LRs]
        tf_acts: {receiver: np.array} 形状 [n_cells, n_TFs]
        tg_outputs: {receiver: np.array} 形状 [n_cells, n_TGs]
        tgs: {receiver: np.array} 真实靶基因表达
        coords: {receiver: np.array} 细胞坐标
        lr_tf_importance: {receiver: np.array} 形状 [n_TFs, n_LRs]
        tf_tg_importance: {receiver: np.array} 形状 [n_TGs, n_TFs]
    """
    model.eval()
    results = {
        'lr_scores': {r: [] for r in receivers},
        'tf_acts': {r: [] for r in receivers},
        'tg_outputs': {r: [] for r in receivers},
        'tgs': {r: [] for r in receivers},
        'coords': {r: [] for r in receivers},
        'lr_tf_importance': {r: [] for r in receivers},
        'tf_tg_importance': {r: [] for r in receivers}
    }

    for receiver in receivers:
        data_loader = DataLoader(dataset[receiver], batch_size=batch_size, shuffle=False)
        
        for batch in data_loader:
            # 准备输入数据（确保LR相关输入可求导）
            inputs = [
                item.clone().to(device).requires_grad_(True) if i in [0,1,2,3,4]  # diff_lig, cont_lig, diff_rec, cont_rec,tf
                else item.to(device) 
                for i, item in enumerate(batch[:-4])
            ]
            
            # 其他数据
            tg = batch[-3].to(device)
            coord = batch[-2].to(device)

            # 前向传播（确保计算梯度）
            with torch.set_grad_enabled(True):
                lr_scores, tf_act, tg_output = model(receiver, *inputs)

            # 存储基础结果
            results['lr_scores'][receiver].append(lr_scores.detach().cpu().numpy())
            results['tf_acts'][receiver].append(tf_act.detach().cpu().numpy())
            results['tg_outputs'][receiver].append(tg_output.detach().cpu().numpy())
            results['tgs'][receiver].append(tg.cpu().numpy())
            results['coords'][receiver].append(coord.cpu().numpy())

            # ===== 重要性计算 =====
            # 1. 计算TF对TG的重要性
            n_tgs = tg_output.shape[1]
            tf_tg_grads = []
            for tg_idx in range(n_tgs):
                grad = torch.autograd.grad(
                    outputs=tg_output[:, tg_idx],
                    inputs=tf_act,  # 直接对tf_act求导
                    grad_outputs=torch.ones_like(tg_output[:, tg_idx]),
                    retain_graph=True
                )[0]  # [batch, n_TFs]
                tf_tg_grads.append(grad.unsqueeze(1))  # [batch, 1, n_TFs]
            
            tf_tg_importance = torch.cat(tf_tg_grads, dim=1)  # [batch, n_TGs, n_TFs]
            results['tf_tg_importance'][receiver].append(tf_tg_importance.detach().cpu().numpy())

            # 2. 计算LR对TF的重要性（使用带梯度的lr_score）
            n_tfs = tf_act.shape[1]
            lr_tf_grads = []
            for tf_idx in range(n_tfs):
                grad = torch.autograd.grad(
                    outputs=tf_act[:, tf_idx],
                    inputs=lr_scores,  # 直接对lr_scores求导
                    grad_outputs=torch.ones_like(tf_act[:, tf_idx]),
                    retain_graph=True
                )[0]  # [batch, n_LRs]
                lr_tf_grads.append(grad.unsqueeze(1))  # [batch, 1, n_LRs]
            
            lr_tf_importance = torch.cat(lr_tf_grads, dim=1)  # [batch, n_TFs, n_LRs]
            results['lr_tf_importance'][receiver].append(lr_tf_importance.detach().cpu().numpy())
        
        # 拼接batch结果
        for key in ['lr_scores', 'tf_acts', 'tg_outputs', 'tgs', 'coords', 'lr_tf_importance', 'tf_tg_importance']:
            results[key][receiver] = np.concatenate(results[key][receiver], axis=0)
        
        # 标准化重要性
        results['lr_tf_importance'][receiver] = normalize_importance(results['lr_tf_importance'][receiver])
        results['tf_tg_importance'][receiver] = normalize_importance(results['tf_tg_importance'][receiver])

    return tuple(results[key] for key in [
        'lr_scores', 'tf_acts', 'tg_outputs', 'tgs', 'coords',
        'lr_tf_importance', 'tf_tg_importance'])

def normalize_importance(matrix):
    """标准化重要性矩阵（0-1范围）"""
    abs_matrix = np.abs(matrix)
    matrix = np.mean(abs_matrix, axis=0)  # 平均所有细胞
    return (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)  # 避免除零


def visualize_correlations(tg_outputs, tgs, save_path=None):
    """
    可视化每个基因的 R² 分数和皮尔逊相关系数。

    参数:
    - tg_outputs: 预测结果
    - tgs: 真实值
    - gene_means: 每个基因的真实表达均值
    - save_path: 保存图片的路径（默认为 None，不保存）
    """
    
    # 计算每个基因的 R² 分数
    num_genes = tgs.shape[1]
    #r2_scores = [np.sum((tg_outputs[:, gene]-np.mean(tgs[:, gene])) ** 2)/np.sum((tgs[:, gene]-np.mean(tgs[:, gene])) ** 2) for gene in range(num_genes)]
    r2_scores = [1-(np.sum((tg_outputs[:, gene]-tgs[:, gene]) ** 2)/np.sum((tgs[:, gene]-np.mean(tgs[:, gene])) ** 2)) for gene in range(num_genes)]
    gene_means = [np.mean(tgs[:, gene]) for gene in range(num_genes)]
    predicted_means = [np.mean(tg_outputs[:, gene]) for gene in range(num_genes)]


    pearson_corrs = []

    for gene in range(num_genes):
        # 提取单个基因的数据
        gene_tgs = tgs[:, gene]
        gene_tg_outputs = tg_outputs[:, gene]
        
        # 计算皮尔逊相关系数
        corr_coef = np.corrcoef(gene_tgs, gene_tg_outputs)[0, 1]
        pearson_corrs.append(corr_coef)

    pearson_corrs = np.nan_to_num(pearson_corrs, nan=0.0)

    # 计算均值
    mean_corr = np.mean(pearson_corrs)
    print(f"Average Pearson R² Score: {mean_corr}")
    print(f"Average R² Score: {np.mean(r2_scores)}")

    # 绘制每个基因的 R² 分布图
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=gene_means, y=r2_scores)
    plt.axhline(y=np.mean(r2_scores), color='red', linestyle='--', label=f'Mean R² Score: {np.mean(r2_scores):.2f}')
    plt.legend()
    plt.title("R² Score vs Gene Expression Mean")
    plt.xlabel("Gene Expression Mean")
    plt.ylabel("R² Score")
    # 添加红色的均值文本

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "r2_scores_vs_gene_mean.png"), dpi=300, bbox_inches="tight")
        print(f"R² Score图已保存到 {os.path.join(save_path, 'r2_scores_vs_gene_mean.png')}")
    else:
        plt.show()

    # 绘制每个基因的皮尔逊相关系数图
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=gene_means, y=pearson_corrs)
    plt.axhline(y=np.mean(pearson_corrs), color='red', linestyle='--', label=f'Mean Pearson R² Score: {np.mean(pearson_corrs):.2f}')
    plt.legend()
    plt.title(" Pearson R² Score vs Gene Expression Mean")
    plt.xlabel("Gene Expression Mean")
    plt.ylabel("pearson R² Score")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "pearson_corrs_vs_gene_mean.png"), dpi=300, bbox_inches="tight")
        print(f"pearson R² Score图已保存到 {os.path.join(save_path, 'r2_scores_vs_gene_mean.png')}")
    else:
        plt.show()

    # 绘制基因的真实表达均值与预测表达均值的关系图
    r_squared = np.corrcoef(gene_means, y=predicted_means)[0, 1]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=gene_means, y=predicted_means)
    plt.title("Predicted Gene Expression Mean vs True Gene Expression Mean")
    plt.xlabel("True Gene Expression Mean")
    plt.ylabel("Predicted Gene Expression Mean")
    plt.plot([min(gene_means), max(gene_means)], [min(gene_means), max(gene_means)], color='red', linestyle='--', label='y=x')
    plt.text(min(gene_means), max(predicted_means), f'R² = {r_squared:.2f}', fontsize=12)
    plt.legend()

    if save_path:
        plt.savefig(os.path.join(save_path, "predicted_vs_true_means.png"), dpi=300, bbox_inches="tight")
        print(f"Average R² Score saved to {os.path.join(save_path, 'predicted_vs_true_means.png')}")
    else:
        plt.show()

    
def visualize_tf_acts(tf_acts, coords, gene_index=0, save_path=None):
    """
    可视化特定基因的 tf_act 值在空间坐标上的分布。

    参数:
    - tf_acts: tf_act 值
    - coords: 细胞坐标
    - gene_index: 要可视化的基因索引（默认为 0）
    - save_path: 保存图片的路径（默认为 None，不保存）
    """
    # 选择一个特定的基因进行可视化
    gene_tf_act = tf_acts[:, gene_index]

    # 创建散点图
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=coords[:, 0], y=coords[:, 1], 
                            size=gene_tf_act, hue=gene_tf_act, palette='viridis', legend=None)

    # 添加颜色条
    cbar = plt.colorbar(scatter.collections[0], label="tf_act")
    cbar.set_label("tf_act")

    # 设置标题和坐标轴标签
    plt.title(f"Visualization of tf_act for Gene {gene_index}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"tf_act_gene_{gene_index}.png"), dpi=300, bbox_inches="tight")
        print(f"tf_act plot for Gene {gene_index} saved to {os.path.join(save_path, f'tf_act_gene_{gene_index}.png')}")
    else:
        plt.show()





