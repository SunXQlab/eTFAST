from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
from esda.moran import Moran
import libpysal as lib
import seaborn as sns
import matplotlib.pyplot as plt
from models.input_prepare import *
from tqdm import tqdm


def _single(gene, TF_df, W, permutations):

    cur_X = TF_df[gene].values  # 提取当前基因的表达值
    mbi = Moran(cur_X, W, permutations=permutations, two_tailed=False)
    Moran_I = mbi.I
    p_value = mbi.p_sim
    z_score = mbi.z_sim
    return [gene, Moran_I, p_value, z_score]

def compute_morans_i(TF_df, coords, k=5, permutations=199, n_jobs=4):

    kd = lib.cg.KDTree(np.array(coords))
    nw = lib.weights.KNN(kd, k)
    weight = lib.weights.W(nw.neighbors, nw.weights)

    genes = TF_df.columns  # 获取所有基因名称
    res = Parallel(n_jobs=n_jobs)(delayed(_single)(gene, TF_df, weight, permutations) for gene in genes)
    res = pd.DataFrame(res, columns=["gene", "moran_i", "moran_p_val", "moran_z"])
    res.set_index("gene", inplace=True)  # 将基因名称设置为索引
    res["moran_q_val"] = multipletests(res["moran_p_val"], method="fdr_bh")[1]  # 计算多重假设检验校正后的 q 值
    return res
def permutation_test(model, distance_mat, adata, senders, receivers, 
                    LR_pairs_diff, LR_pairs_cont, num_permutations=100, n_jobs=4):
    """执行高效置换检验评估LR对重要性"""
    
    # 1. 基础数据准备
    cell_types = pd.DataFrame({'cell_ID': adata.obs_names, 
                              'cell_type': adata.obs['cell_type']})
    
    # 2. 计算原始得分（单次)
    original_scores = model.calculate_LRscore_coords(
        adata, cell_types, distance_mat, senders, receivers,
        LR_pairs_diff, LR_pairs_cont, shuffle=False
    )
    
    # 3. 并行化置换检验
    def _permute_and_score(seed):
        np.random.seed(seed)
        return model.calculate_LRscore_coords(
            adata, cell_types, distance_mat, senders, receivers,
            LR_pairs_diff, LR_pairs_cont, shuffle=True
        )
    
    # 使用joblib并行计算
    permuted_scores_list = Parallel(n_jobs=n_jobs)(
        delayed(_permute_and_score)(i) 
        for i in tqdm(range(num_permutations), desc="Running permutations")
    )
    
    # 4. 重组置换结果
    permuted_scores_dict = {k: [] for k in original_scores}
    for perm in permuted_scores_list:
        for k in perm:
            permuted_scores_dict[k].append(perm[k].iloc[0])
    
    # 5. 构建结果DataFrame并计算p值
    results = []
    for sender_recv, orig_df in original_scores.items():
        sender, receiver = sender_recv.split('_')
        perm_scores = pd.concat(permuted_scores_dict[sender_recv], axis=1).T
        
        for lr_pair, orig_score in orig_df.iloc[0].items():
            # 计算p值（向量化操作)
            p_val = (np.abs(perm_scores[lr_pair]) >= np.abs(orig_score)).mean()
            
            results.append({
                'sender_receiver': sender_recv,
                'sender': sender,
                'receiver': receiver,
                'lr_pair': lr_pair,
                'score': orig_score,
                'p_value': p_val
            })
    
    return pd.DataFrame(results)

def compute_correlations(tg_outputs_df, tgs_df, receivers):
    """
    可视化每个基因的 R² 分数和皮尔逊相关系数。

    参数:
    - tg_outputs_df: 预测结果的 DataFrame,每个键是 receiver,每个键对应的值是该 receiver 的 tg_outputs 的值
    - tgs_df: 真实值的 DataFrame,每个键是 receiver,每个键对应的值是该 receiver 的 tgs 的值
    - receivers: 接收器类型列表
    """

    # 初始化结果 DataFrame
    R2_results_df = pd.DataFrame(columns=['Gene', 'Receiver', 'True_Exp_Mean', 'Predicted_Mean', 'Pearson_Corr', 'R2_Score'])

    # 遍历每个 receiver
    for receiver in receivers:
        tg_outputs = tg_outputs_df[receiver]
        tgs = tgs_df[receiver]

        # 计算每个基因的 R² 分数
        num_genes = tgs.shape[1]
        r2_scores = r2_scores = [max(0, 1 - (np.sum((tg_outputs.iloc[:, gene] - tgs.iloc[:, gene]) ** 2) / np.sum((tgs.iloc[:, gene] - np.mean(tgs.iloc[:, gene])) ** 2))) for gene in range(num_genes)]        
        gene_means = [np.mean(tgs.iloc[:, gene]) for gene in range(num_genes)]
        predicted_means = [np.mean(tg_outputs.iloc[:, gene]) for gene in range(num_genes)]

        pearson_corrs = []

        for gene in range(num_genes):
            # 提取单个基因的数据
            gene_tgs = tgs.iloc[:, gene]
            gene_tg_outputs = tg_outputs.iloc[:, gene]
            
            # 计算皮尔逊相关系数
            corr_coef = np.corrcoef(gene_tgs, gene_tg_outputs)[0, 1]
            pearson_corrs.append(corr_coef)

        pearson_corrs = np.nan_to_num(pearson_corrs, nan=0.0)

        # 计算均值
        mean_corr = np.mean(pearson_corrs)
        print(f"Average Pearson R² Score for {receiver}: {mean_corr}")
        print(f"Average R² Score for {receiver}: {np.mean(r2_scores)}")

        # 将结果添加到 DataFrame
        temp_df = pd.DataFrame({
            'Gene': tgs.columns,  
            'Receiver': [receiver] * num_genes,
            'True_Exp_Mean': gene_means,
            'Predicted_Mean': predicted_means,
            'Pearson_Corr': pearson_corrs,
            'R2_Score': r2_scores
        })
        R2_results_df = pd.concat([R2_results_df, temp_df], ignore_index=True)
    return R2_results_df
def visualize_R2_scores(R2_results_df, save_path=None):
    plt.figure(figsize=(12, 8))
    # 使用seaborn绘制箱线图
    ax = sns.boxplot(
        x='Receiver', 
        y='R2_Score',
        data=R2_results_df,
        showfliers=False,  # 不显示异常值
        width=0.6
    )
    # 图表修饰
    plt.title('R² Score Distribution by Receiver Type', fontsize=24, pad=20)
    plt.xlabel('Receiver Cell Type', fontsize=20)
    plt.ylabel('R² Score', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=18)  # 旋转x轴标签
    plt.yticks(fontsize=18)  # 设置y轴标签字体大小
    # 调整布局
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "R2_scores.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.figure(figsize=(12, 8))
    # 使用seaborn绘制箱线图
    ax = sns.boxplot(
        x='Receiver', 
        y='Pearson_Corr',
        data=R2_results_df,
        showfliers=False,  # 不显示异常值
        width=0.6
    )
    # 图表修饰
    plt.title('Pearson Correlation Distribution by Receiver Type', fontsize=24, pad=20)
    plt.xlabel('Receiver Cell Type', fontsize=20)
    plt.ylabel('Pearson Correlation', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=18)  # 旋转x轴标签
    plt.yticks(fontsize=18)  # 设置y轴标签字体大小
    # 调整布局
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "Pearson Correlation.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
def visualize_tf_acts(tf_acts, tfs,receiver,coords_receiver, coords, gene_name=0, save_path=None):
    """
    可视化特定基因的 tf_act 值和 tf 值在空间坐标上的分布。
    
    参数:
    - tf_acts: tf_act 值
    - coords: 细胞坐标
    - gene_name: 要可视化的基因名称或索引（默认为 0)
    - save_path: 保存图片的路径（默认为 None,不保存)
    """
    # 选择一个特定的基因进行可视化
    gene_tf_act = tf_acts[receiver][gene_name]
    gene_tf = tfs[receiver][gene_name]

    # 创建一个包含两个子图的图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：tf_act 值的可视化
    axes[0].scatter(coords.iloc[:, 0], coords.iloc[:, 1], color='gray', s=5, alpha=0.8)  # 绘制灰色底图
    sns.scatterplot(x=coords_receiver[:, 0], y=coords_receiver[:, 1], 
                    size=gene_tf_act.values, hue=gene_tf_act.values, palette='viridis', legend=None, ax=axes[0])
    cbar1 = plt.colorbar(axes[0].collections[0], ax=axes[0], label="tf_act")
    axes[0].set_title(f"TF_act for Gene {gene_name}")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")

    # 右图：TF 值的可视化
    axes[1].scatter(coords.iloc[:, 0], coords.iloc[:, 1], color='gray', s=5, alpha=0.8)  # 绘制灰色底图
    sns.scatterplot(x=coords_receiver[:, 0], y=coords_receiver[:, 1], 
                    size=gene_tf.values, hue=gene_tf.values, palette='viridis', legend=None, ax=axes[1])
    cbar2 = plt.colorbar(axes[1].collections[0], ax=axes[1], label="TF")
    axes[1].set_title(f"TF_exp for Gene {gene_name}")
    axes[1].set_xlabel("X Coordinate")
    axes[1].set_ylabel("Y Coordinate")



    # 自动调整子图间距
    plt.tight_layout()

    # 如果提供了保存路径,保存图像,否则显示图像
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"tf_act_and_tf_gene_{gene_name}.png"), dpi=300, bbox_inches="tight")
        print(f"tf_act and TF plot for Gene {gene_name} saved to {os.path.join(save_path, f'tf_act_and_tf_gene_{gene_name}.png')}")
    else:
        plt.show()