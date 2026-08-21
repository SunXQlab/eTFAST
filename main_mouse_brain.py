import os
import pickle
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from models.input_prepare import *
from models.train_model import *
from models.evalueate_model import *
from models.utils import *
# from tools.plot import visualize_receivers

# 设置随机种子
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

# 全局路径
DATA_DIR = "./data/"
MODEL_DIR = "./results_human_OV/model_params/"
TG_PRED_DIR = "./results_human_OV/tg_prediction/"
MORAN_DIR = "./results_human_OV/moran_i/"
LOSS_DIR = "./results_human_OV/loss_curves/"
VISUALIZE_DIR = './results_human_OV/visualize/'

# 创建目录
for dir_path in [DATA_DIR, MODEL_DIR, TG_PRED_DIR, MORAN_DIR, LOSS_DIR, VISUALIZE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def filter_receivers_by_cell_count(adata, min_cells=200, max_cells=500):
    """根据细胞数量划分receivers"""
    cell_type_counts = adata.obs['cell_type'].value_counts()
    
    # 单独训练的receivers（>500个细胞）
    single_receivers = cell_type_counts[cell_type_counts > max_cells].index.tolist()
    
    # 合并训练的receivers（200-500个细胞）
    grouped_receivers = cell_type_counts[(cell_type_counts >= min_cells) & 
                                       (cell_type_counts <= max_cells)].index.tolist()
    
    return single_receivers, grouped_receivers

def train_and_evaluate(adata, senders, receivers, is_grouped=False):
    """训练模型并评估结果"""
    cell_types = pd.DataFrame({'cell_ID': adata.obs_names, 'cell_type': adata.obs['cell_type']})
    coords = pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)
    
    # 准备数据集
    batch_size = 64
    dataset, TFTG_matrix = prepare_lig_rec_tf_tg_dataset(adata, cell_types, coords, receivers)
    #保存数据集
    if is_grouped:
        save_name = "grouped_dataset.pkl"
    else:
        save_name = f"{'_'.join(receivers)}.pkl"
    #保存数据集
    with open(os.path.join(DATA_DIR, f"dataset_{save_name}"), "wb") as f:
        pickle.dump(dataset, f)
    with open(os.path.join(DATA_DIR, f"TFTG_matrix_{save_name}"), "wb") as f:
        pickle.dump(TFTG_matrix, f)

    #加载数据集
    with open(os.path.join(DATA_DIR, f"dataset_{save_name}"), "rb") as f:
        dataset = pickle.load(f)
    with open(os.path.join(DATA_DIR, f"TFTG_matrix_{save_name}"), "rb") as f:
        TFTG_matrix = pickle.load(f)

    pretrain_loader, all_loader = prepare_dataloaders(dataset, receivers, batch_size)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGPredictionModel(
        lig_rec_dim=dataset[receivers[0]].diff_neighbor_ligand.shape[2] + dataset[receivers[0]].cont_neighbor_ligand.shape[2],
        diff_lig_dim=dataset[receivers[0]].diff_Rec_df.shape[1],
        hidden_dim_1=64,
        tf_dim_dict={receiver: dataset[receiver].TF_df.shape[1] for receiver in receivers},
        tg_dim_dict={receiver: dataset[receiver].TG_df.shape[1] for receiver in receivers},
        receivers=receivers,
        TFTG_matrix_dict=TFTG_matrix
    ).to(device)
    
    # 训练流程
    pretrain_model(model, pretrain_loader, receivers, epochs=20, lr=1e-4, device=device, save_fig_path=LOSS_DIR)
    estimated_mean, estimated_fisher = estimate_ewc_params(model, all_loader, receivers)
    train_model_with_ewc(model, all_loader, all_loader, receivers, estimated_mean, estimated_fisher, 
                         epochs=60, lr=1e-3, device=device, lambda_ewc=1e2, save_fig_path=LOSS_DIR)
    
    # 保存模型
    model_name = "grouped_model.pth" if is_grouped else f"model_{'_'.join(receivers)}.pth"
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))
    
    #加载模型

    model_name = "grouped_model.pth" if is_grouped else f"model_{'_'.join(receivers)}.pth"
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_name)))
    
    # 预测和评估
    # tg_outputs, tf_acts, tgs, coords_receiver, tf_grads = predict_model(model, dataset, receivers, batch_size, device=device, return_gradients=True)
    (lr_scores, tf_acts, tg_outputs, tgs, coords_receiver,lr_tf_imp, tf_tg_imp) = predict_model(model, dataset, receivers)
    #转化为表格
    lr_scores_df ={receiver:pd.DataFrame(lr_scores[receiver], index=dataset[receiver].diff_Rec_df.index, columns=dataset[receiver].diff_Rec_df.columns.tolist()+dataset[receiver].cont_Rec_df.columns.tolist()) for receiver in receivers}
    tgs_df ={receiver:pd.DataFrame(tgs[receiver], index=dataset[receiver].TG_df.index, columns=dataset[receiver].TG_df.columns) for receiver in receivers}
    tg_outputs_df ={receiver:pd.DataFrame(tg_outputs[receiver], index=dataset[receiver].TG_df.index, columns=dataset[receiver].TG_df.columns) for receiver in receivers}
    tfs_df ={receiver:pd.DataFrame(dataset[receiver].TF_df, index=dataset[receiver].TF_df.index, columns=dataset[receiver].TF_df.columns) for receiver in receivers}
    tf_acts_df ={receiver:pd.DataFrame(tf_acts[receiver], index=dataset[receiver].TF_df.index, columns=dataset[receiver].TF_df.columns) for receiver in receivers}
    lr_imp_for_tf_df ={receiver:pd.DataFrame(lr_tf_imp[receiver],index=dataset[receiver].TF_df.columns,columns=dataset[receiver].diff_Rec_df.columns.tolist()+dataset[receiver].cont_Rec_df.columns.tolist()) for receiver in receivers}
    tf_imp_for_tg_df ={receiver:pd.DataFrame(tf_tg_imp[receiver],index=dataset[receiver].TG_df.columns,columns=dataset[receiver].TF_df.columns) for receiver in receivers}
    # R2_results = compute_correlations(tg_outputs_df,tgs_df, receivers)

    # visualize_R2_scores(R2_results)

    # distance_mat = compute_distance_matrix(coords)
    # # 提取 LR 对
    # LR_pairs_diff = dataset[receivers[0]].diff_Rec_df.columns
    # LR_pairs_cont = dataset[receivers[0]].cont_Rec_df.columns
    # LR_pairs = LR_pairs_diff.tolist() + LR_pairs_cont.tolist()


    # LR_pairs_permulated = permutation_test(model,distance_mat,adata,
    #                                        senders,receivers,
    #                                        LR_pairs_diff,LR_pairs_cont,
    #                                        num_permutations=100,n_jobs=4)
    # if is_grouped:
    #     LR_pairs_permulated.to_csv(os.path.join(MODEL_DIR, f"LR_pairs_permulated_grouped.csv"), index=False)
    # else:
    #     LR_pairs_permulated.to_csv(os.path.join(MODEL_DIR, f"LR_pairs_permulated_{'_'.join(receivers)}.csv"), index=False)
    # LR_pairs_permulated.to_csv(os.path.join(MODEL_DIR, f"LR_pairs_permulated_{'_'.join(receivers)}.csv"), index=False)


    results = []
    for receiver in receivers:
        result={
            'receiver': receiver,
            'lr_scores': lr_scores_df[receiver],
            'lr_imp_for_tf': lr_imp_for_tf_df[receiver],
            'tf_imp_for_tg': tf_imp_for_tg_df[receiver],
            'tfs': tfs_df[receiver],
            'tf_acts': tf_acts_df[receiver],
            'tg_outputs': tg_outputs_df[receiver],
            'tgs': tgs_df[receiver],
            'coords': coords_receiver[receiver]
        }
        results.append(result)
    return results

    

def main(adata_path):
    # 加载数据
    adata = ad.read_h5ad(adata_path)
    senders = adata.obs['cell_type'].value_counts().head(15).index.tolist()
    # 根据细胞数量划分receivers
    single_receivers, grouped_receivers = filter_receivers_by_cell_count(adata,min_cells=200, max_cells=500)
    print(f"单独训练的receivers: {single_receivers}")
    print(f"合并训练的receivers: {grouped_receivers}")
    
    # 存储所有结果
    all_results = []

    # receiver = 'Isocortex L6'
    # print(f"\n=== 训练单独receiver: {receiver} ===")
    # result = train_and_evaluate(adata,senders, [receiver])
    # all_results.extend(result)

    # results_dict = {res['receiver']: res for res in all_results}

    # with open(os.path.join(TG_PRED_DIR, "results_dict_L6.pkl"), "wb") as f:
    #     pickle.dump(results_dict, f)


    # 1. 训练单独receivers
    for receiver in single_receivers:  # 只训练第一个receiver
        print(f"\n=== 训练单独receiver: {receiver} ===")
        result = train_and_evaluate(adata,senders, [receiver])
        all_results.extend(result)
    
    # 2. 训练合并的receivers
    if grouped_receivers:
        print(f"\n=== 训练合并的receivers: {grouped_receivers} ===")
        result = train_and_evaluate(adata, senders,grouped_receivers, is_grouped=True)
        all_results.extend(result)

    results_dict = {res['receiver']: res for res in all_results}

    with open(os.path.join(TG_PRED_DIR, "results_dict.pkl"), "wb") as f:
        pickle.dump(results_dict, f)

    # with open(os.path.join(TG_PRED_DIR, "results_dict.pkl"), "rb") as f:
    #     results_dict = pickle.load(f)



if __name__ == "__main__":
    adata_path = os.path.join(DATA_DIR, "2_human_OV/transcriptome/human_OV_processed_sub.h5ad")
    main(adata_path)