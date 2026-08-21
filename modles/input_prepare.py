import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import  MinMaxScaler
import os
import pandas as pd
import anndata as ad
import scanpy as sc
import os
import pandas as pd
import anndata as ad
from models.train_model import TGDataset
import pickle
from scipy import sparse
import numpy as np
from scipy.spatial.distance import cdist
#######data_preprocess##########


def preprocess_data(DATA_DIR):
    """Preprocess the raw data from DATA_DIR and save the preprocessed data to DATA_DIR"""
    counts_matrix = pd.read_csv(os.path.join(DATA_DIR, "counts_raw.csv"), index_col=0)
    cell_metadata = pd.read_csv(os.path.join(DATA_DIR, "cell_metadata.csv"), index_col=0)
    gene_info = pd.read_csv(os.path.join(DATA_DIR, "gene_info.csv"), index_col=1)

    adata = ad.AnnData(
        X=counts_matrix.values.T,
        obs=cell_metadata,
        var=gene_info
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.write(os.path.join(DATA_DIR, "MERSCOPE_prostate_sub.h5ad"))
    print("Data preprocessing completed.")

def load_preprocessed_data(DATA_DIR):
    """Load the preprocessed data"""
    adata = ad.read_h5ad(os.path.join(DATA_DIR, "MERSCOPE_prostate_sub.h5ad"))
    cell_types = adata.obs['orig.annotation'].values
    coordinates = np.column_stack((adata.obs['center_x'], adata.obs['center_y']))
    return adata, cell_types, coordinates

def prepare_lig_rec_tf_tg_dataset(adata, cell_types, coordinates, receivers=None):
    """Prepare the train dataset and TFTG matrix"""
    print('Computing the distance matrix')
    distance_matrix = compute_distance_matrix(coordinates)  # Normalize the distance matrix

    # Initialize dictionaries to store dataframes and matrices
    dataset = {}
    TFTG_matrix = {}
    diff_ligand_dfs = {}
    cont_ligand_dfs = {}
    diff_receptor_dfs = {}
    cont_receptor_dfs = {}
    tf_dfs = {}
    tg_dfs = {}
    diff_lig_rec_columns = set()
    cont_lig_rec_columns = set()
    for receiver in receivers:
        print(f"Preparing dataset and TFTG matrix for receiver {receiver}")
        (
            diff_ligand_df,
            diff_receptor_df,
            cont_ligand_df,
            cont_receptor_df,
            tf_dfs[receiver],
            tg_dfs[receiver],
            TFTG_matrix[receiver]
        ) = extract_lig_rec_tf_tg(
            adata, cell_types, receiver=receiver,
            avg_expr_lr=0.01, avg_expr_tg=0.1,
            percent_cells_lr=0.05, percent_cells_tg=0.1, normalize=True
        )
        diff_ligand_dfs[receiver] = diff_ligand_df
        diff_receptor_dfs[receiver] = diff_receptor_df
        cont_ligand_dfs[receiver] = cont_ligand_df
        cont_receptor_dfs[receiver] = cont_receptor_df
        diff_lig_rec_columns.update(diff_ligand_df.columns)
        cont_lig_rec_columns.update(cont_ligand_df.columns)

    for receiver in receivers:
        current_diff_ligand_df = diff_ligand_dfs[receiver]
        current_diff_receptor_df = diff_receptor_dfs[receiver]
        standardized_diff_ligand_df = pd.DataFrame(
            0, index=current_diff_ligand_df.index, columns=sorted(diff_lig_rec_columns)
        )
        standardized_diff_receptor_df = pd.DataFrame(
            0, index=current_diff_receptor_df.index, columns=sorted(diff_lig_rec_columns)
        )
        for column in diff_lig_rec_columns:
            if column in current_diff_ligand_df.columns:
                standardized_diff_ligand_df[column] = current_diff_ligand_df[column]
                standardized_diff_receptor_df[column] = current_diff_receptor_df[column]
        diff_ligand_dfs[receiver] = standardized_diff_ligand_df
        diff_receptor_dfs[receiver] = standardized_diff_receptor_df

        current_cont_ligand_df = cont_ligand_dfs[receiver]
        current_cont_receptor_df = cont_receptor_dfs[receiver]
        standardized_cont_ligand_df = pd.DataFrame(
            0, index=current_cont_ligand_df.index, columns=sorted(cont_lig_rec_columns)
        )
        standardized_cont_receptor_df = pd.DataFrame(
            0, index=current_cont_receptor_df.index, columns=sorted(cont_lig_rec_columns)
        )
        for column in cont_lig_rec_columns:
            if column in current_cont_ligand_df.columns:
                standardized_cont_ligand_df[column] = current_cont_ligand_df[column]
                standardized_cont_receptor_df[column] = current_cont_receptor_df[column]
        cont_ligand_dfs[receiver] = standardized_cont_ligand_df
        cont_receptor_dfs[receiver] = standardized_cont_receptor_df

        print(f'Extracting neighbor ligands and distances for receiver: {receiver}')
        (
            diff_neighbor_ligands,
            diff_neighbor_distances,
            cont_neighbor_ligands,
            cont_neighbor_distances,
            receiver_distances,
            receiver_coordinates
        ) = extract_neighbor_info(
            coordinates,
            distance_matrix,
            diff_ligand_dfs[receiver].values,
            cont_ligand_dfs[receiver].values,
            cell_types,
            sender=None,
            receiver=receiver,
            num_neighbors=100
        )
        dataset[receiver] = TGDataset(
            diff_neighbor_ligands,
            cont_neighbor_ligands,
            diff_receptor_dfs[receiver],
            cont_receptor_dfs[receiver],
            tf_dfs[receiver],
            diff_neighbor_distances,
            cont_neighbor_distances,
            receiver_distances,
            tg_dfs[receiver],
            receiver_coordinates,
            receiver_type=receiver
        )
    return dataset, TFTG_matrix


def save_dataset(dataset, TFTG_matrix, DATA_DIR):
    """Save the dataset and TFTG matrix"""
    dataset_path = os.path.join(DATA_DIR, f"dataset.pkl")
    TFTG_matrix_path = os.path.join(DATA_DIR, f"TFTG_matrix.pkl")
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    with open(TFTG_matrix_path, "wb") as f:
        pickle.dump(TFTG_matrix, f)

    print(f"Dataset saved to {dataset_path} and TFTG_matrix saved to {TFTG_matrix_path}")

def load_dataset(DATA_DIR):
    """Load the saved dataset"""
    dataset_path = os.path.join(DATA_DIR, f"dataset.pkl")
    TFTG_matrix_path = os.path.join(DATA_DIR, f"TFTG_matrix.pkl")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    with open(TFTG_matrix_path, "rb") as f:
        TFTG_matrix = pickle.load(f)
    return dataset, TFTG_matrix

#######extract_lig_rec_tf_tg##########
def extract_highly_expressed_genes(adata, cells, avg_expr_threshold=0.05, percent_cells_threshold=0.1):
    """
    Extract highly expressed genes in a given cell type, sender, or receiver.
    If cell type, sender, or receiver is None, screen in all cells.

    Parameters:
        adata (AnnData): Annotated data object.
        cells (array-like): Boolean array indicating cells to consider.
        avg_expr_threshold (float): Average expression threshold.
        percent_cells_threshold (float): Percentage of cells threshold.

    Returns:
        set: Set of highly expressed genes.
    """
    expression_matrix = adata[cells, :].X
    avg_expression = expression_matrix.mean(axis=0)
    expr_percent = (expression_matrix > 0).mean(axis=0)
    high_expr_genes =adata[cells, (avg_expression > avg_expr_threshold) & (expr_percent > percent_cells_threshold)].var_names
    return set(high_expr_genes)

def extract_Lig_Rec_df(adata, sender, receiver, LigRecDB_path='./data/data_utils/LigRecDB.csv', avg_expr_lr=0.05, percent_cells_lr=0.1):
    """
    Extract ligand-receptor expression data.
    Ligand gene screening is based on sender cells, receptor gene screening is based on receiver cells.

    Parameters:
        adata (AnnData): Annotated data object.
        sender (array-like): Boolean array indicating sender cells.
        receiver (array-like): Boolean array indicating receiver cells.
        LigRecDB_path (str): Path to Ligand-Receptor database.
        avg_expr_lr (float): Average expression threshold for ligands and receptors.
        percent_cells_lr (float): Percentage of cells threshold for ligands and receptors.

    Returns:
        tuple: Ligand expression DataFrame, Receptor expression DataFrame.
    """
    high_expr_Lig = extract_highly_expressed_genes(adata, sender, avg_expr_lr, percent_cells_lr)
    high_expr_Rec = extract_highly_expressed_genes(adata, receiver, avg_expr_lr, percent_cells_lr)
    LigRecDB = pd.read_csv(LigRecDB_path)
    valid_Lig_Rec = LigRecDB[(LigRecDB['source'].isin(high_expr_Lig)) & (LigRecDB['target'].isin(high_expr_Rec))]
    Lig_expression = adata[sender, valid_Lig_Rec['source']].to_df()
    Rec_expression = adata[receiver, valid_Lig_Rec['target']].to_df()
    Lig_expression.columns = [f"{source}_{target}" for source, target in zip(valid_Lig_Rec['source'], valid_Lig_Rec['target'])]
    Rec_expression.columns = [f"{source}_{target}" for source, target in zip(valid_Lig_Rec['source'], valid_Lig_Rec['target'])]

    return Lig_expression, Rec_expression


def extract_TF_TG_df(adata, receiver, TFTGDB_path='./data/data_utils/TFTGDB.csv', avg_expr_tg=0.1, percent_cells_tg=0.1):
    """
    Extract transcription factor-target gene data.
    Transcription factors and target genes are screened in receiver cells.

    Parameters:
        adata (AnnData): Annotated data object.
        receiver (array-like): Boolean array indicating receiver cells.
        TFTGDB_path (str): Path to TF-TG database.
        avg_expr_tg (float): Average expression threshold for target genes.
        percent_cells_tg (float): Percentage of cells threshold for target genes.

    Returns:
        tuple: TF expression DataFrame, TG expression DataFrame, TF-TG interaction matrix.
    """
    high_expr_TF = extract_highly_expressed_genes(adata, receiver, 0, 0)  # No threshold for TFs
    high_expr_TG = extract_highly_expressed_genes(adata, receiver, avg_expr_tg, percent_cells_tg)
    TFTGDB = pd.read_csv(TFTGDB_path)
    valid_TF_TG = TFTGDB[(TFTGDB['target'].isin(high_expr_TG)) & (TFTGDB['source'].isin(high_expr_TF))]
    TFTG_matrix = pd.crosstab(valid_TF_TG['source'], valid_TF_TG['target'])
    TFTG_matrix[TFTG_matrix > 0] = 1
    TF_expression = adata[receiver, valid_TF_TG['source'].unique()].to_df()
    TG_expression = adata[receiver, valid_TF_TG['target'].unique()].to_df()
    TFTG_matrix = TFTG_matrix.reindex(index=TF_expression.columns, columns=TG_expression.columns, fill_value=0)
    return TF_expression, TG_expression, TFTG_matrix

def min_max_normalize(df):
    """
    Normalize gene expression matrix using Min-Max scaling.

    Parameters:
        df (DataFrame): Gene expression DataFrame.

    Returns:
        DataFrame: Normalized gene expression DataFrame.
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns, index=df.index)


def extract_lig_rec_tf_tg(adata, cell_types, sender=None, receiver=None, avg_expr_lr=0.05, avg_expr_tg=0.05, 
                          percent_cells_lr=0.1, percent_cells_tg=0.1, normalize=False):
    """
    Extract ligand-receptor and transcription factor-target gene data.

    Parameters:
        adata (AnnData): Annotated data object.
        cell_types (array-like): Cell type annotations.
        sender (str, optional): Sender cell type. Defaults to None.
        receiver (str, optional): Receiver cell type. Defaults to None.
        avg_expr_lr (float): Average expression threshold for ligands and receptors.
        avg_expr_tg (float): Average expression threshold for target genes.
        percent_cells_lr (float): Percentage of cells threshold for ligands and receptors.
        percent_cells_tg (float): Percentage of cells threshold for target genes.
        normalize (bool): Whether to normalize the data. Defaults to False.

    Returns:
        tuple: Processed ligand, receptor, TF, TG DataFrames, and TF-TG interaction matrix.
    """
    if sender:
        sender_cells = cell_types['cell_type'] == sender
    else:
        sender_cells = cell_types['cell_type'] == cell_types['cell_type']

    if receiver:
        receiver_cells = cell_types['cell_type'] == receiver
    else:
        receiver_cells = cell_types['cell_type'] == cell_types['cell_type']

    # Extract ligand-receptor data
    diff_Lig_df, diff_Rec_df = extract_Lig_Rec_df(adata, sender_cells, receiver_cells,
                                                  LigRecDB_path='./data/data_utils/diff_LigRecDB.csv',
                                                  avg_expr_lr=avg_expr_lr, percent_cells_lr=percent_cells_lr)
    cont_Lig_df, cont_Rec_df = extract_Lig_Rec_df(adata, sender_cells, receiver_cells,
                                                  LigRecDB_path='./data/data_utils/cont_LigRecDB.csv',
                                                  avg_expr_lr=avg_expr_lr, percent_cells_lr=percent_cells_lr)

    # Extract TF-TG data
    TF_df, TG_df, TFTG_matrix = extract_TF_TG_df(adata, receiver_cells, avg_expr_tg=avg_expr_tg, percent_cells_tg=percent_cells_tg)

    print(diff_Lig_df.shape, cont_Lig_df.shape, TF_df.shape, TG_df.shape, TFTG_matrix.shape)
    print("Number of sender cells in diff_Lig_df:", diff_Lig_df.shape[0])
    print("Number of receiver cells in cont_Lig_df:", diff_Rec_df.shape[0])
    print("Number of diffussion-type LR pairs in diff_Lig_df:", diff_Lig_df.shape[1])
    print("Number of contact-type LR pairs in cont_Lig_df:", cont_Lig_df.shape[1])
    print("Number of TFs:", TF_df.shape[1])
    print("Number of TGs:", TG_df.shape[1])

    # Normalize data if required
    if normalize:
        print('Applying Min-Max normalization to the dataset')
        diff_Lig_df = min_max_normalize(diff_Lig_df)
        diff_Rec_df = min_max_normalize(diff_Rec_df)
        cont_Lig_df = min_max_normalize(cont_Lig_df)
        cont_Rec_df = min_max_normalize(cont_Rec_df)
        TF_df = min_max_normalize(TF_df)
        TG_df = min_max_normalize(TG_df)
    else:
        print('No normalization applied to the dataset')

    return diff_Lig_df, diff_Rec_df, cont_Lig_df, cont_Rec_df, TF_df, TG_df, TFTG_matrix





########compute_distance_matrix##########
def compute_distance_matrix(coordinates):
    """
    Compute the distance matrix and normalize it based on the mean of the smallest distances.

    Parameters:
        coordinates (array-like): Cell coordinates (n_cells, 2).

    Returns:
        array: Normalized distance matrix.
    """
    distance_matrix = cdist(coordinates, coordinates, metric='euclidean').astype(np.float16)
    sum_min = np.sum(np.partition(distance_matrix, 1, axis=1)[:, :2], axis=1)
    mean_sum_min = np.mean(sum_min)
    print("Normalization unit (mean):", mean_sum_min)
    return np.where(distance_matrix < mean_sum_min, 1, distance_matrix / mean_sum_min)

def extract_neighbor_info(cell_coordinates, distance_mat,diff_Lig_mat, cont_Lig_mat, cell_types, sender, receiver, num_neighbors=500):
    """
    Extract neighbor information for each receiver cell.

    Parameters:
        cell_coordinates (array-like): Cell coordinates (n_cells, 2).
        diff_Lig_mat (DataFrame): Diffussion ligand expression matrix.
        cont_Lig_mat (DataFrame): Contact ligand expression matrix.
        cell_types (array-like): Cell type annotations.
        sender (str, optional): Sender cell type. Defaults to None.
        receiver (str, optional): Receiver cell type. Defaults to None.
        num_neighbors (int): Number of neighbors to consider. Defaults to 500.

    Returns:
        tuple: Neighbor ligand information, neighbor distances, and receiver distances.
    """

    if sender:
        sender_cells = cell_types['cell_type'] == sender
    else:
        sender_cells = cell_types['cell_type'] == cell_types['cell_type']

    if receiver:
        receiver_cells = cell_types['cell_type'] == receiver
    else:
        receiver_cells = cell_types['cell_type'] == cell_types['cell_type']

    diff_Lig_mat =diff_Lig_mat.astype(np.float16)
    cont_Lig_mat = cont_Lig_mat.astype(np.float16)
    neighbors_distance_mat = distance_mat[receiver_cells, :][:, sender_cells]
    receivers_distance_mat = distance_mat[receiver_cells, :][:, receiver_cells]
    receiver_coords = cell_coordinates[receiver_cells]

    num_receivers = neighbors_distance_mat.shape[0]
    num_neighbors = min(neighbors_distance_mat.shape[1], num_neighbors)
    num_diff_ligands = diff_Lig_mat.shape[1]
    num_cont_neighbor_ligands = cont_Lig_mat.shape[1]


    diff_neighbor_ligand = np.zeros((num_receivers, num_neighbors, num_diff_ligands), dtype=np.float16)
    cont_neighbor_ligand = np.zeros((num_receivers, num_neighbors, num_cont_neighbor_ligands), dtype=np.float16)

    neighbor_indices = np.argsort(neighbors_distance_mat, axis=1)[:, :num_neighbors]
    diff_neighbor_ligand = diff_Lig_mat[neighbor_indices]
    cont_neighbor_ligand = cont_Lig_mat[neighbor_indices]

    diff_neighbor_distance = neighbors_distance_mat[np.arange(num_receivers)[:, None], neighbor_indices]
    cont_neighbor_distance = np.where(diff_neighbor_distance < np.sqrt(3), 1, 0).astype(np.float16)

    return diff_neighbor_ligand, diff_neighbor_distance, cont_neighbor_ligand, cont_neighbor_distance, receivers_distance_mat,receiver_coords
#     if not sparse.issparse(diff_Lig_mat):
#         diff_Lig_sparse = sparse.csr_matrix(diff_Lig_mat.values if hasattr(diff_Lig_mat, 'values') else diff_Lig_mat, 
#                                           dtype=np.float16)
#     else:
#         diff_Lig_sparse = diff_Lig_mat.astype(np.float16)
    
#     if not sparse.issparse(cont_Lig_mat):
#         cont_Lig_sparse = sparse.csr_matrix(cont_Lig_mat.values if hasattr(cont_Lig_mat, 'values') else cont_Lig_mat,
#                                           dtype=np.float16)
#     else:
#         cont_Lig_sparse = cont_Lig_mat.astype(np.float16)

#     # 原细胞筛选逻辑不变
#     sender_cells = cell_types['cell_type'] == sender if sender else np.ones(len(cell_types), dtype=bool)
#     receiver_cells = cell_types['cell_type'] == receiver if receiver else np.ones(len(cell_types), dtype=bool)

#     neighbors_distance_mat = distance_mat[receiver_cells, :][:, sender_cells]
#     receivers_distance_mat = distance_mat[receiver_cells, :][:, receiver_cells]
#     receiver_coords = cell_coordinates[receiver_cells]

#     num_receivers = neighbors_distance_mat.shape[0]
#     num_neighbors = min(neighbors_distance_mat.shape[1], num_neighbors)
    
#     # 改为稀疏矩阵存储结果
#     diff_neighbor_distance = sparse.lil_matrix((num_receivers, num_neighbors * diff_Lig_sparse.shape[1]), dtype=np.float16)
#     cont_neighbor_distance = sparse.lil_matrix((num_receivers, num_neighbors * cont_Lig_sparse.shape[1]), dtype=np.float16)

#     # 获取邻居索引（原逻辑不变）
#     neighbor_indices = np.argsort(neighbors_distance_mat, axis=1)[:, :num_neighbors]
    
#     # 填充稀疏矩阵
#     for i in range(num_receivers):
#         nbrs = neighbor_indices[i]
#         diff_neighbor_distance[i] = diff_Lig_sparse[nbrs].toarray().flatten()
#         cont_neighbor_distance[i] = cont_Lig_sparse[nbrs].toarray().flatten()

#     # 距离计算保持原逻辑
#     diff_neighbor_distance = neighbors_distance_mat[np.arange(num_receivers)[:, None], neighbor_indices]
#     cont_neighbor_distance = (diff_neighbor_distance < np.sqrt(3)).astype(np.float16)
#     return (diff_neighbor_distance.tocsr(),  # 返回CSR格式稀疏矩阵
#         diff_neighbor_distance,
#         cont_neighbor_distance.tocsr(),  # 返回CSR格式稀疏矩阵
#         cont_neighbor_distance,
#         receivers_distance_mat,
#         receiver_coords)
from scipy import sparse
import numpy as np

# def extract_neighbor_info(cell_coordinates, distance_mat, diff_Lig_mat, cont_Lig_mat, 
#                          cell_types, sender, receiver, num_neighbors=500):
#     # 输入矩阵稀疏化（保持您的原始代码）
#     if not sparse.issparse(diff_Lig_mat):
#         diff_Lig_sparse = sparse.csr_matrix(diff_Lig_mat.values if hasattr(diff_Lig_mat, 'values') else diff_Lig_mat, 
#                                           dtype=np.float16)
#     else:
#         diff_Lig_sparse = diff_Lig_mat.astype(np.float16)
    
#     if not sparse.issparse(cont_Lig_mat):
#         cont_Lig_sparse = sparse.csr_matrix(cont_Lig_mat.values if hasattr(cont_Lig_mat, 'values') else cont_Lig_mat,
#                                           dtype=np.float16)
#     else:
#         cont_Lig_sparse = cont_Lig_mat.astype(np.float16)

#     # 原细胞筛选逻辑不变
#     sender_cells = cell_types['cell_type'] == sender if sender else np.ones(len(cell_types), dtype=bool)
#     receiver_cells = cell_types['cell_type'] == receiver if receiver else np.ones(len(cell_types), dtype=bool)

#     neighbors_distance_mat = distance_mat[receiver_cells, :][:, sender_cells]
#     receivers_distance_mat = distance_mat[receiver_cells, :][:, receiver_cells]
#     receiver_coords = cell_coordinates[receiver_cells]

#     num_receivers = neighbors_distance_mat.shape[0]
#     num_neighbors = min(neighbors_distance_mat.shape[1], num_neighbors)
    
#     # 向量化获取邻居索引（替换argsort提升速度）
#     neighbor_indices = np.argpartition(neighbors_distance_mat, num_neighbors-1, axis=1)[:, :num_neighbors]
    
#     # 向量化构造稀疏矩阵（关键优化点）
#     row_indices = np.repeat(np.arange(num_receivers), num_neighbors)
#     col_indices = np.tile(np.arange(num_neighbors * diff_Lig_sparse.shape[1]), num_receivers)
    
#     # 一次性获取所有邻居数据（向量化核心）
#     global_indices = np.where(sender_cells)[0][neighbor_indices]  # 转换为全局索引
#     diff_data = diff_Lig_sparse[global_indices.ravel()].toarray().flatten()
#     cont_data = cont_Lig_sparse[global_indices.ravel()].toarray().flatten()
    
#     # 直接构造CSR矩阵（避免LIL格式转换）
#     diff_neighbor_ligand = sparse.csr_matrix(
#         (diff_data, (row_indices, col_indices)),
#         shape=(num_receivers, num_neighbors * diff_Lig_sparse.shape[1]),
#         dtype=np.float16
#     )
    
#     cont_neighbor_ligand = sparse.csr_matrix(
#         (cont_data, (row_indices, col_indices)),
#         shape=(num_receivers, num_neighbors * cont_Lig_sparse.shape[1]),
#         dtype=np.float16
#     )

#     # 距离计算保持原逻辑
#     receiver_idx = np.arange(num_receivers)[:, None]
#     diff_neighbor_distance = neighbors_distance_mat[receiver_idx, neighbor_indices]
#     cont_neighbor_distance = (diff_neighbor_distance < np.sqrt(3)).astype(np.float16)

#     return (diff_neighbor_ligand, 
#             diff_neighbor_distance,
#             cont_neighbor_ligand,
#             cont_neighbor_distance,
#             receivers_distance_mat,
#             receiver_coords)


