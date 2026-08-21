import matplotlib.pyplot as plt
import pandas as pd
def visualize_receivers(receivers, coords, cell_types):
    """
    可视化指定的接收者细胞类型（receivers），并将其他细胞类型以灰色显示。

    参数:
    - receivers: 列表，包含需要高亮显示的细胞类型。
    - coords: DataFrame，包含细胞的坐标信息，列名为 ['x_coord', 'y_coord']。
    - cell_types: DataFrame，包含细胞类型信息，列名为 ['cell_ID', 'cell_type']。
    """
    # 定义颜色映射
    colors = {receiver: f'C{i}' for i, receiver in enumerate(receivers)}  # 为每个receiver分配一个颜色
    other_color = 'grey'  # 其他细胞类型定义为灰色

    # 创建一个新列来存储颜色
    cell_types['color'] = cell_types['cell_type'].apply(lambda x: colors.get(x, other_color))

    # 绘制散点图，将点的大小设置为较小值
    plt.figure(figsize=(12, 10))
    for cell_type, group in cell_types.groupby('cell_type'):
        coords_subset = coords.loc[group.index]
        plt.scatter(coords_subset['x_coord'], coords_subset['y_coord'], 
                    color=group['color'].iloc[0], 
                    label=cell_type, 
                    s=10,
                    alpha=0.6)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Receiver Cell Types Visualization')
    
    # 调整图例位置和大小
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), title='Cell Types')
    
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig('receiver_cell_types_visualization.png')
