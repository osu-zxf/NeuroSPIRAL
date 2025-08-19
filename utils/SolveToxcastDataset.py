import torch
from torch_geometric.datasets import MoleculeNet
import pandas as pd
import os
from tqdm import tqdm # 用于显示处理进度

def process_toxcast_to_csv(root_dir='./data/Toxcast', output_dir='./data/Toxcast/csvs'):
    """
    处理 MoleculeNet 的 ToxCast 数据集，并为每个毒性终点生成一个独立的 CSV 文件。

    每个 CSV 文件将包含两列：'smiles' 和 'toxicity'。
    'toxicity' 列的值为 0 (无毒) 或 1 (有毒)。
    如果某个分子的某个终点标签为 NaN (缺失)，则该分子不会出现在该终点的 CSV 中。

    Args:
        root_dir (str): MoleculeNet 数据集下载和存储的根目录。
        output_dir (str): 生成的 CSV 文件将保存到的目录。
    """
    print(f"Attempting to load ToxCast dataset from {root_dir}...")
    try:
        # 加载 ToxCast 数据集。如果本地不存在，它会自动下载。
        dataset = MoleculeNet(root=root_dir, name='ToxCast')
        print(f"ToxCast dataset loaded successfully. Total molecules: {len(dataset)}")
        print(f"Number of tasks (endpoints): {dataset.num_classes}")
    except Exception as e:
        print(f"Error loading ToxCast dataset: {e}")
        print("Please ensure you have an active internet connection for the first download,")
        print(f"or that the dataset is correctly placed in {root_dir}/ToxCast.")
        return

    # 获取所有任务（终点）的名称
    if hasattr(dataset, 'task_names'):
        task_names = dataset.task_names
        print(f"Identified {len(task_names)} tasks from dataset.task_names.")
    else:
        # 备用方案：如果 dataset.task_names 不可用，则根据 num_classes 生成通用名称
        num_tasks = dataset.num_classes
        task_names = [f"task_{i}" for i in range(num_tasks)]
    print(f"Identified {len(task_names)} tasks:")
    # print(task_names) # 可以取消注释查看所有任务名称

    task_data_collections = {task_name: [] for task_name in task_names}

    print("\nProcessing molecules and collecting data for each endpoint...")
    # 遍历数据集中的每个分子
    for i, data_item in enumerate(tqdm(dataset, desc="Collecting data")):
        smiles = data_item.smiles
        # data_item.y 的形状通常是 (1, num_tasks)，包含每个任务的标签
        labels = data_item.y

        # 检查标签是否有效
        if labels is None or labels.numel() == 0:
            # print(f"Warning: Molecule {i} (SMILES: {smiles}) has no labels. Skipping.")
            continue
        
        # 遍历当前分子的所有任务标签
        for task_idx, task_name in enumerate(task_names):
            # 确保 task_idx 不会超出 labels 的维度
            if task_idx >= labels.shape[1]:
                print(f"Warning: Task index {task_idx} out of bounds for labels shape {labels.shape}. Skipping remaining tasks for this molecule.")
                break # 跳出当前分子的任务循环
            
            # 提取当前任务的标签
            # labels 是 (1, num_tasks) 形状，所以用 labels[0, task_idx]
            label = labels[0, task_idx]

            # 检查标签是否为 NaN (缺失值)。如果是 NaN，则跳过该分子在该任务的数据收集。
            if not torch.isnan(label):
                # 将 PyTorch tensor 标签转换为 Python int
                task_data_collections[task_name].append({
                    'smiles': smiles,
                    'toxicity': int(label.item()) # label.item() 获取标量值
                })

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving CSV files to: {output_dir}")

    # 为每个任务保存数据到独立的 CSV 文件
    for task_name, data_list in task_data_collections.items():
        if not data_list:
            print(f"No valid data found for task: '{task_name}'. Skipping CSV creation for this task.")
            continue

        # 将列表数据转换为 Pandas DataFrame
        df = pd.DataFrame(data_list)
        
        csv_filename = os.path.join(output_dir, f"toxcast_{task_name}.csv")
        
        # 保存 DataFrame 到 CSV 文件，不包含索引列
        df.to_csv(csv_filename, index=True)
        print(f"Saved {len(data_list)} entries for task '{task_name}' to {csv_filename}")

    print("\nProcessing complete.")

# 调用函数开始处理
if __name__ == '__main__':
    process_toxcast_to_csv()