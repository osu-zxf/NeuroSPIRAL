import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
# 1. 加载分子数据集（SMILES 格式）
# 假设数据集是一个 CSV 文件，包含一列 "smiles"
# 如果你有自己的 SMILES 数据集，请替换文件路径
data = pd.read_csv("result/tox_smiles_dataset.csv")  # 替换为实际文件路径
smiles_list = data["SMILES"].dropna().tolist()  # 获取 SMILES 列

# 2. 分割分子并统计基团
fragments = set()  # 使用集合确保基团不重复
for smiles in tqdm(smiles_list):
    try:
        mol = Chem.MolFromSmiles(smiles)  # 转换 SMILES 为分子对象
        if mol:
            # 使用 BRICS 分割分子
            frag_set = BRICS.BRICSDecompose(mol)
            fragments.update(frag_set)  # 将新基团加入集合
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")

# 3. 统计基团的原子数
atom_counts = []
for frag_smiles in tqdm(fragments):
    try:
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        if frag_mol:
            atom_counts.append(frag_mol.GetNumAtoms())  # 获取基团的原子数
    except Exception as e:
        print(f"Error processing fragment {frag_smiles}: {e}")

# 4. 绘制直方图
# 统计各原子数的频次
count_dict = Counter(atom_counts)
sorted_counts = sorted(count_dict.items())  # 按原子数排序

# 提取横坐标（原子数）和纵坐标（频次）
x = [item[0] for item in sorted_counts]
y = [item[1] for item in sorted_counts]

print('x:', x)
print('y:', y)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.bar(x, y, color='skyblue', edgecolor='black')
plt.xlabel("Number of Atoms in Fragment", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of Fragment Atom Counts", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()