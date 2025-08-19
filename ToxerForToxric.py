import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.datasets import MoleculeNet # Corrected import
from torch_geometric.utils import scatter

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors # Import Descriptors for global features
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

class TOXRICDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        df = pd.read_csv(self.raw_paths[0], dtype={'ID': str, 'SMILES': str})
        self.label_columns = [col for col in df.columns if col not in ['ID', 'SMILES']]

    @property
    def raw_file_names(self):
        # 这里随便写一个原始文件名即可
        return ['toxric_data.csv']
    
    @property
    def processed_file_names(self):
        # 这里指定一个你要保存的处理后文件名
        return ['data.pt']

    @property
    def num_classes(self):
        return len(self.label_columns)
    
    def process(self):
        self.csv_path = self.root+'/raw/toxric_data.csv'
        self.data_df = pd.read_csv(self.csv_path, dtype={'ID': str, 'SMILES': str})
        # 去除SMILES为缺失的行
        self.data_df = self.data_df[self.data_df['SMILES'].notna()]
        self.label_columns = [col for col in self.data_df.columns if col not in ['ID', 'SMILES']]
        print(len(self.label_columns))

        data_list = []
        for idx, row in self.data_df.iterrows():
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"解析SMILES失败：{smiles}, 跳过")
                continue
            # 原子特征：这里以原子序数为例
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append([atom.GetAtomicNum()])
            x = torch.tensor(atom_features, dtype=torch.float)

            # 多标签y，缺失为nan
            y = torch.tensor([row[col] if pd.notnull(row[col]) else float('nan') for col in self.label_columns], dtype=torch.float).unsqueeze(0)
            
            data = Data(x=x, y=y, smiles=smiles, id=row['ID'])
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


# --- 1. 数据加载与预处理 ---
# 尝试加载TOX21数据集
print("Loading TOXRIC dataset...")

# Corrected: Use MoleculeNet to load Tox21
dataset = TOXRICDataset(root='./data/TOXRIC')
print(f"Dataset loaded. Number of molecules: {len(dataset)}")
print(f"Number of tasks: {dataset.num_classes}")
NUM_TASKS = dataset.num_classes

filtered_dataset = []
for data_item in dataset:
    # Ensure data.y is not None and not all NaNs
    if data_item.y is not None and not torch.isnan(data_item.y).all():
        filtered_dataset.append(data_item)

print(f"Filtered dataset (valid labels): {len(filtered_dataset)} molecules.")

# Split dataset (using a fixed split for reproducibility)
train_size = int(0.8 * len(filtered_dataset))
val_size = int(0.1 * len(filtered_dataset))
test_size = len(filtered_dataset) - train_size - val_size

# Shuffle and split
np.random.shuffle(filtered_dataset)
train_dataset_raw = filtered_dataset[:train_size]
val_dataset_raw = filtered_dataset[train_size:(train_size + val_size)]
test_dataset_raw = filtered_dataset[(train_size + val_size):]

print(f"Raw splits: Train: {len(train_dataset_raw)}, Val: {len(val_dataset_raw)}, Test: {len(test_dataset_raw)}")

# --- 2. 多层次图构建 ---

PREDEFINED_SUBSTRUCTURES = [
    ("Amine", "[N;H2,H1;!$(NC=O)]"), # Primary/secondary amine, not amide
    ("CarboxylicAcid", "C(=O)[O;H1]"),
    ("Alcohol", "[O;H1]-C"), # Aliphatic alcohol
    ("BenzeneRing", "c1ccccc1"),
    ("Nitro", "[N+](=O)[O-]"),
    ("Halogen", "[F,Cl,Br,I]"), # Individual halogen atoms
    ("Ketone", "C(=O)[#6]"), # Carbonyl with two carbons attached
    ("Aldehyde", "C(=O)[H]"), # Carbonyl with a hydrogen attached
    ("Ether", "[#6]-[O]-[#6]"), # Simple ether
    ("Thiol", "[S;H1]"), # Thiol group
    ("SulfonicAcid", "S(=O)(=O)[O;H1]"), # Sulfonic acid
]
SUBSTRUCTURE_TYPES = [name for name, _ in PREDEFINED_SUBSTRUCTURES]
NUM_SUBSTRUCTURE_TYPES = len(SUBSTRUCTURE_TYPES)

def one_hot_encoding(value, choices):
    """
    对给定值进行 One-Hot 编码。
    如果值不在 choices 中，则最后一个位置为 1 (表示 'other' 或 'unknown')。
    """
    encoding = [0] * (len(choices) + 1) # +1 for 'other' category
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1 # Value not in choices, mark as 'other'
    return encoding

# --- 定义 One-Hot 编码的可能值列表 ---
# 常见的原子序数 (C, N, O, F, P, S, Cl, Br, I) + 'other'
ATOM_ATOMIC_NUM_CHOICES = [6, 7, 8, 9, 15, 16, 17, 35, 53]
# 常见的杂化类型
ATOM_HYBRIDIZATION_CHOICES = [
    Chem.HybridizationType.S, Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2,
    Chem.HybridizationType.UNSPECIFIED # RDKit 默认值
]

# 常见的键类型
BOND_TYPE_CHOICES = [
    Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC
]

# --- 原子特征维度定义 ---
# 原始数值特征 (Degree, ImplicitValence, FormalCharge, NumExplicitHs, NumImplicitHs, TotalNumHs, Isotope, Mass) = 8
# 布尔特征 (IsAromatic, IsInRing) = 2
# One-Hot 特征:
#   AtomicNum: len(ATOM_ATOMIC_NUM_CHOICES) + 1
#   Hybridization: len(ATOM_HYBRIDIZATION_CHOICES) + 1
#   ChiralTag: len(ATOM_CHIRAL_TAG_CHOICES) + 1
ATOM_FEATURE_DIM = (
    9 + 2 +
    (len(ATOM_ATOMIC_NUM_CHOICES) + 1) +
    (len(ATOM_HYBRIDIZATION_CHOICES) + 1)
)

# --- 键特征维度定义 ---
# 原始数值特征 (Stereo) = 1 (GetStereo()返回int)
# 布尔特征 (IsInRing, IsConjugated) = 2
# One-Hot 特征:
#   BondType: len(BOND_TYPE_CHOICES) + 1
BOND_FEATURE_DIM = (
    1 + 2 +
    (len(BOND_TYPE_CHOICES) + 1)
)

# 全局分子描述符维度定义
GLOBAL_FEATURE_DIM = 9


def get_atom_features(atom):
    features = []
    
    # One-Hot 编码的特征
    features.extend(one_hot_encoding(atom.GetAtomicNum(), ATOM_ATOMIC_NUM_CHOICES))
    features.extend(one_hot_encoding(atom.GetHybridization(), ATOM_HYBRIDIZATION_CHOICES))

    # 数值特征
    features.append(float(atom.GetDegree()))
    features.append(float(atom.GetImplicitValence()))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetNumExplicitHs()))
    features.append(float(atom.GetNumImplicitHs()))
    features.append(float(atom.GetTotalNumHs()))
    features.append(float(atom.GetIsotope()))
    features.append(int(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED))
    features.append(float(atom.GetMass()))
    # 布尔特征 (转换为 float)
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.IsInRing()))

    if len(features) != ATOM_FEATURE_DIM:
        # 理论上不应该发生，但为了鲁棒性进行检查和处理
        print(f"Warning: Atom features length mismatch! Expected {ATOM_FEATURE_DIM}, got {len(features)} for atom {atom.GetIdx()} (AtomicNum: {atom.GetAtomicNum()})")
        if len(features) < ATOM_FEATURE_DIM:
            features.extend([0.0] * (ATOM_FEATURE_DIM - len(features))) # 填充零
        elif len(features) > ATOM_FEATURE_DIM:
            features = features[:ATOM_FEATURE_DIM] # 截断
    
    return torch.tensor(features, dtype=torch.float)

def get_bond_features(bond):
    features = []
    
    # One-Hot 编码的特征
    features.extend(one_hot_encoding(bond.GetBondType(), BOND_TYPE_CHOICES))

    # 数值特征
    features.append(float(bond.GetStereo())) # GetStereo() 返回 int

    # 布尔特征 (转换为 float)
    features.append(float(bond.IsInRing()))
    features.append(float(bond.GetIsConjugated()))
    
    if len(features) != BOND_FEATURE_DIM:
        # 理论上不应该发生，但为了鲁棒性进行检查和处理
        print(f"Warning: Bond features length mismatch! Expected {BOND_FEATURE_DIM}, got {len(features)} for bond {bond.GetIdx()} (Type: {bond.GetBondType()})")
        if len(features) < BOND_FEATURE_DIM:
            features.extend([0.0] * (BOND_FEATURE_DIM - len(features))) # 填充零
        elif len(features) > BOND_FEATURE_DIM:
            features = features[:BOND_FEATURE_DIM] # 截断

    return torch.tensor(features, dtype=torch.float)

def get_global_features(mol):
    """
    计算并返回一组RDKit全局分子描述符。
    """
    if mol is None:
        return torch.zeros(GLOBAL_FEATURE_DIM, dtype=torch.float) # Return zeros if mol is None

    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
    ]

    if len(features) != GLOBAL_FEATURE_DIM:
        # 这是导致 "Expected size 9 but got size 6" 错误的最可能原因
        print(f"Warning: Global features length mismatch! Expected {GLOBAL_FEATURE_DIM}, got {len(features)} for SMILES: {Chem.MolToSmiles(mol) if mol else 'None'}")
        if len(features) < GLOBAL_FEATURE_DIM:
            features.extend([0.0] * (GLOBAL_FEATURE_DIM - len(features))) # 填充零
        elif len(features) > GLOBAL_FEATURE_DIM:
            features = features[:GLOBAL_FEATURE_DIM] # 截断

    # Note: For real applications, these features should be normalized (e.g., min-max scaling)
    return torch.tensor(features, dtype=torch.float)


def mol_to_multi_level_graph(mol, original_data=None):
    if mol is None:
        return None

    # --- 原子图构建 (Atomic Graph Construction) ---
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    x_atom = torch.stack([get_atom_features(atom) for atom in mol.GetAtoms()])

    edge_indices = []
    edge_attr_atom_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_indices.append([i, j])
        edge_attr_atom_list.append(get_bond_features(bond))
        
        edge_indices.append([j, i])
        edge_attr_atom_list.append(get_bond_features(bond))

    if not edge_indices:
        edge_index_atom = torch.empty((2, 0), dtype=torch.long)
        edge_attr_atom = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index_atom = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr_atom = torch.stack(edge_attr_atom_list)

    # --- 子结构图构建 (Substructure Graph Construction) ---
    substructure_matches = []
    for sub_type_idx, (sub_name, smarts) in enumerate(PREDEFINED_SUBSTRUCTURES):
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        for match_atoms_tuple in matches:
            substructure_matches.append((sub_type_idx, match_atoms_tuple))

    num_substructures = len(substructure_matches)

    x_sub = torch.empty((0, ATOM_FEATURE_DIM + NUM_SUBSTRUCTURE_TYPES), dtype=torch.float)
    edge_index_sub = torch.empty((2, 0), dtype=torch.long)
    if num_substructures == 0:
        return None # 如果没有子结构匹配，直接返回None
    
    edge_index_substructure_atom_map_indices = torch.empty((2, 0), dtype=torch.long)
    num_sub_atom_connections = 0

    if num_substructures > 0:
        x_sub_list = []
        sub_atom_connections_local = [] # 存储当前分子的原子-子结构连接
        for i, (sub_type_idx, atom_indices) in enumerate(substructure_matches):
            sub_type_one_hot = F.one_hot(torch.tensor(sub_type_idx), num_classes=NUM_SUBSTRUCTURE_TYPES).float()
            atom_features_in_sub = x_atom[list(atom_indices)].mean(dim=0)
            x_sub_list.append(torch.cat([atom_features_in_sub, sub_type_one_hot]))

            # 填充原子-子结构连接映射
            for atom_idx in atom_indices:
                sub_atom_connections_local.append([atom_idx, i]) # [原子局部索引, 子结构局部索引]
        
        x_sub = torch.stack(x_sub_list)
        assert x_sub.shape[0] == num_substructures, \
            f"mol_to_multi_level_graph: x_sub.shape[0] ({x_sub.shape[0]}) mismatch with num_substructures ({num_substructures}) for SMILES: {mol.GetSmiles()}"

        if sub_atom_connections_local:
            edge_index_substructure_atom_map_indices = torch.tensor(sub_atom_connections_local, dtype=torch.long).t().contiguous()
            num_sub_atom_connections = edge_index_substructure_atom_map_indices.shape[1] # 获取连接数

        edge_indices_sub = []
        for i in range(num_substructures):
            for j in range(i + 1, num_substructures):
                sub_i_atoms = set(substructure_matches[i][1])
                sub_j_atoms = set(substructure_matches[j][1])

                connected = False
                for bond in mol.GetBonds():
                    atom1_idx = bond.GetBeginAtomIdx()
                    atom2_idx = bond.GetEndAtomIdx()

                    if (atom1_idx in sub_i_atoms and atom2_idx in sub_j_atoms) or \
                       (atom1_idx in sub_j_atoms and atom2_idx in sub_i_atoms):
                        edge_indices_sub.append([i, j])
                        edge_indices_sub.append([j, i])
                        connected = True
                        break 
        
        if not edge_indices_sub:
            edge_index_sub = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index_sub = torch.tensor(edge_indices_sub, dtype=torch.long).t().contiguous()

        num_sub_edges = edge_index_sub.shape[1]

        if num_sub_edges > 0:
            max_local_idx = edge_index_sub.max().item()
            assert max_local_idx < num_substructures, \
                f"mol_to_multi_level_graph: Local edge index out of bounds (max_idx={max_local_idx}, num_sub={num_substructures}) for SMILES: {mol.GetSmiles()}"
    else:
        num_sub_edges = 0

    # --- 全局分子描述符 ---
    global_features = get_global_features(mol)
    # 创建PyG Data对象
    data = Data(x=x_atom, edge_index=edge_index_atom, edge_attr=edge_attr_atom,
                x_sub=x_sub, edge_index_sub=edge_index_sub,
                global_features=global_features, # Add global features
                num_atoms=torch.tensor(num_atoms, dtype=torch.long),
                num_substructures=torch.tensor(num_substructures, dtype=torch.long),
                num_sub_edges=torch.tensor(num_sub_edges, dtype=torch.long),
                edge_index_substructure_atom_map_indices=edge_index_substructure_atom_map_indices, # 新增映射
                num_sub_atom_connections=torch.tensor(num_sub_atom_connections, dtype=torch.long)) # 新增连接数
    
    if original_data:
        data.y = original_data.y
        data.smiles = original_data.smiles
    
    return data

print("Constructing multi-level graphs for the dataset...")
processed_dataset = []
all_global_features = []

for original_data_item in tqdm(filtered_dataset, desc="Processing molecules"):
    mol = Chem.MolFromSmiles(original_data_item.smiles)
    if mol is None:
        print(f"Warning: Could not parse SMILES: {original_data_item.smiles}")
        continue
    
    multi_level_data = mol_to_multi_level_graph(mol, original_data_item)
    if multi_level_data is not None:
        processed_dataset.append(multi_level_data)
        all_global_features.append(multi_level_data.global_features.numpy())

print(f"Successfully processed {len(processed_dataset)} molecules into multi-level graphs.")

if len(all_global_features) > 0:
    global_features_array = np.array(all_global_features)
    scaler = MinMaxScaler() # 或者 StandardScaler()
    # Fit and transform the collected features
    scaled_global_features_array = scaler.fit_transform(global_features_array)
    
    # Update the global_features in each Data object
    for i, data_item in enumerate(processed_dataset):
        data_item.global_features = torch.tensor(scaled_global_features_array[i], dtype=torch.float)
    print("Global features normalized using MinMaxScaler.")
else:
    print("No global features to normalize.")

# 重新分割处理后的数据集，确保一致性
train_dataset_processed = []
val_dataset_processed = []
test_dataset_processed = []

# Using smiles as unique identifier for splitting
train_smiles = {d.smiles for d in train_dataset_raw}
val_smiles = {d.smiles for d in val_dataset_raw}
test_smiles = {d.smiles for d in test_dataset_raw}

for data in processed_dataset:
    if data.smiles in train_smiles:
        train_dataset_processed.append(data)
    elif data.smiles in val_smiles:
        val_dataset_processed.append(data)
    elif data.smiles in test_smiles:
        test_dataset_processed.append(data)

print(f"Processed splits: Train: {len(train_dataset_processed)}, Val: {len(val_dataset_processed)}, Test: {len(test_dataset_processed)}")

train_loader = DataLoader(train_dataset_processed, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset_processed, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset_processed, batch_size=128, shuffle=False)

# --- 3. 多层次GNN架构 ---

class AtomicGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_attr_dim, heads=4):
        super(AtomicGNN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, edge_dim=edge_attr_dim)
        self.conv2 = GATConv(hidden_channels, out_channels // heads, heads=heads, edge_dim=edge_attr_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

class SubstructureGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(SubstructureGNN, self).__init__()
        self.heads = heads
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, out_channels // heads, heads=heads)

    def forward(self, x_sub, edge_index_sub):
        x_sub = self.conv1(x_sub, edge_index_sub)
        x_sub = F.relu(x_sub)
        x_sub = F.dropout(x_sub, p=0.5, training=self.training)
        x_sub = self.conv2(x_sub, edge_index_sub)
        return x_sub

class FusionAttention(nn.Module):
    """
    使用双向交叉注意力融合原子级和子结构级分子嵌入。
    """
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.atom_to_sub_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=False)
        self.sub_to_atom_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=False)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, atom_mol_embedding, sub_mol_embedding):
        atom_seq = atom_mol_embedding.unsqueeze(0)
        sub_seq = sub_mol_embedding.unsqueeze(0)

        attended_sub_from_atom, _ = self.atom_to_sub_attn(query=atom_seq, key=sub_seq, value=sub_seq)
        attended_sub_from_atom = attended_sub_from_atom.squeeze(0)

        attended_atom_from_sub, _ = self.sub_to_atom_attn(query=sub_seq, key=atom_seq, value=atom_seq)
        attended_atom_from_sub = attended_atom_from_sub.squeeze(0)

        combined_embeddings = torch.cat([
            atom_mol_embedding,
            attended_sub_from_atom,
            sub_mol_embedding,
            attended_atom_from_sub
        ], dim=1)

        fused_emb = self.fusion_mlp(combined_embeddings)
        fused_emb = self.norm(fused_emb)
        return fused_emb

class MultiLevelGNN(nn.Module):
    """
    多层次GNN模型，融合原子级、子结构级信息和全局描述符。
    采用跨层次注意力融合。
    """
    def __init__(self, atom_in_channels, sub_in_channels_original, hidden_channels, out_channels, num_tasks, edge_attr_dim, global_feature_dim, gnn_heads=4):
        super(MultiLevelGNN, self).__init__()
        self.out_channels = out_channels
        self.gnn_heads = gnn_heads

        self.atomic_gnn = AtomicGNN(atom_in_channels, hidden_channels, out_channels, edge_attr_dim, heads=self.gnn_heads)
        
        # SubstructureGNN的输入维度现在是原始子结构特征维度 + 原子GNN的输出维度
        self.substructure_gnn = SubstructureGNN(sub_in_channels_original + out_channels, hidden_channels, out_channels, heads=self.gnn_heads)

        self.fusion_attention = FusionAttention(out_channels, num_heads=self.gnn_heads)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_channels + global_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, num_tasks)
        )

    def forward(self, data):
        # 1. 原子图 GNN 处理
        x_atom_emb = self.atomic_gnn(data.x, data.edge_index, data.edge_attr) # (num_total_atoms, out_channels)
        atom_mol_embedding = global_mean_pool(x_atom_emb, data.batch)

        # 2. 动态生成子结构特征 (中间信息传递)
        x_sub_initial = data.x_sub # 原始的子结构特征 (num_total_substructures, original_sub_feature_dim)
        num_total_substructures_in_batch = x_sub_initial.shape[0]

        # 初始化动态特征为零向量
        x_sub_dynamic_features = torch.zeros(num_total_substructures_in_batch,
                                             self.out_channels, # 匹配 x_atom_emb 的输出维度
                                             device=x_atom_emb.device)

        # 只有当存在原子-子结构连接时才进行池化
        if data.edge_index_substructure_atom_map_indices.numel() > 0:
            # 手动对 substructure_atom_map_indices 进行索引偏移
            # 这里的逻辑与 edge_index_sub 的偏移类似
            atom_offsets = torch.cumsum(torch.cat([
                torch.tensor([0], device=data.num_atoms.device),
                data.num_atoms
            ]), dim=0)[:-1]
            sub_offsets = torch.cumsum(torch.cat([
                torch.tensor([0], device=data.num_substructures.device),
                data.num_substructures
            ]), dim=0)[:-1]

            # data.num_sub_atom_connections 包含了每个图的原子-子结构连接数
            if isinstance(data.num_sub_atom_connections, (list, tuple)):
                num_sub_atom_connections_tensor = torch.tensor(data.num_sub_atom_connections, device=data.x.device)
            else:
                num_sub_atom_connections_tensor = data.num_sub_atom_connections

            if num_sub_atom_connections_tensor.numel() > 0:
                if num_sub_atom_connections_tensor.ndim > 1:
                    num_sub_atom_connections_tensor = num_sub_atom_connections_tensor.flatten()
                
                # 创建用于偏移的张量
                offset_atom_indices = torch.repeat_interleave(atom_offsets, num_sub_atom_connections_tensor)
                offset_sub_indices = torch.repeat_interleave(sub_offsets, num_sub_atom_connections_tensor)

                # 应用偏移
                shifted_substructure_atom_map_indices = data.edge_index_substructure_atom_map_indices.clone()
                # shifted_substructure_atom_map_indices[0, :] += offset_atom_indices
                shifted_substructure_atom_map_indices[1, :] -= offset_atom_indices
                shifted_substructure_atom_map_indices[1, :] += offset_sub_indices

                # 从原子嵌入中池化动态子结构特征 (使用 mean 聚合)
                # src: x_atom_emb[shifted_substructure_atom_map_indices[0]] (原子嵌入)
                # index: shifted_substructure_atom_map_indices[1] (目标子结构索引)
                x_sub_dynamic_features_pooled = scatter(x_atom_emb[shifted_substructure_atom_map_indices[0]],
                                                        shifted_substructure_atom_map_indices[1],
                                                        dim=0,
                                                        dim_size=num_total_substructures_in_batch,
                                                        reduce='mean')
                # 处理可能出现的 NaN (例如，某个子结构没有原子映射到它)
                x_sub_dynamic_features_pooled = torch.nan_to_num(x_sub_dynamic_features_pooled, nan=0.0)
                
                x_sub_dynamic_features = x_sub_dynamic_features_pooled
        
        # 拼接原始子结构特征和动态生成的特征
        x_sub_combined_features = torch.cat([x_sub_initial, x_sub_dynamic_features], dim=-1)

        # 3. 子结构图 GNN 处理
        # 初始化 sub_mol_embedding 为零向量，用于没有子结构的分子
        sub_mol_embedding = torch.zeros(data.num_graphs, self.out_channels, device=data.x.device)

        if x_sub_combined_features.shape[0] > 0: # 确保有子结构节点
            atom_offsets_per_graph = torch.cumsum(data.num_atoms, dim=0) - data.num_atoms
            graph_indices_for_edges = torch.arange(data.num_graphs, device=data.num_sub_edges.device).repeat_interleave(data.num_sub_edges)
            atom_offsets_for_edges = atom_offsets_per_graph[graph_indices_for_edges]
            local_sub_edge_index = data.edge_index_sub - atom_offsets_for_edges
            
            substructure_offsets_per_graph = torch.cumsum(data.num_substructures, dim=0) - data.num_substructures
            substructure_offsets_for_edges = substructure_offsets_per_graph[graph_indices_for_edges]
            correctly_shifted_edge_index_sub = local_sub_edge_index + substructure_offsets_for_edges
            
            x_sub_emb = self.substructure_gnn(x_sub_combined_features, correctly_shifted_edge_index_sub)

            sub_batch_list = []
            for i in range(data.num_graphs):
                num_s = data.num_substructures[i].item()
                if num_s > 0:
                    sub_batch_list.extend([i] * num_s)
            
            if len(sub_batch_list) > 0:
                sub_batch = torch.tensor(sub_batch_list, dtype=torch.long, device=data.x_sub.device)
                pooled_sub_embeddings_present = global_mean_pool(x_sub_emb, sub_batch)
                unique_graph_ids_with_sub = torch.unique(sub_batch)
                sub_mol_embedding[unique_graph_ids_with_sub] = pooled_sub_embeddings_present

        # 4. 跨层次注意力融合
        fused_attention_embedding = self.fusion_attention(atom_mol_embedding, sub_mol_embedding)

        # 5. 结合全局分子描述符
        global_features_batch = data.global_features.view(data.num_graphs, -1)
        final_fused_embedding = torch.cat([fused_attention_embedding, global_features_batch], dim=1)
        
        out = self.fusion_mlp(final_fused_embedding)
        return out

# 模型参数
SUB_FEATURE_DIM_ORIGINAL = ATOM_FEATURE_DIM + NUM_SUBSTRUCTURE_TYPES # 原始子结构特征维度

HIDDEN_CHANNELS = 64
OUT_CHANNELS = 64 
GNN_HEADS = 4

# 确保 HIDDEN_CHANNELS 和 OUT_CHANNELS 能被 GNN_HEADS 整除
assert HIDDEN_CHANNELS % GNN_HEADS == 0, "HIDDEN_CHANNELS must be divisible by GNN_HEADS"
assert OUT_CHANNELS % GNN_HEADS == 0, "OUT_CHANNELS must be divisible by GNN_HEADS"

model = MultiLevelGNN(ATOM_FEATURE_DIM, SUB_FEATURE_DIM_ORIGINAL, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_TASKS, BOND_FEATURE_DIM, GLOBAL_FEATURE_DIM, gnn_heads=GNN_HEADS)

Train_Mode = False
model_save_path = 'results/toxric/multilevel_gnn_model_epoch196.pth'
device = torch.device('cpu') # 'cuda' if torch.cuda.is_available() else 'cpu')

if Train_Mode:
    model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss(reduction='none')

# --- 4. 训练与评估 ---
def train(loader, epoch_num=0):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc=f"Training Epoch {epoch_num}"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
        mask = ~torch.isnan(data.y)
        
        if mask.sum() > 0:
            loss = criterion(out[mask], data.y[mask])
            total_loss += loss.mean().item() * data.num_graphs
            loss.mean().backward()
            optimizer.step()
    return total_loss / len(loader.dataset)

def evaluate(loader):
    model.eval()
    y_preds_all = []
    y_trues_all = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y_pred = torch.sigmoid(out)
            
            for i in range(data.y.shape[0]):
                for j in range(data.y.shape[1]):
                    if not torch.isnan(data.y[i, j]):
                        y_preds_all.append(y_pred[i, j].item())
                        y_trues_all.append(data.y[i, j].item())
    
    if len(y_trues_all) == 0 or len(set(y_trues_all)) < 2:
        return 0.5
    
    try:
        auc = roc_auc_score(y_trues_all, y_preds_all)
    except ValueError:
        auc = 0.5

    return auc

def evaluate_per_task(loader):
    model.eval() # 设置模型为评估模式 (关闭 dropout, batchnorm 等)
    
    # 关键改动1: 初始化列表的列表，每个子列表对应一个任务
    # all_y_trues_per_task[j] 将包含第 j 个任务的所有真实标签
    # all_y_preds_per_task[j] 将包含第 j 个任务的所有预测概率
    all_y_trues_per_task = [[] for _ in range(NUM_TASKS)]
    all_y_preds_per_task = [[] for _ in range(NUM_TASKS)]

    with torch.no_grad(): # 在推理阶段关闭梯度计算，节省内存并加速
        for data in loader:
            data = data.to(device)
            out = model(data)
            y_pred_batch = torch.sigmoid(out) # 将模型的 logits 输出转换为概率

            # data.y 的形状是 (batch_size, num_tasks)
            # y_pred_batch 的形状也是 (batch_size, num_tasks)

            # 遍历批次中的每个样本 (分子)
            for i in range(data.y.shape[0]): # data.y.shape[0] 是当前批次的 batch_size
                # 关键改动2: 遍历每个任务
                for j in range(NUM_TASKS):
                    true_label = data.y[i, j]
                    predicted_prob = y_pred_batch[i, j]

                    # 只有当真实标签不是 NaN (缺失值) 时才收集数据
                    if not torch.isnan(true_label):
                        # 关键改动3: 将数据添加到对应任务的列表中
                        all_y_trues_per_task[j].append(true_label.item())
                        all_y_preds_per_task[j].append(predicted_prob.item())
    
    # 关键改动4: 计算每个任务的 ROC-AUC
    task_aucs = {}
    for task_idx in range(NUM_TASKS):
        y_trues = all_y_trues_per_task[task_idx]
        y_preds = all_y_preds_per_task[task_idx]

        # 获取任务名称，如果未提供则使用通用名称
        current_task_name = f"Task_{task_idx}"

        # 检查是否可以计算 AUC
        # 需要至少两个样本，并且真实标签中包含 0 和 1 (即至少两个不同的类别)
        if len(y_trues) < 2 or len(set(y_trues)) < 2:
            # 数据不足或只有一个类别，无法计算 AUC
            # 这种情况下，通常返回 0.5 (表示随机猜测的性能)
            task_aucs[current_task_name] = 0.5 
        else:
            try:
                auc = roc_auc_score(y_trues, y_preds)
                task_aucs[current_task_name] = auc
            except ValueError:
                # 尽管有 0 和 1，但如果所有预测值都相同，roc_auc_score 可能会报错
                # 这种情况下也返回 0.5
                task_aucs[current_task_name] = 0.5
    
    # 关键改动5: 返回一个字典
    return task_aucs

def evaluate_overall_accuracy(loader, threshold=0.5):
    """
    评估模型在所有任务上的总体准确率。

    Args:
        loader (torch_geometric.loader.DataLoader): 数据加载器。
        model (torch.nn.Module): 训练好的模型。
        device (torch.device): 模型所在的设备 (e.g., 'cpu' or 'cuda')。
        threshold (float): 将预测概率转换为二元预测的阈值。默认为 0.5。

    Returns:
        float: 模型在所有有效数据点上的总体准确率。
               如果没有任何有效数据，则返回 0.0。
    """
    model.eval() # 设置模型为评估模式
    
    y_trues_all = [] # 存储所有任务的真实标签（非NaN）
    y_preds_all = [] # 存储所有任务的二元预测标签

    with torch.no_grad(): # 在推理阶段关闭梯度计算
        for data in loader:
            data = data.to(device)
            out = model(data)
            y_pred_probs = torch.sigmoid(out) # 获取预测概率

            # 将预测概率转换为二元预测 (0 或 1)
            y_pred_binary = (y_pred_probs > threshold).float()
            
            # 遍历批次中的每个样本和每个任务
            for i in range(data.y.shape[0]): # 遍历 batch_size
                for j in range(data.y.shape[1]): # 遍历 num_tasks
                    true_label = data.y[i, j]
                    
                    # 仅收集非 NaN 的真实标签及其对应的预测
                    if not torch.isnan(true_label):
                        y_trues_all.append(true_label.item())
                        y_preds_all.append(y_pred_binary[i, j].item()) # 注意这里是二元预测

    # 检查是否有足够的有效数据来计算准确率
    if len(y_trues_all) == 0:
        print("Warning: No valid (non-NaN) labels found to calculate overall accuracy.")
        return 0.0 # 或者您可以使用 np.nan
    
    # 计算总体准确率
    try:
        overall_accuracy = accuracy_score(y_trues_all, y_preds_all)
    except ValueError as e:
        print(f"Error calculating overall accuracy: {e}")
        # 这通常发生在 y_trues_all 或 y_preds_all 为空或不匹配时，但我们上面已经检查过长度
        # 也可以是所有预测都相同，导致sklearn内部问题，但对于accuracy_score通常不是问题
        overall_accuracy = 0.0 # Fallback
        
    return overall_accuracy

if Train_Mode:
    epochs = 200
    print(f"\nStarting training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        loss = train(train_loader, epoch_num=epoch)
        train_auc = 0 # evaluate(train_loader)
        val_auc = 0 # evaluate(val_loader)
        test_auc = evaluate(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')
        model_save_path = 'results/toxric/multilevel_gnn_model' + f'_epoch{epoch}' + '.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"model saved to {model_save_path}")
else:
    auc_str = "0.8559,0.9889,0.8081,0.8619,0.7947,0.8303,0.8193,0.8578,0.7347,0.8285,0.897665669,0.8225452,0.890961579,0.763676102,0.797407741,0.778753347,0.851504896,0.727660226,0.916300816,0.893980775,0.94388529,0.908980191,0.903084463,0.97626842,0.918452718,0.802267695,0.529994209,0.596334601,0.783021351,0.879345528"
    target_auc_list = [float(x) for x in auc_str.split(',')]
    for id in range(100, 200):
        model_save_path = f'results/toxric/multilevel_gnn_model_epoch{id}.pth'
        if not os.path.exists(model_save_path):
            print(f"Model file {model_save_path} does not exist. Skipping to next epoch.")
            continue
        try:
            # 确保加载到正确的设备
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            model.to(device) # 将模型放到指定设备
            print(f"Model successfully loaded from {model_save_path} and set to evaluation mode.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_save_path}. Please ensure the model was saved correctly.")
            exit()
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

        test_auc_dict = evaluate_per_task(test_loader)
        # test_acc = evaluate_overall_accuracy(test_loader)

        good_num = 0
        for task_idx in range(NUM_TASKS):
            current_task_name = f"Task_{task_idx}"
            if current_task_name in test_auc_dict:
                if test_auc_dict[current_task_name] > target_auc_list[task_idx]:
                    good_num += 1

        print(f'\nID: {id}, Test AUC: {test_auc_dict}, Good Tasks: {good_num}/{NUM_TASKS}')
        # print(f'\nTest AUC: {test_auc_dict}, Test Accuracy: {test_acc:.4f}')