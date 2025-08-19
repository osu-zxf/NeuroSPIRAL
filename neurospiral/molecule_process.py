import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors # Import Descriptors for global features


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