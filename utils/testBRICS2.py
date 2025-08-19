from rdkit import Chem
from rdkit.Chem import BRICS

smiles = "CCOC(=O)C(OCC)(c1ccccc1)c1ccccc1"

mol = Chem.MolFromSmiles(smiles)  # 转换 SMILES 为分子对象
if mol:
    # 使用 BRICS 分割分子
    frag_set = BRICS.BRICSDecompose(mol)

print(frag_set)