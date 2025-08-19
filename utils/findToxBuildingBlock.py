from rdkit import Chem
import pandas as pd
from tqdm import trange,tqdm
import os
from glob import glob

valid_rules = []
rule_smarts_list = ['[NX3]C(=S)[S]', 'c1cc2[nR]ccc2cc1', '[SX2,OX2][Ge][SX2,OX2]', '[CX4][Ge]([Cl,Br,I])([Cl,Br,I])[Cl,Br,I]', '[C]1COCO1[C](=[O,N,S])[Cl,Br,I]']
for rule_smarts in rule_smarts_list:
    pattern = Chem.MolFromSmarts(rule_smarts)
    valid_rules.append((rule_smarts, pattern))

def isMatched(pattern, mol):
    if mol and mol.HasSubstructMatch(pattern):
        return True
    return False

def hasMatchedRules(mol, patterns):
    for pattern in patterns:
        if isMatched(pattern, mol):
            return True
    return False

patterns = [pattern for _, pattern in valid_rules]


# 示例使用
# sdf_file_path = "WUXILibrary.sdf"  # 替换为你的 SDF 文件路径
# mols = Chem.SDMolSupplier(sdf_file_path)

def getMols(file_path):
    smiles = []
    data = pd.read_csv(file_path)
    for i in trange(len(data)):
        smile = data.iloc[i, 1]
        if pd.isna(smile):
            continue  # 跳过空值
        smiles.append(smile)
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles]
    return mols

mols = getMols("result/no_matched_mols.csv")

results = [[] for _ in range(len(valid_rules))]
for mol in tqdm(mols):
    for patt_id in range(len(patterns)):
        if isMatched(patterns[patt_id], mol):
            results[patt_id].append(mol)

for patt_id in range(len(valid_rules)):
    results[patt_id] = list(set(results[patt_id]))
    print(f"Rule {patt_id}: {len(results[patt_id])}")

with open("output_smiles.txt", "w") as f:
    for patt_id in range(len(valid_rules)):
        f.write(f"Rule {patt_id}:\n")
        for mol in results[patt_id]:
            f.write(f"{Chem.MolToSmiles(mol)}\n")
        f.write("\n")