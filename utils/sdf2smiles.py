from rdkit import Chem

def sdf_to_smiles(sdf_file):
    """
    将 SDF 文件转换为 SMILES 列表
    :param sdf_file: 输入的 SDF 文件路径
    :return: SMILES 字符串列表
    """
    smiles_list = []
    supplier = Chem.SDMolSupplier(sdf_file)
    
    for mol in supplier:
        if mol is not None:  # 跳过无效分子
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
    
    return smiles_list

# 示例使用
sdf_file_path = "WUXILibrary.sdf"  # 替换为你的 SDF 文件路径
smiles_list = sdf_to_smiles(sdf_file_path)

# 打印或保存 SMILES 列表
# for idx, smiles in enumerate(smiles_list):
#     print(f"{idx + 1}: {smiles}")

with open("output_smiles.txt", "w") as f:
    f.write("\n".join(smiles_list))
