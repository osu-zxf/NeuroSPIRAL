from rdkit import Chem
import pandas as pd
from tqdm import trange,tqdm
import os
from glob import glob

#########################  load all rules and validate smarts ####################################

# rules_file_path = './RulesAll.csv'
# rules_data = pd.read_csv(rules_file_path)
# valid_rules = []
# for i in trange(len(rules_data)):
#     rule_smarts_item = rules_data.iloc[i, 1]
#     if pd.isna(rule_smarts_item):
#         continue  # 跳过空值
#     rule_smarts = rule_smarts_item.strip()

#     try:
#         pattern = Chem.MolFromSmarts(rule_smarts)
#     except Exception as e:
#         print(f"rule {i}: {rule_smarts} is invalid")
#         continue

#     if pattern is None:
#         print(f"rule {i}: {rule_smarts} is invalid")
#         continue

#     endpoint_item = rules_data.iloc[i, 2]
#     if pd.isna(endpoint_item):
#         continue  # 跳过空值
#     endpoint = endpoint_item.strip()
#     if 'Acute' in endpoint:
#         endpoint_type = 'A'
#     elif 'Develop' in endpoint:
#         endpoint_type = 'D'
#     elif 'Endocrine' in endpoint:
#         endpoint_type = 'E'
#     elif 'carcinogenicity' in endpoint:
#         endpoint_type = 'C'
#     else:
#         continue  # 跳过不是急性或慢性的规则

#     valid_rules.append((endpoint_type, rule_smarts))

# with open('./result/valid_rules.csv', 'w') as f:
#     f.write("endpoint,SMARTS\n")
#     for endpoint_type, rule_smarts in valid_rules:
#         f.write(f"{endpoint_type},\"{rule_smarts}\"\n")

######################## rules and mols matching #####################################

valid_rules_file_path = './result/valid_rules.csv'
valid_rules_data = pd.read_csv(valid_rules_file_path)
valid_rules = []
for i in trange(len(valid_rules_data)):
    endpoint_item = valid_rules_data.iloc[i, 0]
    if pd.isna(endpoint_item):
        continue  # 跳过空值
    endpoint_type = endpoint_item.strip()
    
    rule_smarts_item = valid_rules_data.iloc[i, 1]
    if pd.isna(rule_smarts_item):
        continue  # 跳过空值
    rule_smarts = rule_smarts_item.strip()
    
    pattern = Chem.MolFromSmarts(rule_smarts)

    valid_rules.append((endpoint_type, rule_smarts, pattern))

a_files = ['./TOXRIC_ALL/Acute Toxicity.csv']
d_files = ['./TOXRIC_ALL/Developmental Toxicity.csv']
e_files = ['./TOXRIC_ALL/Endocrine Disruption.csv']
c_files = ['./TOXRIC_ALL/Carcinogenicity.csv']
other_files = ['TOXRIC_ALL/Cardiotoxicity.csv', 'TOXRIC_ALL/Clinical Toxicity.csv', 'TOXRIC_ALL/CYP450.csv',
               'TOXRIC_ALL/Ecotoxicity.csv', 'TOXRIC_ALL/Genotoxicity.csv', 'TOXRIC_ALL/Hepatotoxicity.csv', 
               'TOXRIC_ALL/Irritation and Corrosion.csv', 'TOXRIC_ALL/Reproductive Toxicity.csv', 'TOXRIC_ALL/Respiratory Toxicity.csv']

a_smiles = []
d_smiles = []
e_smiles = []
c_smiles = []
other_smiles = []

def getSmiles(file_paths):
    smiles = []
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        for i in trange(len(data)):
            smile = data.iloc[i, 4]
            if pd.isna(smile):
                continue  # 跳过空值
            smiles.append(smile)
    return smiles

a_smiles = getSmiles(a_files)
a_smiles = list(set(a_smiles))
d_smiles = getSmiles(d_files)
d_smiles = list(set(d_smiles))
e_smiles = getSmiles(e_files)
e_smiles = list(set(e_smiles))
c_smiles = getSmiles(c_files)
c_smiles = list(set(c_smiles))
other_smiles = getSmiles(other_files)
other_smiles = list(set(other_smiles))

with open('./result/tox_smiles_dataset_stat.csv', 'w') as f:
    f.write('Endpoint,Count\n')
    f.write(f"A,{len(a_smiles)}\n")
    f.write(f"D,{len(d_smiles)}\n")
    f.write(f"E,{len(e_smiles)}\n")
    f.write(f"C,{len(c_smiles)}\n")
    f.write(f"U,{len(other_smiles)}\n")

lists = {
    "A": a_smiles,
    "D": d_smiles,
    "E": e_smiles,
    "C": c_smiles,
    "U": other_smiles
}

########################### smiles & endpoint label to file ##################################

# string_occurrences = {}

# for label, lst in lists.items():
#     for item in lst:
#         if item not in string_occurrences:
#             string_occurrences[item] = []
#         string_occurrences[item].append(label)

# with open('./result/tox_smiles_dataset.csv', 'w') as f:
#     f.write("SMILES,Endpoint\n")
#     for smiles, labels in string_occurrences.items():
#         label = ",".join(list(set(labels)))  # 将列表转换为逗号分隔的字符串
#         f.write('%s,\"%s\"\n' % (smiles, label))

#############################################################

a_mols = [(smiles, Chem.MolFromSmiles(smiles)) for smiles in a_smiles]
d_mols = [(smiles, Chem.MolFromSmiles(smiles)) for smiles in d_smiles]
e_mols = [(smiles, Chem.MolFromSmiles(smiles)) for smiles in e_smiles]
c_mols = [(smiles, Chem.MolFromSmiles(smiles)) for smiles in c_smiles]
other_mols = [(smiles, Chem.MolFromSmiles(smiles)) for smiles in other_smiles]

def isMatched(pattern, mol):
    if mol and mol.HasSubstructMatch(pattern):
        return True
    return False

def matchedMols(pattern, mols):
    matches = []
    for mol in tqdm(mols):
        if isMatched(pattern, mol[1]):
            matches.append(mol[0])
    return matches

matches_results_A = []
matches_results_D = []
matches_results_E = []
matches_results_C = []

for pattern_item in valid_rules:
    pattern = pattern_item[2]
    ep_type = pattern_item[0]
    if ep_type == 'A':
        matches = matchedMols(pattern, a_mols)
        if len(matches) > 0:
            matches_results_A.append((pattern_item[1], matches))
    elif ep_type == 'D':
        matches = matchedMols(pattern, d_mols)
        if len(matches) > 0:
            matches_results_D.append((pattern_item[1], matches))
    elif ep_type == 'E':
        matches = matchedMols(pattern, e_mols)
        if len(matches) > 0:
            matches_results_E.append((pattern_item[1], matches))
    elif ep_type == 'C':
        matches = matchedMols(pattern, c_mols)
        if len(matches) > 0:
            matches_results_C.append((pattern_item[1], matches))

# 统计所有元素的出现次数
from collections import Counter

def unique_rule_filter(matches_results):
    '''
    matches_results: [(rule_smarts, matches_list), ...]
    '''
    all_items = [item for _, lst in matches_results for item in lst]
    item_counts = Counter(all_items)

    cleaned_data = []
    for key, lst in matches_results:
        unique_items = [item for item in lst if item_counts[item] == 1]  # 只保留唯一出现的元素
        if len(unique_items):
            cleaned_data.append((key, unique_items))
    return cleaned_data

print("unique rule filter before:")
print('A:', len(matches_results_A))
print('D:', len(matches_results_D))
print('E:', len(matches_results_E))
print('C:', len(matches_results_C))

matches_results_A = unique_rule_filter(matches_results_A)
matches_results_D = unique_rule_filter(matches_results_D)
matches_results_E = unique_rule_filter(matches_results_E)
matches_results_C = unique_rule_filter(matches_results_C)

print("unique rule filter after:")
print('A:', len(matches_results_A))
print('D:', len(matches_results_D))
print('E:', len(matches_results_E))
print('C:', len(matches_results_C))

matches_results = []
matches_results.extend([('A', pattern, lst) for pattern, lst in matches_results_A])
matches_results.extend([('D', pattern, lst) for pattern, lst in matches_results_D])
matches_results.extend([('E', pattern, lst) for pattern, lst in matches_results_E])
matches_results.extend([('C', pattern, lst) for pattern, lst in matches_results_C])

with open('./result/multi_rules_matched.csv', 'w') as f:
    f.write("endpoint,SMARTS,matched_SMILES\n")
    for ep_type, rule_smarts, matches in matches_results:
        f.write(f"{ep_type},\"{rule_smarts}\",\"{','.join(matches)}\"\n")

with open('./result/match_statistic.csv', 'w') as f:
    f.write("endpoint,SMARTS,matched,total\n")
    for ep_type, rule_smarts, matches in matches_results:
        if ep_type == 'A':
            f.write(f"{ep_type},\"{rule_smarts}\",{len(matches)},{len(a_mols)}\n")
        elif ep_type == 'D':
            f.write(f"{ep_type},\"{rule_smarts}\",{len(matches)},{len(d_mols)}\n")
        elif ep_type == 'E':
            f.write(f"{ep_type},\"{rule_smarts}\",{len(matches)},{len(e_mols)}\n")
        elif ep_type == 'C':
            f.write(f"{ep_type},\"{rule_smarts}\",{len(matches)},{len(c_mols)}\n")

# folder_path = './TOXRIC_ALL'
# all_matches = []  # 记录所有文件中匹配的SMILES

# cnt = 0
# all_smiles = [] 
# for file_path in glob(os.path.join(folder_path, "*.csv")):
#     print(f"正在处理文件: {file_path}")
#     data = pd.read_csv(file_path)
#     matches = []
#     for i in trange(len(data)):
#         smiles = data.iloc[i, 4]
#         if pd.isna(smiles):
#             continue  # 跳过空值
#         all_smiles.append(smiles)
#         mol = Chem.MolFromSmiles(smiles)
#         if mol and mol.HasSubstructMatch(pattern):
#             matches.append(smiles)
#     if len(matches) > 0:
#         all_matches.append((file_path, matches))
#         cnt += len(matches)

# print(f"rule {0}: matched {cnt}")

# with open('./result/matched.txt', 'w') as f:
#     f.write("文件路径,匹配的SMILES\n")
#     for file_path, matches in all_matches:
#         f.write(f"{file_path},{','.join(matches)}\n")


# mols = []
# for file_path, matches in all_matches:
#     mols.extend(matches)

# mols = list(set(mols))
# print(f"rule {0}: matched {len(mols)}")

# with open('./result/matched_mols.txt', 'w') as f:
#     f.write("匹配的SMILES\n")
#     for mol in mols:
#         f.write(f"{mol}\n")

# all_mols = list(set(all_smiles))
# print(f"total: {len(all_mols)}")

# from rdkit.Chem import Draw
# n = 0
# for smiles in mols:
#     mol = Chem.MolFromSmiles(smiles)
#     img = Draw.MolToImage(mol, size=(300, 300))  # 设置图像大小
#     img.save(f"./result/imgs/mol_{n}.png")             # 保存到文件
#     n+=1