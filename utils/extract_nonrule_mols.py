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

def getCSmiles(file_paths):
    smiles = []
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        for i in trange(len(data)):
            smile = data.iloc[i, 4]
            if pd.isna(smile):
                continue  # 跳过空值
            if data.iloc[i, 6] == 0:
                continue
            smiles.append(smile)
    return smiles

a_smiles = getSmiles(a_files)
a_smiles = list(set(a_smiles))
d_smiles = getSmiles(d_files)
d_smiles = list(set(d_smiles))
e_smiles = getSmiles(e_files)
e_smiles = list(set(e_smiles))
c_smiles = getCSmiles(c_files)
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

def hasMatchedRules(mol, patterns):
    for pattern in patterns:
        if isMatched(pattern, mol):
            return True
    return False

matches_results_A = []
matches_results_D = []
matches_results_E = []
matches_results_C = []

patterns = [pattern for _, _, pattern in valid_rules]

for mol in tqdm(a_mols):
    if hasMatchedRules(mol[1], patterns):
        continue
    matches_results_A.append(mol[0])

for mol in tqdm(d_mols):
    if hasMatchedRules(mol[1], patterns):
        continue
    matches_results_D.append(mol[0])

for mol in tqdm(e_mols):
    if hasMatchedRules(mol[1], patterns):
        continue
    matches_results_E.append(mol[0])

for mol in tqdm(c_mols):
    if hasMatchedRules(mol[1], patterns):
        continue
    matches_results_C.append(mol[0])

matches_results_A = list(set(matches_results_A))
matches_results_D = list(set(matches_results_D))
matches_results_E = list(set(matches_results_E))
matches_results_C = list(set(matches_results_C))

print("non rule filter after:")
print('A:', len(matches_results_A))
print('D:', len(matches_results_D))
print('E:', len(matches_results_E))
print('C:', len(matches_results_C))

with open('./result/no_matched_mols.csv', 'w') as f:
    f.write("endpoint,SMILES\n")
    for smiles in matches_results_A:
        f.write(f"A,{smiles}\n")
    for smiles in matches_results_D:
        f.write(f"D,{smiles}\n")
    for smiles in matches_results_E:
        f.write(f"E,{smiles}\n")
    for smiles in matches_results_C:
        f.write(f"C,{smiles}\n")

with open('./result/no_matched_mols_A.csv', 'w') as f:
    f.write("endpoint,SMILES\n")
    for smiles in matches_results_A:
        f.write(f"{smiles}\n")

with open('./result/no_matched_mols_D.csv', 'w') as f:
    f.write("endpoint,SMILES\n")
    for smiles in matches_results_D:
        f.write(f"{smiles}\n")

with open('./result/no_matched_mols_E.csv', 'w') as f:
    f.write("endpoint,SMILES\n")
    for smiles in matches_results_E:
        f.write(f"{smiles}\n")

with open('./result/no_matched_mols_C.csv', 'w') as f:
    f.write("endpoint,SMILES\n")
    for smiles in matches_results_C:
        f.write(f"{smiles}\n")