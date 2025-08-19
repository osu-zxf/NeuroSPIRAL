import os
import pandas as pd
from glob import glob
from functools import reduce

# 1. 读取所有 csv 文件路径
data_dir = '/datapool/data2/home/majianzhu/xuefeng/reuse/datasets/toxric_datasets'   # 你的csv文件夹路径
csv_files = glob(os.path.join(data_dir, '*.csv'))

tox_tables = []
tox_labels = []

# 2. 处理每个 csv 文件
for file in csv_files:
    toxicity_type = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    if not set(['TAID','Canonical SMILES','Toxicity Value']).issubset(df.columns):
        print(f"文件{file}缺少必要列，已跳过。")
        continue
    sub_df = df[['TAID','Canonical SMILES','Toxicity Value']].copy()
    # 标准化 Toxicity Value 为 0/1
    sub_df[toxicity_type] = sub_df['Toxicity Value'].astype(str).str.lower().map(
        lambda x: 1 if x in ['1','yes','true','positive'] else 0
    )
    tox_tables.append(sub_df[['TAID', 'Canonical SMILES', toxicity_type]])
    tox_labels.append(toxicity_type)

# 3. 合并所有毒性表（外连接）
summary_df = reduce(
    lambda left, right: pd.merge(left, right, on=['TAID', 'Canonical SMILES'], how='outer'),
    tox_tables
)

# 4. 改列名 & 保存
summary_df.rename(columns={'TAID': 'ID', 'Canonical SMILES': 'SMILES'}, inplace=True)
summary_df.to_csv('./data/TOXRIC/toxicity_summary.csv', index=False)
print("汇总完成，已保存为toxicity_summary.csv（缺失为NaN）")