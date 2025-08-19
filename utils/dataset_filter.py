
import pandas as pd

# 读取 CSV 文件
file_path = "result/multi_rules_matched.csv"  # 替换为实际文件路径
df = pd.read_csv(file_path)

# 假设第三列为索引2，进行处理
filtered_rows = []
for index, row in df.iterrows():
    values = row[2].split(",")  # 分隔第三列的内容
    # filtered_values = [v for v in values if 50 <= int(v) <= 100]  # 保留数量在 50 到 100 的值
    if len(values)>=30 and len(values)<=100:  # 如果有符合条件的值
        filtered_rows.append(row)

# 创建新 DataFrame 并保存
filtered_df = pd.DataFrame(filtered_rows, columns=df.columns)
filtered_df.to_csv("result/multi_rules_matched_30_100_filtered_file.csv", index=False)

print("处理完成，保留的行已保存到 'filtered_file.csv' 文件中。")
