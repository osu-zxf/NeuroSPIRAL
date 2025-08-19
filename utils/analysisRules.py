import pandas as pd

data = pd.read_csv('./result/match_statistic.csv')

cnt_list = data['matched'].tolist()

print(cnt_list)



data = [41, 133, 8, 45, 4, 5, 66, 320, 86, 440, 1, 225, 2, 519, 50, 57, 409, 40, 475, 56, 341, 312, 16, 361, 23, 98, 3, 4, 57, 40, 2, 449, 356, 125, 4, 391, 4, 2, 43, 469, 497, 76, 239, 56, 82, 351, 524, 178]
import matplotlib.pyplot as plt

plt.hist(data, bins=30, color='skyblue', edgecolor='black')  # bins参数表示分箱数

plt.xlabel('Number of matched rules')

# 显示图表
plt.show()