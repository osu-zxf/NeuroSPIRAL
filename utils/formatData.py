task_res = []
with open('results/toxcast/multilevel_gnn_results.txt', 'r') as f:
    read_data = f.readlines()
    for line in read_data:
        data_list = map(float, line.strip().split(','))
        task_res.append(data_list)
print(len(task_res))
with open('results/toxcast/multilevel_gnn_results.csv', 'w') as f:
    for task in task_res:
        f.write(','.join([f'{data:.4f}' for data in task]) + '\n')