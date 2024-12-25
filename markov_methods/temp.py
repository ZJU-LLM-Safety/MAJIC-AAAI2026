import numpy as np

def softmax_normalize_with_temperature(ratios, temperature=1.0): # 对列进行一个归一化
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    ratios = np.array(ratios)
    exp_ratios = np.exp((ratios) / temperature) # 我们的数值小 不用担心溢出
    return exp_ratios / np.sum(exp_ratios)

# 设置浮点数打印格式
np.set_printoptions(formatter={'float': '{:0.3f}'.format})

# 加载 matrix.npy 文件
matrix = np.load("matrix_T_0.1.npy")

# 打印 matrix[1] 到 matrix[10] 的值
print("matrix[1] 到 matrix[10] 的值:")
for i in range(0, 10):
    print(f"matrix[{i}] = {matrix[i]}")

# 计算每一行的加和
row_sums = np.sum(matrix, axis=1)

# 打印每一行的加和
print("每一行的加和:")
for i, row_sum in enumerate(row_sums):
    print(f"matrix[{i}] 的加和 = {row_sum:.3f}")

# 计算每一列的加和
column_sums = np.sum(matrix, axis=0)

# 打印每一列的加和
print("每一列的加和(1*10 列表):")
print(column_sums)

templist = softmax_normalize_with_temperature(column_sums,temperature=1.0)
print("templist is ",templist)