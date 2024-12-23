import numpy as np

np.set_printoptions(formatter={'float': '{:0.3f}'.format})

# 加载 matrix.npy 文件
matrix = np.load("matrix.npy")

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
