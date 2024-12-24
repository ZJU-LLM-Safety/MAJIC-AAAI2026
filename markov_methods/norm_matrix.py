import os
import json
import numpy as np

def calculate_best_score_ratios(folder_path):
    """
    计算每个 JSON 文件中 `best_score` 为 1.0 的比例，只处理以 `results_f` 开头的 JSON 文件
    """
    ratios = []
    # 筛选以 `results_f` 开头且以 `.json` 结尾的文件
    for file_name in sorted([f for f in os.listdir(folder_path) if f.startswith('results_f') and f.endswith('.json')]):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not data:
                ratios.append(0.0)  # 如果文件为空，比例为 0
                continue
            total_count = len(data)
            best_score_count = sum(1 for item in data if item.get("best_score") == 1.0)
            ratio = best_score_count / total_count if total_count > 0 else 0.0
            ratios.append(ratio)
    
    # 将第一个比例移到最后
    if ratios:
        ratios = ratios[1:] + ratios[:1]
    
    return ratios

def softmax_normalize(ratios):
    """
    使用 Softmax 对比例进行归一化
    """
    exp_ratios = np.exp(ratios)  # 对每个值取指数
    return exp_ratios / np.sum(exp_ratios)  # 归一化

def softmax_normalize_with_temperature(ratios, temperature=1.0):
    """
    使用 Softmax 对比例进行归一化，并引入温度系数
    参数:
        ratios: 输入的一维列表或数组，表示需要归一化的比例
        temperature: 温度系数，控制分布的平滑程度，默认为 1.0
        T>1 会使分布更加平滑，所有值的概率变得接近；
        T<1 会使分布更加尖锐，大值进一步放大，小值进一步缩小。
    返回:
        归一化后的概率分布（以 NumPy 数组形式返回）
    """
    # 防止温度为零，避免除零错误
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    
    # 将输入的列表转换为 NumPy 数组（如果尚未是数组）
    ratios = np.array(ratios)
    
    # 数值稳定性处理：减去最大值以避免指数计算时的溢出
    exp_ratios = np.exp((ratios - np.max(ratios)) / temperature)
    
    # 归一化
    return exp_ratios / np.sum(exp_ratios)

def sum_normalize(ratios):
    """
    使用加和占比进行归一化
    """
    total = sum(ratios)  # 计算总和
    if total == 0:
        # 如果所有值加起来为 0，则返回全 0
        return [0.0 for _ in ratios]
    return [x / total for x in ratios]  # 每个值除以总和，计算占比

def power_normalize(ratios, gamma=2):
    """
    使用幂次放大归一化，γ 控制差异放大程度
    """
    amplified = [x ** gamma for x in ratios]  # 对比例进行幂次放大
    total = sum(amplified)  # 计算总和
    if total == 0:
        return [0.0 for _ in ratios]  # 如果总和为 0，返回全 0
    return [x / total for x in amplified]  # 归一化

def main():
    # 定义10个文件夹路径
    base_path = './'  # 假设文件夹在当前目录下
    folder_names = [f"f{i+1}_{suffix}" for i, suffix in enumerate(
        ["hypo", "history", "space", "dialogue", "security", "word", "char", "literary", "language", "emoji"]
    )]
    
    all_normalized_ratios = []  # 用于存储每个文件夹的归一化比例

    for folder_name in folder_names:
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"文件夹 {folder_name} 不存在，跳过")
            all_normalized_ratios.append([0.0] * 10)  # 如果文件夹不存在，用全 0 填充
            continue
        # 计算每个 JSON 文件的 `best_score` 比例
        ratios = calculate_best_score_ratios(folder_path)
        print(f"文件夹 {folder_name} 中的比例：{ratios}")
        # 使用 Softmax 进行归一化
        # normalized_ratios = softmax_normalize(ratios)
        # normalized_ratios = power_normalize(ratios)
        normalized_ratios = softmax_normalize_with_temperature(ratios, temperature=0.1)
        # normalized_ratios = sum_normalize(ratios)
        all_normalized_ratios.append(normalized_ratios)

    # 转换为 NumPy 矩阵
    matrix = np.array(all_normalized_ratios)
    # 格式化矩阵中每个数为三位小数
    matrix = np.round(matrix, decimals=3)
    np.save("matrix_t.npy", matrix)
    print("矩阵已保存为 matrix.npy 文件")

    # 打印矩阵，保留三位小数
    for row in matrix:
        formatted_row = ["{:.3f}".format(x) for x in row]  # 格式化每个值为三位小数
        print(" ".join(formatted_row))

if __name__ == "__main__":
    main()