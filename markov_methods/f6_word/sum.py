import os
import json
from collections import defaultdict

def find_best_score_nums(directory, file_prefix):
    # 存储每个文件的结果
    file_nums = defaultdict(list)
    # 存储所有文件的num值（去重、有序集合）
    all_nums = set()
    # 存储每个文件的 best_score == 1.0 的比例
    file_ratios = {}

    # 按字典序获取所有符合前缀的文件
    files = sorted([f for f in os.listdir(directory) if f.startswith(file_prefix) and f.endswith(".json")])

    for filename in files:
        file_path = os.path.join(directory, filename)
        try:
            # 打开并解析JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 检查JSON是否为列表
                if isinstance(data, list):
                    total_entries = len(data)  # 获取文件中的单元格总数
                    best_score_count = 0  # best_score 为 1.0 的计数器

                    for entry in data:
                        # 检查"best_score"是否为1.0
                        if entry.get("best_score") == 1.0:
                            num = entry.get("num")
                            if num is not None:
                                file_nums[filename].append(num)
                                all_nums.add(num)
                                best_score_count += 1

                    if total_entries > 0:  # 避免除以零
                        file_ratios[filename] = best_score_count / total_entries


        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"无法读取文件 {filename}: {e}")

        # Calculate overall coverage
    total_cells_across_files = 0
    for filename in file_nums:  # Iterate through files that have best_score == 1.0 entries
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
            total_cells_across_files = len(data)  # Accumulate total cells

    overall_coverage = len(all_nums) / total_cells_across_files if total_cells_across_files else 0

    return file_nums, sorted(all_nums), file_ratios, overall_coverage

if __name__ == "__main__":
    # 指定存放JSON文件的目录
    directory = "./"  # 当前目录
    # 指定文件名前缀
    file_prefix = "results_f6o"

    # 调用统计函数
    file_nums, all_nums, file_ratios ,overall_coverage = find_best_score_nums(directory, file_prefix)

    # 打印每个文件的统计结果
    print("每个文件中 'best_score' 为 1.0 的项的 num 值:")
    for filename in sorted(file_nums.keys()):
        nums = sorted(file_nums[filename])
        ratio = file_ratios.get(filename, 0) # 获取比例，如果文件不存在则为0
        print(f"{filename}: {nums} (总数: {len(nums)}, 比例: {ratio:.2%})") # 格式化输出比例

    # 打印所有文件的 num 值（有序集合）
    print("\n所有文件中 'best_score' 为 1.0 的项的 num 值 (有序集合):")
    print(all_nums)
    print(f"总数: {len(all_nums)} (覆盖率: {overall_coverage:.2%})") 