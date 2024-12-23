import pandas as pd
import numpy as np
import json
import random

# 文件路径
input_file = "demo_behaviors.json"  # 替换为你的输入 JSON 文件名
output_file = "markov_data_100.json"  # 替换为你的输出 JSON 文件名

# 读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 过滤出 id 在 129 到 519 范围内的对象
filtered_data = [item for item in data if 129 <= item["id"] <= 519]

# 检查范围内的数据是否足够
if len(filtered_data) < 100:
    print(f"范围内的数据不足 100 条，仅有 {len(filtered_data)} 条。")
else:
    # 随机抽取 100 个对象
    random_sample = random.sample(filtered_data, 100)
    
    # 按 id 从小到大排序
    sorted_sample = sorted(random_sample, key=lambda x: x["id"])

    # 保存结果到新的 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_sample, f, ensure_ascii=False, indent=4)

    print(f"随机选取的 100 条数据已保存到文件：{output_file}")