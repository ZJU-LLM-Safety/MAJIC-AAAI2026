# import os
# import json

# # 定义文件夹路径
# input_folder = "markov_results"
# output_folder = "markov_failed_results"

# # 确保输出文件夹存在，如果不存在则创建
# os.makedirs(output_folder, exist_ok=True)

# # 遍历输入文件夹中的所有 JSON 文件
# for filename in os.listdir(input_folder):
#     # 检查文件是否是 JSON 文件
#     if filename.endswith(".json"):
#         input_file_path = os.path.join(input_folder, filename)
#         output_file_path = os.path.join(output_folder, filename)
        
#         # 打开并读取 JSON 文件
#         with open(input_file_path, "r", encoding="utf-8") as file:
#             try:
#                 data = json.load(file)
                
#                 # 如果 JSON 文件是一个列表，则筛选其中的数据
#                 if isinstance(data, list):
#                     filtered_data = [item for item in data if item.get("score") != 1.0]
                    
#                     # 如果筛选结果不为空，则保存到输出文件夹
#                     if filtered_data:
#                         with open(output_file_path, "w", encoding="utf-8") as outfile:
#                             json.dump(filtered_data, outfile, indent=4, ensure_ascii=False)
#                         print(f"已保存筛选结果到: {output_file_path}")
#                     else:
#                         print(f"文件 {filename} 中没有符合条件的数据，未保存。")
#                 else:
#                     print(f"文件 {filename} 的内容不是列表，跳过处理。")
#             except json.JSONDecodeError:
#                 print(f"文件 {filename} 无法解析为 JSON，跳过处理。")

import os
import json

# 定义读取和保存的文件夹路径
input_folder = "markov_failed_results"

# 遍历文件夹中的所有 JSON 文件
for filename in os.listdir(input_folder):
    # 确保只处理 JSON 文件
    if filename.endswith(".json"):
        input_file_path = os.path.join(input_folder, filename)
        
        # 读取 JSON 文件内容
        with open(input_file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                
                # 确保文件内容是一个列表
                if isinstance(data, list):
                    # 为每个单元格添加递增的 "num" 字段
                    for idx, item in enumerate(data):
                        item["num"] = idx  # 添加 "num" 字段，从 0 开始递增

                    # 保存修改后的数据回原文件
                    with open(input_file_path, "w", encoding="utf-8") as outfile:
                        json.dump(data, outfile, indent=4, ensure_ascii=False)
                    print(f"已处理并保存文件: {input_file_path}")
                else:
                    print(f"文件 {filename} 的内容不是列表，跳过处理。")
            except json.JSONDecodeError:
                print(f"文件 {filename} 无法解析为 JSON，跳过处理。")



