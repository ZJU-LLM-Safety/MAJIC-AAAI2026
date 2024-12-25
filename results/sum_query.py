import json

# 假设你的 JSON 文件名为 data.json
file_name = '/data/home/Weiwei/code/jailbreak/results/llama3/markov_test.json' # 402/50 = 8.04 ; ASR = 0.88
# file_name = '/data/home/Weiwei/code/jailbreak/results/gemma-2-9b/markov_test.json' # 330/50 = 6.6 ; ASR =0.54

try:
    # 读取 JSON 文件
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化总和和计数
    total_query_num = 0
    best_score_count = 0

    # 遍历 JSON 对象，计算 query_num 总和和 best_score 等于 1.0 的项数
    for item in data:
        if 'query_num' in item and isinstance(item['query_num'], (int, float)):
            total_query_num += item['query_num']
        if 'best_score' in item and item['best_score'] == 1.0:
            best_score_count += 1

    print(f"所有单元格的 query_num 总和是: {total_query_num}")
    print(f"best_score 等于 1.0 的项数是: {best_score_count}")

except FileNotFoundError:
    print("文件未找到，请确认文件路径和文件名是否正确！")
except json.JSONDecodeError:
    print("JSON 文件格式有误，请检查文件内容！")
except Exception as e:
    print(f"发生错误: {e}")