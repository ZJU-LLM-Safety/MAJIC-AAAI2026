import os
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# 确保下载 nltk 的必要资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
    """
    文本预处理函数：去除标点符号、停用词、小写化等。
    """
    # 转小写
    text = text.lower()
    # 去除标点符号和非字母字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_goals(directory, file_prefix):
    """
    提取所有 JSON 文件中 'best_score' 为 1.0 的项的 'goal' 字段。
    """
    goals_by_file = {}

    # 按字典序获取文件
    files = sorted([f for f in os.listdir(directory) if f.startswith(file_prefix) and f.endswith(".json")])
    
    for filename in files:
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    goals = []
                    for entry in data:
                        if entry.get("best_score") == 1.0:
                            goal = entry.get("goal", "").strip()
                            if goal:
                                goals.append(goal)
                    if goals:
                        goals_by_file[filename] = goals
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"无法读取文件 {filename}: {e}")
    
    return goals_by_file

def cluster_goals(goals_by_file, num_clusters=5):
    """
    对提取的 'goal' 文本进行聚类分析。
    """
    # 合并所有文件的 goals
    all_goals = []
    goal_sources = []  # 记录每个 goal 来源于哪个文件
    for filename, goals in goals_by_file.items():
        all_goals.extend(goals)
        goal_sources.extend([filename] * len(goals))
    
    # 文本预处理
    preprocessed_goals = [preprocess_text(goal) for goal in all_goals]
    
    # 特征提取：TF-IDF
    vectorizer = TfidfVectorizer(max_features=500)  # 限制最多 500 个特征
    X = vectorizer.fit_transform(preprocessed_goals)
    
    # 聚类：K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 输出聚类结果
    clustered_goals = {}
    for idx, label in enumerate(labels):
        if label not in clustered_goals:
            clustered_goals[label] = []
        clustered_goals[label].append((all_goals[idx], goal_sources[idx]))
    
    return clustered_goals

if __name__ == "__main__":
    # 指定存放 JSON 文件的目录
    directory = "./"  # 当前目录
    # 指定文件名前缀
    file_prefix = "results_1_gpt4o_o"

    # 提取 'goal' 字段
    goals_by_file = extract_goals(directory, file_prefix)

    # 打印提取的目标
    print("提取到的 'goal' 数据:")
    for filename, goals in goals_by_file.items():
        print(f"{filename}: {goals} (总数: {len(goals)})")

    # 聚类分析
    print("\n正在对 'goal' 文本进行聚类分析...")
    num_clusters = 10  # 设置聚类数量
    clustered_goals = cluster_goals(goals_by_file, num_clusters=num_clusters)

    # 输出聚类结果
    print("\n聚类结果:")
    for cluster_id in sorted(clustered_goals.keys()):  # 按键排序
        print(f"\n聚类 {cluster_id}:")
        for goal, source in clustered_goals[cluster_id]:
            print(f"- 来源文件: {source} | goal: {goal}")