from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from tqdm import tqdm
from judgeutils import get_jailbreak_score,judge_llama3,judge_gpt
import os
from m1_hypo_attackLLM import hypo_method
from m2_history_attackLLM import history_method
from m3_space_attackLLM import space_method
from m4_dialogue_attackLLM import dialogue_method
from m5_security_attackLLM import security_method
from m6_word_attackLLM import word_method
from m7_char_attackLLM import char_method
from m8_literary_attackLLM import literary_method
from m9_language_attackLLM import language_method
from m10_emoji_attack import emoji_method
import random
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from openai import OpenAI   

def get_attacker_model_inference_pipeline(model_id = "meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        temperature=1.00
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024,do_sample=True,temperature=0.9)
    return pipe
def get_model_inference_pipeline(model_id = "meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
    return pipe

# 从json文件中读取的是json文件的每个单元格，并返回单元格组成的列表
def get_prompts(file_name):
    data_list = []
    with open(file_name, 'r') as file:
        data = json.load(file)
        for item in data:
            data_list.append(item)
    return data_list

def select_init_method(init_num,harmful_prompt,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,judgetype,attacktype,iter_num):
    """
    根据传入的 init_num 选择并调用对应的方法。
    """
    methods = {
        1: lambda: hypo_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype,attacktype,iter_num),
        2: lambda: history_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num),
        3: lambda: space_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num),
        4: lambda: dialogue_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num),
        5: lambda: security_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num),
        6: lambda: word_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num),
        7: lambda: char_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num),
        8: lambda: literary_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num),
        9: lambda: language_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num),
        10: lambda: emoji_method(harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num),
    }

    if init_num in methods:
        return methods[init_num]()  # 调用对应的方法
    else:
        raise ValueError(f"Unsupported init_num: {init_num}. Please provide a number between 1 and 10.")
    
def select_optimize_method(failed_num,optimize_num,harmful_prompt,disguised_failed_sentence,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,judgetype,attacktype,iter_num):
    best_score = -0.1
    best_disguised_prompt = "xxx"
    best_victim_response = "xxx"
    
    for i in range(iter_num):
        harmful_prompt = harmful_prompt
        disguised_failed_sentence = disguised_failed_sentence
        simple_description = df[df['id'] == failed_num]['simple_description'].iloc[0]
        optimize_description = df[df['id'] == optimize_num]['optimize_description'].iloc[0]
        optimize_prompt = simple_description + '\n' + optimize_description
        optimize_prompt = optimize_prompt.replace("{harmful_prompt}", harmful_prompt)
        optimize_prompt = optimize_prompt.replace("{disguised_failed_sentence}", disguised_failed_sentence)
        print("optimize_prompt: ", optimize_prompt)
        optimize_messages = [
            {"role": "user", "content": optimize_prompt},
        ]
        optimized_disguised_prompt = attacker_pipe(optimize_messages)[0]['generated_text'][1]['content']
        optimized_disguised_prompt = optimized_disguised_prompt.strip().strip('\"')
        victim_messages = [
            {"role": "user", "content": optimized_disguised_prompt},
        ]
        
        if attacktype == "local":
            victim_response = victim_pipe(victim_messages)[0]['generated_text'][1]['content']
        else :    
            API_SECRET_KEY= "sk-wvdsTSCg4Edw5do22c734f65145349A99b48Ed06202cD59a" # 填写我们给您的apikey
            BASE_URL = "https://api.ai-gaochao.cn/v1"
            gpt_client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
            victim_messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": optimized_disguised_prompt},
            ]
            try:
                victim_response = gpt_client.chat.completions.create(model=attacktype, messages=victim_messages, max_tokens=512).choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")
                victim_response = "api error"
            finally:
                import time
                time.sleep(2)
        
        if judgetype == "gpt":
            jailbreak_score = judge_gpt(victim_pipe, harmful_prompt, optimized_disguised_prompt, victim_response, victim_tokenizer)
        elif judgetype == "llama3":
            jailbreak_score_1 = get_jailbreak_score(victim_pipe, harmful_prompt, optimized_disguised_prompt, victim_response, victim_tokenizer)
            jailbreak_score_2 = get_jailbreak_score(victim_pipe, harmful_prompt, optimized_disguised_prompt, victim_response, victim_tokenizer)
            jailbreak_score = (jailbreak_score_1 + jailbreak_score_2) / 2
            
        if jailbreak_score > best_score:
            best_score = jailbreak_score
            best_disguised_prompt = optimized_disguised_prompt
            best_victim_response = victim_response
        if best_score == 1.0:
            break
    
    return best_score,best_disguised_prompt,best_victim_response
           
def generate_number_by_probability(init_vector):
    if not (len(init_vector) == 10 and abs(sum(init_vector) - 1) < 1e-2):
        raise ValueError("init_vector must be a list of 10 probabilities summing to 1.")

    return random.choices(range(1, 11), weights=init_vector, k=1)[0]

def softmax_normalize_with_temperature(ratios, temperature=1.0): # 对列进行一个归一化
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    ratios = np.array(ratios)
    exp_ratios = np.exp((ratios) / temperature) # 我们的数值小 不用担心溢出
    return exp_ratios / np.sum(exp_ratios)

# 如果有其他顶层代码，在这里加上
if __name__ == '__main__':
    
    # 1.initialization
    demo_item_list = get_prompts("../data/harmful_behaviors_50.json")
    results_path = "../results/gpt-3.5-turbo/markov_test.json"
    
    # demo_item_list = get_prompts("/data/home/Weiwei/code/jailbreak/results/llama3/markov_test_12-25.json")

    attacker_model_id = "/hub/huggingface/models/jbmodels/mistral-7B-Instruct-v0.2"
    attacker_pipe = get_attacker_model_inference_pipeline(attacker_model_id)
    attacker_tokenizer = AutoTokenizer.from_pretrained(attacker_model_id)

    victim_model_id = "/hub/huggingface/models/jbmodels/llama3-8b-it"
    # victim_model_id = "/hub/huggingface/models/jbmodels/gemma-2-9b-it"
    # victim_model_id = "/hub/huggingface/models/jbmodels/Llama-2-7b-chat-hf"
    victim_pipe = get_model_inference_pipeline(victim_model_id)
    victim_tokenizer = AutoTokenizer.from_pretrained(victim_model_id)

    matrix = np.load("matrix_T_0.1.npy")
    print("matrix is: ", matrix)
    column_sums = np.sum(matrix, axis=0)
    init_vector = softmax_normalize_with_temperature(column_sums, temperature=1.0)
    print("init vector is ",init_vector)
    df = pd.read_csv('methods.csv')
    
    
    # testnum = 1
    chain_count = 3
    init_qnum = 2
    chain_qnum = 2
    chain_length = 3
    index = 0
    # 2. attack
    # attack_type = "local"
    attack_type = "gpt-3.5-turbo-0613"
    for item in tqdm(demo_item_list):
        item_query = 0
        chain_score = -0.1
        chain_disguised_prompt = "xxx"
        chain_victim_response = "xxx"
        
        for k in range(chain_count):
            # testnum += 1
            # if testnum > 10:
            #     break
            # 2.1初始化
            harmful_prompt = item["goal"] # 获取goal字段的值
            # init_vector = [0,0,0,0,0,0,0,1,0,0] # 暂时手动设置
            # init_vector = [0.2,0.05,0.05,0,0,0,0,0.5,0,0.2] # 暂时手动设置
            # init_vector = [0,0,0,0,0,0,0,1,0,0] # 暂时手动设置
            init_num = generate_number_by_probability(init_vector) # 返回1~10的方法序号
            print("init_num is: ", init_num)
            item_query += init_qnum
            chain_score,chain_disguised_prompt,chain_victim_response = select_init_method(init_num,harmful_prompt,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,"gpt",attack_type,iter_num=init_qnum)
            print("init_chain_score: ", chain_score)
            print("init_chain_disguised_prompt: ", chain_disguised_prompt)
            print("init_chain_victim_response: ", chain_victim_response)
            if chain_score == 1.0:
                item["best_score"] = chain_score
                item["best_disguised_prompt"] = chain_disguised_prompt
                item["best_victim_response"] = chain_victim_response
                item["type"] = "init"
                with open(results_path, 'w') as file:
                    json.dump(demo_item_list, file, indent=4)  
                break
            # optimize,opnums是一条链的最大长度
            failed_num = init_num
            failed_disguised_prompt = chain_disguised_prompt
            failed_score = chain_score
            
        # for k in range(chain_count):
            for i in range(chain_length):
                optimize_vector = matrix[failed_num-1]
                print("optimize_vector is: ", optimize_vector)
                optimize_num = generate_number_by_probability(optimize_vector)
                print("optimize_num is: ", optimize_num)
                item_query += chain_qnum
                chain_score,chain_disguised_prompt,chain_victim_response = select_optimize_method(failed_num,optimize_num,harmful_prompt,failed_disguised_prompt,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,"gpt",attack_type,iter_num=chain_qnum)
                if chain_score == 1.0:
                    item["best_score"] = chain_score
                    item["best_disguised_prompt"] = chain_disguised_prompt
                    item["best_victim_response"] = chain_victim_response
                    item["type"] = "markov"
                    break
                else:
                    failed_num = optimize_num
                    failed_disguised_prompt = chain_disguised_prompt
            
            if chain_score == 1.0:
                break    
                
        item["best_score"] = chain_score
        item["best_disguised_prompt"] = chain_disguised_prompt
        item["best_victim_response"] = chain_victim_response
        item["query_num"] = item_query
        # item["type"] = "markov"
        
        # print("Average Jailbreak Score: ", avg_jailbreak_score)
        with open(results_path, 'w') as file:
            json.dump(demo_item_list, file, indent=4)     
    
