from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from tqdm import tqdm
from judgeutils import get_jailbreak_score,judge_llama3,judge_gpt
import os

# Import all attack methods

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



os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
def get_victim_model_inference_pipeline(model_id = "meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
    return pipe

attacker_model_id = "/hub/huggingface/models/jbmodels/mistral-7B-Instruct-v0.2"
attacker_pipe = get_attacker_model_inference_pipeline(attacker_model_id)
attacker_tokenizer = AutoTokenizer.from_pretrained(attacker_model_id)

victim_model_id = "/hub/huggingface/models/jbmodels/llama3-8b-it"
victim_pipe = get_victim_model_inference_pipeline(victim_model_id)
victim_tokenizer = AutoTokenizer.from_pretrained(victim_model_id)

# 从json文件中读取的是json文件的每个单元格，并返回单元格组成的列表
def get_prompts(file_name):
    data_list = []
    with open(file_name, 'r') as file:
        data = json.load(file)
        for item in data:
            data_list.append(item)
    return data_list

demo_item_list = get_prompts("./data/harmful_behaviors_custom_1_25.json")

avg_jailbreak_score = 0.0
index = 0
success_num = 0
judgetype = "gpt"

for item in tqdm(demo_item_list):
    index += 1
    harmful_prompt = item["goal"]
    temp_score = -0.1
    temp_disguised_prompt = "xxx"
    temp_victim_response = "xxx"
    
    # 第一次用文学方式处理
    print("11111111111111111111111111111111")
    
    first_score,first_disguised_prompt,first_victim_response = literary_method(harmful_prompt,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,judgetype,10)
    if(first_score > temp_score):
        temp_score = first_score
        temp_disguised_prompt = first_disguised_prompt
        temp_victim_response = first_victim_response   
        if temp_score == 1.0:
            success_num += 1
            item["disguise_sentence"] = temp_disguised_prompt
            item["victim_response"] = temp_victim_response
            item["score"] = temp_score
            item["method"] = "literary"
            item["query"] = "10"
            avg_jailbreak_score += temp_score
            print("index: ",index)
            print("temp_score: ",temp_score)
            continue
    
    # 第二次用emoji方式处理
    # 2.1叠加emoji
    print("222222222222222222222222222222")
    second_score,second_disguised_prompt,second_victim_response = emoji_method(first_disguised_prompt,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,judgetype,5)
    if(second_score > temp_score):
        temp_score = second_score
        temp_disguised_prompt = second_disguised_prompt
        temp_victim_response = second_victim_response   
        if temp_score == 1.0:
            success_num += 1
            item["disguise_sentence"] = temp_disguised_prompt
            item["victim_response"] = temp_victim_response
            item["score"] = temp_score
            item["method"] = "emoji"
            item["query"] = "15"
            avg_jailbreak_score += temp_score
            print("index: ",index)
            print("temp_score: ",temp_score)
            continue
        
    # 第三次用hypo方式处理
    # 3.1叠加hypo
    print("3333333333333333333333")
    third_score,third_disguised_prompt,third_victim_response = hypo_method(second_disguised_prompt,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,judgetype,5)
    if(third_score > temp_score):
        temp_score = third_score
        temp_disguised_prompt = third_disguised_prompt
        temp_victim_response = third_victim_response   
        if temp_score == 1.0:
            success_num += 1
            item["disguise_sentence"] = temp_disguised_prompt
            item["victim_response"] = temp_victim_response
            item["score"] = temp_score
            item["method"] = "hypo"
            item["query"] = "20"
            avg_jailbreak_score += temp_score
            print("index: ",index)
            print("temp_score: ",temp_score)
            continue

avg_jailbreak_score /= len(demo_item_list)

print("Success Number is: ", success_num)
print("Average Jailbreak Score: ", avg_jailbreak_score)
with open("./results/results_2xgpt4o_framework_easyiter_1_25.json", 'w') as file:
    json.dump(demo_item_list, file, indent=4) 
    
    
    




