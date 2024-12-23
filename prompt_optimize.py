# 导入所需的模块
import json  # 用于处理 JSON 数据
import os  # 用于与操作系统交互
import openai  # 用于调用 OpenAI 的 API 接口
import re  # 用于字符串操作和正则表达式匹配
import sys  # 提供对 Python 解释器的访问
import math  # 提供数学函数和常量
from copy import deepcopy  # 用于深拷贝对象
import random  # 用于生成随机数
import numpy as np  # 提供数值计算支持，特别是矩阵操作
import pdb  # 调试器模块，用于代码调试
import time  # 提供时间相关的功能

from tqdm import tqdm  # 提供进度条显示功能

from tenacity import (  # 从 tenacity 模块导入相关工具，用于实现重试机制
    retry,  # 装饰器，用于装饰需要自动重试的函数
    stop_after_attempt,  # 设置最大重试次数的策略
    wait_exponential,  # 指数回退策略，控制每次重试的等待时间
)

# 定义一个带有重试机制的函数，用于调用 OpenAI 的 ChatCompletion 接口
@retry(wait=wait_exponential(min=8, max=100), stop=stop_after_attempt(6))  
def Promting(messages, temperature, number):
    """
    调用 OpenAI 的 ChatCompletion 接口，生成多轮对话内容。
    参数：
    - messages：消息列表，包含对话历史和当前输入
    - temperature：生成的随机性控制参数，值越高结果越多样化
    - number：生成的响应数量
    返回：
    - 返回生成的响应内容
    """
    response = openai.ChatCompletion.create(  # 调用 OpenAI ChatCompletion 接口
        model="gpt-3.5-turbo",  # 指定使用的 GPT 模型
        messages=messages,  # 传入对话消息列表
        temperature=temperature,  # 控制生成的随机性
        n=number  # 设置返回的响应数量
    )
    return response['choices']  # 返回生成的多个响应（存储在 choices 中）

# 定义另一个带有重试机制的函数，用于调用 OpenAI 的 Completion 接口
@retry(wait=wait_exponential(min=8, max=100), stop=stop_after_attempt(6))  
def Promting_Dis(messages):
    """
    调用 OpenAI 的 Completion 接口，生成文本补全内容。
    参数：
    - messages：消息列表，包含任务定义和用户输入
    返回：
    - 返回生成的补全内容
    """
    # 初始化前缀字符串，将系统消息的内容作为初始 prompt
    prefix = messages[0]['content'] + '\n\n'
    for i in messages[1:]:  # 遍历用户和助手消息
        if i['role'] == 'user':  # 如果是用户角色的消息
            prefix = prefix + i['content']  # 拼接用户消息到前缀
        else:  # 如果是助手角色的消息
            prefix = prefix + i['content'] + '\n\n'  # 拼接助手消息并添加换行符

    # 调用 OpenAI 的 Completion 接口生成内容
    response = openai.Completion.create(
        engine="text-davinci-002",  # 使用 text-davinci-002 模型
        prompt=prefix,  # 传入构建好的 prompt
        temperature=0,  # 设定随机性为 0，确保生成内容更稳定
        max_tokens=374,  # 设置生成文本的最大长度
        top_p=1,  # 设置 nucleus sampling 参数
        logprobs=1  # 请求返回每个 token 的概率值
    )
    return response['choices'][0]  # 返回第一个生成的补全内容

# 定义一个函数，用于构造 OpenAI ChatCompletion 消息格式
def Construct_Message(task_instruction, instance):
    """
    构造消息列表，用于传递给 ChatCompletion 接口。
    参数：
    - task_instruction：系统消息，描述任务目标
    - instance：用户消息，包含用户输入内容
    返回：
    - 消息列表，符合 OpenAI ChatCompletion 格式
    """
    messages = []  # 初始化空的消息列表
    messages.append({"role": "system", "content": task_instruction})  # 添加系统角色消息
    messages.append({"role": "user", "content": instance})  # 添加用户角色消息
    return messages  # 返回构造好的消息列表

# 定义一个函数，用于替换生成器的定义部分
def replace_definition(new_definition, generator_prompt_temp):
    """
    替换生成器提示语中的定义部分。
    参数：
    - new_definition：新的定义内容
    - generator_prompt_temp：原始生成器提示语
    返回：
    - 更新后的生成器提示语
    """
    generator_prompt_temp[0]['content'] = new_definition  # 将新的定义替换为第一个消息的内容
    return generator_prompt_temp  # 返回更新后的提示语

# 定义一个函数，用于替换生成器的示例部分
def replace_example(new_example, generator_prompt_temp, example_index):
    """
    替换生成器提示语中的示例部分。
    参数：
    - new_example：新的示例内容（包含输入和输出）
    - generator_prompt_temp：原始生成器提示语
    - example_index：要替换的示例索引
    返回：
    - 更新后的生成器提示语
    """
    definition = [generator_prompt_temp[0]]  # 获取提示语中的定义部分
    input = re.findall(r"Input\:\s(.*)\nOutput\:\s", new_example, flags=re.DOTALL)  # 用正则表达式提取输入内容
    output = re.findall(r"Output\:\s(.*)", new_example)  # 用正则表达式提取输出内容

    # 替换指定示例的用户输入
    generator_prompt_temp[2 * example_index + 1] = {"role": "user", "content": input[0]}
    # 替换指定示例的助手输出
    generator_prompt_temp[2 * example_index + 2] = {"role": "assistant", "content": output[0]}

    # 将更新后的定义和示例组合成新的提示语
    generator_prompt_new = definition
    for i in generator_prompt_temp[1:]:  # 遍历原始提示语的其余部分
        generator_prompt_new.append(i)  # 依次添加到新提示语中
    return generator_prompt_new  # 返回更新后的提示语

# 替换鉴别器的示例部分
def replace_discriminator(new_example, discriminator_prompt_ori):
    """
    替换鉴别器提示语中的示例部分。
    参数：
    - new_example：新的示例内容
    - discriminator_prompt_ori：原始鉴别器提示语
    返回：
    - 更新后的鉴别器提示语
    """
    try:
        new_input = new_example.split('\n')  # 将新的示例按换行符分割为行
        temp = new_input[-1].split(':')  # 分割最后一行，提取输入和输出的分隔符
        first = new_input[0] + '\n' + new_input[1] + '\n' + new_input[2] + '\n' + temp[0] + ': '  # 重组输入部分
        second = temp[1].strip(' ')  # 提取输出部分并去除多余空格

        examples = discriminator_prompt_ori  # 获取原始鉴别器提示语的所有示例
        discriminator_prompt_new = [examples[0]]  # 保留原始定义部分
                # 遍历原始示例，替换输入和输出
        for example in examples[1:]:
            if example['role'] == "user":  # 如果角色是用户
                example = example['content'].splitlines(True)  # 将用户消息内容按行分割成列表
                str = ''  # 初始化字符串用于存储修改后的内容
                example[-4] = re.split(r'(\.\s|\?\s|\.|\?)', example[-4])  # 将倒数第四行按标点符号分割
                values = example[-4][::2][:-1]  # 提取文本部分
                delimiters = example[-4][1::2]  # 提取标点符号部分
                for i in range(len(values) - 1):  # 遍历文本部分，重新组合
                    str += values[i] + delimiters[i]
                example[-4] = str + first.strip('\n')  # 更新倒数第四行并去除多余换行符
                str = ''  # 重置字符串
                str = str.join(example[0:-3])  # 将其他行重新组合成一个字符串
                discriminator_prompt_new.append({"role": "user", "content": str})  # 更新用户消息
            elif example['role'] == "assistant":  # 如果角色是助手
                discriminator_prompt_new.append({"role": "assistant", "content": second})  # 更新助手的响应内容
        return discriminator_prompt_new  # 返回更新后的鉴别器提示语
    except:  # 如果出现异常
        return discriminator_prompt_ori[:]  # 返回原始提示语作为回退方案

# 替换鉴别器的定义部分
def replace_definition_dis(definition, discriminator_prompt_ori):
    """
    替换鉴别器提示语中的定义部分。
    参数：
    - definition：新的定义内容
    - discriminator_prompt_ori：原始鉴别器提示语
    返回：
    - 更新后的鉴别器提示语
    """
    examples = discriminator_prompt_ori  # 获取原始鉴别器提示语
    discriminator_prompt_new = [examples[0]]  # 只保留定义部分作为初始值

    # 遍历示例，替换定义内容
    for example in examples[1:]:
        if example['role'] == 'user':  # 如果角色是用户
            example = example['content'].splitlines(True)  # 将内容按行分割
            str = ''  # 初始化字符串用于存储修改后的内容
            example[-4] = re.split(r'(\.\s|\?\s|\.|\?)', example[-4])  # 将倒数第四行按标点符号分割
            values = example[-4][::2][:-1]  # 提取文本部分
            delimiters = example[-4][1::2]  # 提取标点符号部分
            str = ' ' + values[-1] + delimiters[-1]  # 重新组合最后部分
            example[-4] = definition[0] + str.strip('\n') + '\n'  # 更新倒数第四行，添加新的定义内容
            str = ''  # 重置字符串
            str = str.join(example)  # 将修改后的内容重新组合
            discriminator_prompt_new.append({"role": "user", "content": str})  # 更新用户消息
        elif example['role'] == 'assistant':  # 如果角色是助手
            discriminator_prompt_new.append({"role": "assistant", "content": example['content']})  # 保持助手的响应内容不变
    return discriminator_prompt_new  # 返回更新后的鉴别器提示语

# 定义一个函数，用于构造生成器的提示语
def generator_prompt(definition, positive):
    """
    构造生成器提示语，用于生成器模型。
    参数：
    - definition：定义内容列表，定义生成任务的规则和目标
    - positive：包含正例的列表，每个正例包含输入和输出
    返回：
    - 消息列表，用于生成器模型
    """
    task_instruction = definition[0]  # 提取定义的第一部分作为任务说明
    examples = []  # 初始化示例列表
    for instance in positive:  # 遍历所有正例
        examples.append([instance['input'], instance["output"]])  # 将正例的输入和输出添加到示例列表中
    
    messages = []  # 初始化消息列表
    messages.append({"role": "system", "content": task_instruction})  # 将任务说明作为系统消息添加
    for i in range(len(examples)):  # 遍历示例
        messages.append({"role": "user", "content": examples[i][0]})  # 添加用户消息（输入内容）
        messages.append({"role": "assistant", "content": examples[i][1]})  # 添加助手消息（输出内容）

    return messages  # 返回构造的消息列表

# 定义一个函数，用于构造鉴别器的提示语
def discriminator_prompt(definition, positive, negative):
    """
    构造鉴别器提示语，用于鉴别器模型。
    参数：
    - definition：定义内容列表，包含任务规则
    - positive：包含正例的列表
    - negative：包含负例的列表
    返回：
    - 消息列表，用于鉴别器模型
    """
    task_instruction = "Judge the answer is correct ground truth or generated fake answer."  # 鉴别任务说明
    examples = []  # 初始化示例列表
    for instance in positive:  # 遍历所有正例
        # 构造正例的输入字符串，包含输入、输出、定义和问题
        input_instance = (
            "Input: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n' +
            definition[0] + ' Is above output correct ground truth?' +
            "\n(A) Yes, it is correct ground truth.\n(B) No, it is generated fake output.\nThe answer is: "
        )
        output_instance = "(A) Yes, it is correct ground truth."  # 正例的标准答案
        examples.append([input_instance, output_instance])  # 添加正例到示例列表

    messages = []  # 初始化消息列表
    messages.append({"role": "system", "content": task_instruction})  # 添加系统消息（任务说明）
    for i in range(len(examples)):  # 遍历示例
        messages.append({"role": "user", "content": examples[i][0]})  # 添加用户消息（正例的输入内容）
        messages.append({"role": "assistant", "content": examples[i][1]})  # 添加助手消息（正例的输出内容）
    
    return messages  # 返回构造的消息列表

# 定义生成器函数，用于生成新的预测
def generator(generator_prompt, instance):
    """
    使用生成器提示语和输入实例生成输出。
    参数：
    - generator_prompt：生成器的提示语
    - instance：用户输入实例
    返回：
    - 生成器模型的预测结果
    """
    generator_prompt_temp = generator_prompt[:]  # 创建提示语的副本，避免修改原始数据
    generator_prompt_temp.append({"role": "user", "content": instance})  # 将用户输入添加到提示语的末尾

    prediction = Promting(generator_prompt_temp, 0, 1)  # 调用生成器模型，生成单个预测（temperature=0，生成稳定输出）
    # print(prediction)  # 调试用，输出生成的预测结果

    return prediction  # 返回生成的预测结果

# 定义鉴别器函数，用于判断预测结果的真实性
def discriminator(discriminator_prompt, instance, prediction):
    """
    使用鉴别器提示语对生成的预测进行真实性判断。
    参数：
    - discriminator_prompt：鉴别器的提示语
    - instance：用户输入实例
    - prediction：生成器模型的预测结果
    返回：
    - 鉴别器的对数概率（log probability）
    """
    qu = instance.strip('\n')  # 去除输入实例末尾的换行符

    example = discriminator_prompt[1]['content']  # 获取鉴别器的第一个示例
    example = example.splitlines(True)  # 将示例按行分割成列表
    example[-1] = example[-1].split(':')[0] + ': '  # 修改最后一行，保留 ":" 前的部分作为模板
    str1 = ''  # 初始化字符串
    str1 = str1.join(example[-4:])  # 提取示例的最后四行并组合成一个字符串

    # 构造新的鉴别器输入，包含用户输入、生成的预测和问题模板
    prefix = 'Input: ' + str(qu) + '\nOutput: ' + str(prediction) + '\n' + str1

    discriminator_prompt_temp = discriminator_prompt[:]  # 创建鉴别器提示语的副本
    discriminator_prompt_temp.append({"role": "user", "content": prefix})  # 将新的输入添加到提示语末尾

    output = Promting_Dis(discriminator_prompt_temp)['logprobs']  # 调用鉴别器模型，并获取返回的对数概率

    try:
        index_A = output['tokens'].index('A')  # 找到标记为 “A” 的位置
        log_probability = output['top_logprobs'][index_A]['A']  # 提取 "A" 的对数概率
    except:
        try:
            index_B = output['tokens'].index('B')  # 找到标记为 “B” 的位置
            log_probability = output['top_logprobs'][index_B]['B']  # 提取 "B" 的对数概率
            log_probability = math.log(1 - math.exp(log_probability))  # 计算 "A" 的对数概率
        except:
            log_probability = -10  # 如果都找不到，则返回一个较低的默认概率值
    
    return log_probability  # 返回计算的对数概率

# 定义损失函数，用于评估生成器和鉴别器的表现
def Loss(generator_prompt, discriminator_prompt, true_instances, train_instances):
    """
    计算生成器和鉴别器的损失值。
    参数：
    - generator_prompt：生成器的提示语
    - discriminator_prompt：鉴别器的提示语
    - true_instances：包含正确实例的列表
    - train_instances：包含训练实例的列表
    返回：
    - 总损失值
    """
    score = 0  # 初始化损失值

    for instance in true_instances:  # 遍历所有正确实例
        # 使用鉴别器判断正确实例的真实性
        log_probability = discriminator(discriminator_prompt, instance["input"], instance["output"][-1])
        score += log_probability  # 累加对数概率到总分

    for instance in train_instances:  # 遍历所有训练实例
        prediction = generator(generator_prompt, instance["input"])  # 使用生成器生成预测
        log_probability = discriminator(discriminator_prompt, instance["input"], prediction)  # 判断预测的真实性
        score += math.log(1 - math.exp(log_probability))  # 累加生成器预测的损失值

    return score  # 返回总损失值

def Update_generator(generator_prompt_ori, discriminator_prompt, definition, true_instances, train_instances, loss_function, negative_data):
    """
    更新生成器的提示语，优化任务定义和示例以降低损失函数。
    参数：
    - generator_prompt_ori: 原始生成器提示语。
    - discriminator_prompt: 鉴别器的提示语。
    - definition: 当前任务的定义。
    - true_instances: 包含真实数据的实例。
    - train_instances: 用于训练的实例数据。
    - loss_function: 当前的损失值。
    - negative_data: 包含负例的额外数据。
    返回：
    - 更新后的生成器提示语和任务定义。
    """
    # 提取当前任务定义内容
    definition_dis = generator_prompt_ori[0]['content']
    # 定义优化任务说明的目标，要求使任务说明更清晰
    task_instruction = 'Diversify the task instruction to be clearer. Keep the task instruction as declarative.'
    # 构造实例化任务，附加原始定义并准备改进的新定义格式
    instance_task = '\n\nTask instruction: ' + definition_dis + '\n\nImproved task instruction: '
    # 使用工具函数生成用于提示的新消息
    messages = Construct_Message(task_instruction, instance_task)
    # 调用模型生成新的定义集合
    new_definition_set = Promting(messages, 0.4, 5)
    # 初始化损失列表和新定义对应的生成器提示列表
    loss_new_definition_set = []
    generator_prompt_def_set = []

    # 遍历新生成的任务定义集合
    for new_definition in new_definition_set:
        # 去除多余的换行符，保持格式一致
        new_definition = new_definition['message']['content'].strip('\n').replace('\n', ' ')
        print(new_definition)  # 输出新定义，便于调试
        print("-------------")
        # 替换生成器提示中的任务定义
        generator_prompt_def = replace_definition(new_definition, generator_prompt_ori[:])
        generator_prompt_def_set.append(generator_prompt_def[:])  # 存储替换后的生成器提示
        # 计算新定义的损失值
        loss_current = Loss(generator_prompt_def[:], discriminator_prompt[:], true_instances, train_instances)
        loss_new_definition_set.append(loss_current)

    # 找出最低的损失值
    minimum_loss = min(loss_new_definition_set)

    # 如果新定义降低了损失值，则更新生成器提示、定义和损失函数
    if minimum_loss < loss_function:
        generator_prompt_ori = generator_prompt_def_set[loss_new_definition_set.index(minimum_loss)][:]
        definition = [new_definition_set[loss_new_definition_set.index(minimum_loss)]['message']['content'].strip('\n').replace('\n', ' ').replace('\r', ' ')]
        loss_function = minimum_loss
        print((loss_new_definition_set.index(minimum_loss), minimum_loss, generator_prompt_ori))
    else:
        print("Fail Optimization")  # 如果未降低损失值，打印失败信息

    # 提取生成器提示中的示例部分
    examples = generator_prompt_ori[1:]

    # 遍历示例，对每个示例进行优化
    for j in range(int(len(examples) / 2)):
        print("For example")  # 标记当前处理的是第几个示例
        print(j + 1)

        # 构造示例的输入格式，包含 "Input" 和 "Output"
        input_instance = 'Input: ' + examples[2 * j]['content'] + '\nOutput: ' + examples[2 * j + 1]['content']
        # 定义任务目标：优化示例，使其更具代表性
        task_instruction = definition[0] + ' Diversify the example to make it more representative. Keep the format as Input: and Output: .' 
        # 准备优化示例的任务格式
        instance_task = '\n\nExample: ' + input_instance + '\n\nImproved example: '
        # 调用工具生成优化后的示例集合
        messages = Construct_Message(task_instruction, instance_task)
        new_example_set = Promting(messages, 0.4, 5)
        # 初始化存储优化示例损失的列表
        loss_new_example_set = []
        generator_prompt_ex_set = []

        # 遍历新生成的示例集合
        for new_example in new_example_set:
            # 去除多余的换行符，保持格式一致
            new_example = new_example['message']['content'].strip('\n').replace('\n\n', '\n')  
            print(new_example)  # 输出优化后的示例，便于调试
            print("-------------")
            # 使用正则表达式提取 "Input" 和 "Output" 部分
            pattern = r"Input: (.*?)\nOutput: (.*?)\n"
            match = re.findall(pattern, new_example + '\n', re.DOTALL)
            new_example = "Input: " + match[0][0] + "\nOutput: " + match[0][1]
            print(new_example)  # 输出格式化后的示例
            print("=============")
            # 替换生成器提示中的示例
            generator_prompt_ex = replace_example(new_example, generator_prompt_ori[:], j)
            # 计算新示例对应的损失值
            loss_current = Loss(generator_prompt_ex[:], discriminator_prompt[:], true_instances, train_instances)
            loss_new_example_set.append(loss_current)
            generator_prompt_ex_set.append(generator_prompt_ex[:])

        # 找到最小损失值
        minimum_loss = min(loss_new_example_set)

        # 如果优化后的示例降低了损失值，则更新生成器提示和损失函数
        if minimum_loss < loss_function:
            generator_prompt_ori = generator_prompt_ex_set[loss_new_example_set.index(minimum_loss)][:]
            loss_function = minimum_loss
            print((loss_new_example_set.index(minimum_loss), minimum_loss, generator_prompt_ori))
        else:
            print("Fail Optimization")  # 如果未降低损失值，打印失败信息
    
    return generator_prompt_ori, definition  # 返回更新后的生成器提示语和定义

def Update_discriminator(generator_prompt, discriminator_prompt_ori, definition, true_instances, train_instances, loss_function, negative_data):
    """
    更新鉴别器提示语，优化任务定义和问题示例以提升损失函数值。
    参数：
    - generator_prompt: 生成器提示语。
    - discriminator_prompt_ori: 原始鉴别器提示语。
    - definition: 当前任务的定义。
    - true_instances: 包含真实数据的实例。
    - train_instances: 用于训练的实例数据。
    - loss_function: 当前的损失值。
    - negative_data: 包含负例的额外数据。
    返回：
    - 更新后的鉴别器提示语和新的损失函数值。
    """
    # 替换鉴别器的任务定义
    discriminator_prompt_ori = replace_definition_dis(definition, discriminator_prompt_ori)

    # 提取当前鉴别器任务定义
    definition_dis = discriminator_prompt_ori[0]['content']
    # 定义任务目标，优化任务说明
    task_instruction = 'Diversify the task instruction to be clearer. Keep the task instruction as declarative.'
    # 构造实例化任务
    instance_task = '\n\nTask instruction: ' + definition_dis + '\n\nImproved task instruction: '
    # 调用工具生成新的定义集合
    messages = Construct_Message(task_instruction, instance_task)
    new_definition_set = Promting(messages, 0.4, number=5)
    # 初始化存储新定义损失的列表
    loss_new_definition_set = []
    discriminator_prompt_def_set = []

    # 遍历新生成的任务定义集合
    for new_definition in new_definition_set:
        new_definition = new_definition['message']['content']  # 提取新定义内容
        print(new_definition)  # 输出新定义，便于调试
        print("-------------")
        # 替换鉴别器提示中的任务定义
        discriminator_prompt_def = replace_definition(new_definition, discriminator_prompt_ori[:])
        # 计算新定义的损失值
        loss_current = Loss(generator_prompt[:], discriminator_prompt_def[:], true_instances, train_instances)
        discriminator_prompt_def_set.append(discriminator_prompt_def)
        loss_new_definition_set.append(loss_current)

    # 找到最大损失值
    maximum_loss = max(loss_new_definition_set)
    # 如果优化提升了损失值，则更新鉴别器提示和损失函数
    if maximum_loss > loss_function:
        discriminator_prompt_ori = discriminator_prompt_def_set[loss_new_definition_set.index(maximum_loss)][:]
        loss_function = maximum_loss
        print((loss_new_definition_set.index(maximum_loss), maximum_loss, discriminator_prompt_ori))
    else:
        print("Fail Optimization")  # 如果未提升损失值，打印失败信息
    
    # 优化问题示例部分
    print("===========================")
    print("After Discriminator Instruction")
    
    # 提取示例的最后几行内容
    example = discriminator_prompt_ori[1]['content']
    example = example.splitlines(True)
    example[-4] = example[-4].split('. ')[-1]
    str = ''
    str = str.join(example[-4:])
    str += discriminator_prompt_ori[2]['content']

    # 定义任务目标，优化多选题和答案
    task_instruction = 'Diversify the multiple-choice question and the answer to make it more representative. Keep the main content. Keep the format as multiple-choice question and the answer.'
    # 构造实例化任务
    instance_task = '\n\nMultiple-choice question and the answer: ' + str + '\n\nImproved multiple-choice question and the answer: '
    # 调用工具生成优化后的问题示例集合
    messages = Construct_Message(task_instruction, instance_task)
    new_example_set = Promting(messages, 0.4, 5)
    # 初始化存储优化示例损失的列表
    loss_new_example_set = []
    discriminator_prompt_ex_set = []

    # 遍历新生成的示例集合
    for new_example in new_example_set:
        new_example = new_example['message']['content']  # 提取新示例内容
        print(new_example)  # 输出新示例，便于调试
        print("-------------")
        # 替换鉴别器提示中的示例
        discriminator_prompt_new = replace_discriminator(new_example, discriminator_prompt_ori[:])
        # 计算新示例的损失值
        loss_current = Loss(generator_prompt[:], discriminator_prompt_new[:], true_instances, train_instances)
        discriminator_prompt_ex_set.append(discriminator_prompt_new[:])
        loss_new_example_set.append(loss_current)
    
    # 找到最大损失值
    maximum_loss = max(loss_new_example_set)
    # 如果优化提升了损失值，则更新鉴别器提示和损失函数
    if maximum_loss > loss_function:
        discriminator_prompt_ori = discriminator_prompt_ex_set[loss_new_example_set.index(maximum_loss)][:]
        loss_function = maximum_loss
        print((loss_new_example_set.index(maximum_loss), maximum_loss, discriminator_prompt_ori))
    else:
        print("Fail Optimization")  # 如果未提升损失值，打印失败信息
    
    return discriminator_prompt_ori, loss_function  # 返回更新后的鉴别器提示语和损失值

def OptimizePrompt(generator_prompt, discriminator_prompt, definition, true_instances_full, train_instances_full, negative_data):
    num_shots = 3  # 设置优化循环的次数
    num_sample = 5  # 每次优化中从实例集中采样的样本数量

    generator_prompt_set = []  # 用于存储每次优化后的生成器提示

    for i in range(num_shots):  # 循环进行num_shots次优化
        true_instances = random.sample(true_instances_full, num_sample)  # 随机采样部分真实实例
        train_instances = random.sample(train_instances_full, num_sample)  # 随机采样部分训练实例
        print("Optimize Iteration")  # 打印当前优化迭代的信息
        print(i)
        print(generator_prompt)  # 打印当前的生成器提示
        loss_function = Loss(generator_prompt[:], discriminator_prompt[:], true_instances, train_instances)  # 计算初始损失
        print("Before Optimize")  # 打印优化前的损失值
        print(loss_function)

        # 更新判别器的提示，并返回优化后的判别器提示和当前的损失值
        discriminator_prompt, loss_function = Update_discriminator(generator_prompt[:], discriminator_prompt[:], definition, true_instances, train_instances, loss_function, negative_data)
        print("After Discriminator")  # 打印判别器更新后的损失值
        print(loss_function)

        # 更新生成器的提示，并返回优化后的生成器提示和任务定义
        generator_prompt, definition = Update_generator(generator_prompt[:], discriminator_prompt[:], definition, true_instances, train_instances, loss_function, negative_data)
        
        # 重新计算损失函数
        loss_function = Loss(generator_prompt[:], discriminator_prompt[:], true_instances, train_instances)
        print("After Optimize")  # 打印优化后的损失值
        print(loss_function)
        
        print(generator_prompt)  # 打印优化后的生成器提示

        # 将优化后的生成器提示加入结果集
        generator_prompt_set.append(generator_prompt[:])

    return generator_prompt_set  # 返回所有优化后的生成器提示


def Attempt(generator_prompt, instance):
    prediction = generator(generator_prompt, instance)[0]['message']['content']  # 使用生成器提示对实例进行预测，并提取生成的内容

    # 返回预测结果
    return prediction


def main(argv):
    localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())  # 获取当前时间，用于结果文件命名
    print(localtime)
    openai.api_key = (argv[0])  # 从命令行参数中读取OpenAI的API密钥
    tasks_dir = "/home/bizon/Desktop/Adv-ICL/tasks/"  # 任务文件所在的目录路径
    num_true_instances = 90  # 设置用于测试的真实实例数量
    num_train_instances = 90  # 设置用于训练的实例数量

    # 遍历不同的任务轨道
    for track in ["default"]:
        test_tasks = [l.strip() for l in open(f"/home/bizon/Desktop/Adv-ICL/splits/{track}/test_tasks.txt")]  # 加载测试任务列表

        # 遍历指定范围内的测试任务
        for task in test_tasks[int(argv[1]):int(argv[2])]:
            print(task)
            file = os.path.join(tasks_dir, 'testset_'+ task + ".json")  # 拼接测试任务文件路径
            with open(file) as fin:
                task_data = json.load(fin)  # 加载任务数据

            task_data["Definition"] = [task_data["Definition"][0].strip('\n')]  # 去除定义中的换行符

            # 初始化生成器和判别器的提示
            GENERATOR_PROMPT = generator_prompt(task_data["Definition"],task_data["Positive Examples"])
            DISCRIMINATOR_PROMPT = discriminator_prompt(task_data["Definition"],task_data["Positive Examples"], task_data["Negative Examples"])
            print(DISCRIMINATOR_PROMPT)

            # 从任务数据中随机采样训练和测试实例
            true_instances = random.sample(task_data["Instances"], num_true_instances)
            train_instances = random.sample(task_data["Instances"], num_train_instances)

            # 调用优化函数对生成器提示进行优化
            GENERATOR_PROMPT_set = OptimizePrompt(GENERATOR_PROMPT, DISCRIMINATOR_PROMPT, task_data["Definition"], true_instances, train_instances, task_data["Negative Examples"])

            test_instances = task_data["Instances"]  # 获取测试实例
            GENERATOR_PROMPT = GENERATOR_PROMPT_set[-1]  # 使用最后一次优化后的生成器提示
            print(GENERATOR_PROMPT)

            # 设置结果文件的路径和名称
            name_file = "/home/bizon/Desktop/Adv-ICL/eval/output/" + "[gan-chat-deversify]_" + str(task) + "_" + localtime + ".jsonl"
            print(name_file)

            with open(name_file, "w") as fout:  # 打开结果文件写入模式
                print("in create predictions")

                # 遍历测试实例，生成预测结果并写入文件
                for instance in tqdm(test_instances):
                    prediction = Attempt(GENERATOR_PROMPT,instance["input"])
                    fout.write(json.dumps({
                    "id": instance["id"],  # 保存实例ID
                    "prediction": prediction},  # 保存生成的预测内容
                    ) + "\n")
                fout.close()  # 关闭文件

if __name__ == "__main__":
    main(sys.argv[1:])  # 从命令行参数启动程序

