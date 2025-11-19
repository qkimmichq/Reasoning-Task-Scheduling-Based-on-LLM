from openai import OpenAI
import logging
import sys
import math
import json
def setup_logger(tailName=""):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("Logs/test_"+tailName+".log")

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger, "test_"+tailName+".log"

def setOpenAi(keyid = 0):
    # set your openai key here.
    if keyid == 0:
        # base_url="https://api.siliconflow.cn"
        # api_key = "sk-jltwxwxwtyknufxsmprabvoqunyfcjyvwzwgzwxxaqubrchw"
        base_url="https://openrouter.ai/api/v1"
        api_key = "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"
    client = OpenAI(base_url = base_url, api_key=api_key)
    return client

def setLocal():
    client = OpenAI(
        api_key="ollama",
        base_url="http://127.0.0.1:6006/v1",
    )
    return client

def update_token_usage(model_name, prompt_tokens, completion_tokens, file_path='token_usage.json'):
    # 读取现有数据
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 如果模型不存在，则初始化模型的数据结构
    if model_name not in data:
        data[model_name] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
    
    # 更新模型的token数量
    data[model_name]['prompt_tokens'] += prompt_tokens
    data[model_name]['completion_tokens'] += completion_tokens
    data[model_name]['total_tokens'] += (prompt_tokens + completion_tokens)
    
    # 将更新后的数据写回文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def askLLM(clients, messages, tokens_path, model="gpt-3.5-turbo", temperature = 1, max_tokens=4000):
    # 需要包括GPT系列以及LLaMA系列的模型调用,分开写已备调用接口略有区别
    
    if model in ['deepseek-ai/DeepSeek-R1', 'Qwen/Qwen3-8B', 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o-mini','deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'gpt-4o', 'deepseek/deepseek-r1:free']: # GPT系列模型调用           
        client = clients['gpt']  # gpt系列共用一个client
        response = client.chat.completions.create(
                model = model,
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
            )
        # update_token_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)
        answer = response.choices[0].message.content
        
    elif model in ['llama3-70b', 'LLM-Research/Meta-Llama-3-8B-Instruct', 'qwen3:4b', 'deepseek-r1:7b', 'llama3:8b']:
        client = clients['llama']  # llama系列共用一个client
        response = client.chat.completions.create(
                model = model, 
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
            )
        answer = response.choices[0].message.content
        # print(f"response:{response}")
        # print(f"answer:{answer}")
    else:
        print('MODEL error')
        print(model)
        sys.exit(0)

    return answer.strip()

def askLLM_withprob(clients, messages, tokens_path, model="gpt-3.5-turbo", temperature = 1, max_tokens=200):
    # 需要包括GPT系列以及LLaMA系列的模型调用,调用接口略有区别
    probs = {}
    if model in ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o-mini']: # GPT系列模型调用           
        client = clients['gpt']  # gpt系列共用一个client
        response = client.chat.completions.create(
                model = model,
                messages = 'gpt-4o',
                temperature = temperature,
                max_tokens = max_tokens,
                logprobs = True,
            )
        # print(response.usage
        # add token 需要更加细致.
        # addtoken(response.usage.total_tokens)
        model ='gpt-4o'
        update_token_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)
        answer = response.choices[0].message.content
        for item in response.choices[0].logprobs.content:
            # 在这一步就把logprob用e指数返回成prob
            probs[item.token] = math.exp(item.logprob)
        
    elif model in ['llama3-70b', 'llama3-8b']:
        client = clients['gpt']  # 这里需要改成llama系列的prompts  # TODO 还没拿到LLaMA的key, 所以先拿gpt-3.5充当.
        response = client.chat.completions.create(
                model = "gpt-3.5-turbo",  # TODO 还没拿到LLaMA的key, 所以先拿gpt-3.5充当.
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
                logprobs = True,
            )
        # addtoken(response.usage.total_tokens)
        update_token_usage("gpt-3.5-turbo", response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)
        answer = response.choices[0].message.content
        for item in response.choices[0].logprobs.content:
            probs[item.token] = math.exp(item.logprob)
    else:
        print('MODEL error')
        print(model)
        sys.exit(0)

    return answer.strip(), probs