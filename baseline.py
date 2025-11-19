# -*- coding: utf-8 -*-
# 任务分解
import json
import os
import sys
import time
import argparse
from datetime import datetime
from tqdm import tqdm
import re
import openai
from openai import OpenAI
import regex 
sys.path.append('../')
import logging
from paper2.evaluate import run_evaluation
from utils.infer_util import *
from utils.math_util import *
from prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)
# 设置客户端
openaiClient = setOpenAi(keyid=0)
llamaClient = setLocal()
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = 'direct_infer'
def parse_args():
    parser = argparse.ArgumentParser(description="Run Task Decomposition for various datasets.")
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=[
            'gsm8k', 'gpqa', 'csqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'all_math', 'champ',
            'triviaqa', 'hotpotqa', '2wiki', 'musique',
            'bamboogle', 'medmcqa', 'pubhealth', 'test'
        ],
        help="Name of the dataset to use (without .json)."
    )
    
    parser.add_argument(
        '--subset_num', 
        type=int, 
        default=-1, 
        help="处理的样本数量，默认选择所有."
    )
    
    # parser.add_argument(
    #     '--model_path', 
    #     type=str, 
    #     required=True,
    #     help="本地模型路径."
    # )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.7, 
        help="Sampling temperature."
    )
    
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.8, 
        help="Top-p sampling parameter."
    )
    
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=20, 
        help="Top-k sampling parameter."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    dataset_name = args.dataset_name

    start_time = time.time()
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    # 日志信息，记录执行过程
    logger, _ = setup_logger(aftername+formatted_now)
    # # 初始化记录token消耗的路径
    tokens_path = f'Tokens/{aftername}/{dataset_name}/token_usage_{formatted_now}.json'  # 记录token消耗的文件
    # 使用 os.makedirs() 确保目录存在，如果不存在则自动创建
    os.makedirs(os.path.dirname(tokens_path), exist_ok=True)

    # 数据集路径
    if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode', 'champ']:
        data_path = f'./task_datasets/{dataset_name}/test.json'
            # 构造数据集路径和输出路径
        with open(data_path, 'r', encoding='utf-8') as file:
            problems = json.load(file)
    elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        data_path = f'./task_datasets/QA_Datasets/{dataset_name}.json'
            # 构造数据集路径和输出路径
        with open(data_path, 'r', encoding='utf-8') as file:
            problems = json.load(file)
    elif dataset_name in ['csqa', 'gsm8k']:
        data_path = f'./task_datasets/{dataset_name}/test.jsonl'
        problems = load_jsonl(data_path)
    else:

        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    #加载模型配置信息
    with open('MATH_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path
    inferRes = {}
    N = len(problems)
    question_ids = list(range(N))
    error_Q = 0
    MAX_TRY = 5
    for question_id in tqdm(question_ids):
        inferRes[question_id] = {}
        answer_MODEL = config["subtask_MODEL"]
         # 构造融合后的提示词
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            question = problems[question_id]['Question']
            # type = problems[question_id]['subject']
            gold_answer = problems[question_id]['answer']

            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}')
            logger.info('problem content:')
            logger.info(question)
            if 'qwq' in answer_MODEL.lower() or 'deepseek' in answer_MODEL.lower() or 'sky-t1' in answer_MODEL.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            question = problems[question_id]['Question']
            # type = problems[question_id]['subject']
            gold_answer = problems[question_id]['answer']

            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}')
            logger.info('problem content:')
            logger.info(question)
            if 'qwq' in answer_MODEL.lower() or 'deepseek' in answer_MODEL.lower() or 'sky-t1' in answer_MODEL.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)
        
        elif dataset_name in ['math']:
            question = problems[question_id]['problem']
            # type = problems[question_id]['subject']
            gold_answer = problems[question_id]['answer']

            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}')
            logger.info('problem content:')
            logger.info(question)
            if 'qwq' in answer_MODEL.lower() or 'deepseek' in answer_MODEL.lower() or 'sky-t1' in answer_MODEL.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)
        
        elif dataset_name in ['champ']:
            question = problems[question_id]['problem_text']
            # type = problems[question_id]['subject']
            gold_answer = problems[question_id]['problem_answer']

            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}')
            logger.info('problem content:')
            logger.info(question)
            if 'qwq' in answer_MODEL.lower() or 'deepseek' in answer_MODEL.lower() or 'sky-t1' in answer_MODEL.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name in ['gpqa']:
            question = problems[question_id]['Question']
            # type = problems[question_id]['subject']
            gold_answer = problems[question_id]['Correct Choice']
            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}')
            logger.info('problem content:')
            logger.info(question)
            if 'qwq' in answer_MODEL.lower() or 'deepseek' in answer_MODEL.lower() or 'sky-t1' in answer_MODEL.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in answer_MODEL.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)
        
        elif dataset_name in ['gsm8k', 'math_word', 'math_gsm', 'word_math']:
            item = problems[question_id]
            question = item['question']
            gold_answer_raw = item['answer']
            gold_answer = extract_final_numeric(gold_answer_raw)  # ← 只取最终数值

            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}')
            logger.info('problem content:')
            logger.info(question)
            logger.info(f'gold answer: {gold_answer}')  # 可选打印

            # 按模型风格选提示（与你给的 math 分支一致）
            if 'qwq' in answer_MODEL.lower() or 'deepseek' in answer_MODEL.lower() or 'sky-t1' in answer_MODEL.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')   # 只要最终答案
            else:
                user_prompt = get_task_instruction_math(question)                     # 默认（可含思考）

        elif dataset_name in ['csqa', 'commonsense', 'commonsenseqa']:
            item = problems[question_id]
            # 题干 & 选项
            stem = item['question']['stem']
            choices = item['question']['choices']  # list of {label, text}
            options_str = "\n".join(f"{c['label']}. {c['text']}" for c in choices)

            # 供 prompt 使用的 question 文本（和 math 分支保持风格一致）
            question = f"{stem}\n\nOptions:\n{options_str}"

            # 标准答案（字母）与可选的文本版（便于评测/打印）
            gold_answer = item['answerKey']  # 'A' ~ 'E'
            label2text = {c['label']: c['text'] for c in choices}
            gold_answer_text = label2text.get(gold_answer, "")

            # 日志
            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}  (sample id: {item.get("id","")})')
            logger.info('problem content:')
            logger.info(question)
            logger.info(f'gold answer: {gold_answer} ({gold_answer_text})')

            # 根据模型风格选择提示词（与你给的 math 分支一致的逻辑）
            if 'qwq' in answer_MODEL.lower() or 'deepseek' in answer_MODEL.lower() or 'sky-t1' in answer_MODEL.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')     # 只要最终答案
            elif 'llama' in answer_MODEL.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')   # 要求 step-by-step 但最终 \boxed{...}
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif dataset_name == 'livecode':
            question_title = problems[question_id].get('question_title', '')
            if 'qwq' in answer_MODEL.lower() or 'deepseek' in answer_MODEL.lower() or 'sky-t1' in answer_MODEL.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched
        temperature = 0.3
        max_tokens = 2000
        last_exception = None
        output_list = []
        for attempt in range(MAX_TRY):
            try:
                messages = [{"role": "user", "content": user_prompt}]
                infer_answer = askLLM(clients, messages, tokens_path=tokens_path, model=answer_MODEL, temperature=temperature, max_tokens=max_tokens) 
                output_list.append(infer_answer)
                inferRes[question_id]['question'] = question
                inferRes[question_id]['infer_answer'] =  infer_answer
                inferRes[question_id]['gold_answer'] = gold_answer
                # inferRes[question_id]['subject'] = type
                break  # 成功，跳出重试循环

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for Q{question_id}: {e}")
                time.sleep(1)

        else:
            # 全部尝试失败
            error_Q += 1
            logger.error(f"All attempts failed for question {question_id}: {last_exception}")
            inferRes[question_id]['question'] = question
            inferRes[question_id]['infer_answer'] = None
            inferRes[question_id]['gold_answer'] = gold_answer
            # inferRes[question_id]['subject'] = type
     # 合并输出
    full_output = {
        "results": inferRes
    }
    # 构造保存路径,先保存推理结果，保存到results/
    output_dir = f'./results/{aftername}/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    model_name = re.sub(r'[<>:"/\\|?*]', '_', answer_MODEL)
    result_path = os.path.join(output_dir, f"{aftername}_{formatted_now}_{model_name}.json")
    with open(result_path, 'w', encoding='utf-8') as f_out:
        json.dump(full_output, f_out, ensure_ascii=False, indent=2)
    logger.info(f"Full question-answer pairs saved to: {result_path}")
    
    # 将推理结果转换为评估函数所需的格式并进行评估,# 进行数值评估/统计F1 em acc math_equal 保存到output/
    input_list = [item['question'] for item in inferRes.values()]
    output_list = [item['infer_answer'] for item in inferRes.values()]
    output_dir = f'./output/{aftername}/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    # 进行评估
    metrics, success_Q, false_Q = run_evaluation(inferRes, input_list, output_list, dataset_name, output_dir=output_dir, total_time=time.time() - start_time,  apply_backoff=False, answer_model=answer_MODEL, aftername=aftername)
    print(metrics)
        # 计算执行准确率
    logger.info(f'\n{tokens_path}')
    logger.info(f'Success_Q: {success_Q}')
    logger.info(f'False_Q: {false_Q}')
    logger.info(f'Error_Q: {error_Q}\n')
    
    # # 计算推理token数量和总推理成本cost
    # with open(tokens_path, 'r') as f:
    #     token_usage = json.load(f)
    #     # logger.info(json.dumps(token_usage, indent=4))
    #     total_tokens, total_cost = CountCost(token_usage)
    #     # 打印结果
    #     logger.info(f"Total Tokens: {total_tokens}")
    #     logger.info(f"Total Cost: ${total_cost:.2f}")
    
if __name__ == '__main__':
    main()
