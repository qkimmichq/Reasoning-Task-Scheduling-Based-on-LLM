'''
根据adapter的任务分配策略进行推理,考虑所有任务都由同一个模型在单机上执行推理
'''
# -*- coding: utf-8 -*-
# 任务分解
import traceback
import os
import json
import os
import sys
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 将项目根目录添加到 sys.path 中
sys.path.append(project_root)
import time
import argparse
from datetime import datetime
from tqdm import tqdm
from paper2.utils.infer_util import *
from paper2.utils.math_util import *
from paper2.utils.decomposition_util import *
from paper2.evaluate import *


# client定义需要满足如下调用方式: client.chat.completions.create(model,messages = messages), 详见askLLM函数
openaiClient = setOpenAi(keyid = 0)
llamaClient = setLocal()
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "after_decomposition_infer"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Task Decomposition for various datasets.")
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=[
            'gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'all_math',
            'triviaqa', 'hotpotqa', '2wiki', 'musique',
            'bamboogle', 'medmcqa', 'pubhealth', 'test'
        ],
        help="Name of the dataset to use (without .json)."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    
    start_time = time.time()
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    # 初始化记录token消耗的路径
    tokens_path = f'Tokens/{aftername}/{dataset_name}/token_usage_{formatted_now}.json'  # 记录token消耗的文件
    # 使用 os.makedirs() 确保目录存在，如果不存在则自动创建
    os.makedirs(os.path.dirname(tokens_path), exist_ok=True)
            
    logger, filename = setup_logger(aftername)
    
    with open('../MATH_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path
        
    # 读取原始数据集
    file_path = f'../task_datasets/{dataset_name}/test.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)
        
    # 读取上一步任务分解之后的数据集
    f = open(f'./results/task_decomposition_result/{dataset_name}/TD_task_decomposition_math500_LLM-Research_Meta-Llama-3-8B-Instruct_2025-08-11-19-38-36.json', 'r')
    content = f.read()
    middleRes = json.loads(content) 

    success_Q = 0
    false_Q = 0
    error_Q = 0
    N = len(middleRes)
    question_ids = list(range(N))
    inferRes = {} # 保存推理结果
    MAX_TRY = 5  # 错误尝试上限
    for question_id in tqdm(question_ids):
        
        question = problems[question_id]['Question']
        type = problems[question_id]['subject']
        gold_answer = problems[question_id]['answer']
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('problem content:\n')
        logger.info(question)

        attempts = 0
        success = False
        # 允许模型进行多次推理尝试
        while attempts < MAX_TRY and not success:
            try:
                steps, steps_dict, depths, int_edges = middleRes[str(question_id)]['steps'], middleRes[str(question_id)]['steps_dict'], middleRes[str(question_id)]['depths'], middleRes[str(question_id)]['int_edges']
                depths = {int(k): v for k, v in depths.items()}
                heights = list(depths.keys())
                MAXHeight = max(heights)+1
                # print(f"MAXHeight: {MAXHeight}")
                answerDict = {} 
                progress_bar = tqdm(total=len(steps))
                for i in range(MAXHeight):
                    subtasks = depths[i]
                    for subtaskid in subtasks:                
                        number = re.findall(r'\d+', subtaskid)
                        number = int(number[0]) if number else None
                        subtask = steps_dict[str(number)]
                        answer_MODEL = config['subtask_MODEL']
                        # 交待解决任务
                        sys_q = f"""There is a math_problem. I need you to solve it and give an answer.
Here is the problem:\n{question} 

I have broken this math problem down into several smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
Please solve the problem and respond according to mathematical logic.
        """  # 系统任务信息
                        
                        if len(answerDict)>0:
                            answersSoFar = f"""\nSo far, the answers to the resolved sub-problems are as follows: The format is Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx."""
                            for key, value in answerDict.items():
                                answersSoFar += f"""\nSub-problem-Id: {key}; Sub-problem: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
                            
                            predecessors = search_Predecessors(int_edges, number)
                            intersection = set(answerDict.keys()).intersection(set(predecessors))
                            count = len(intersection)
                            if count>0:
                                answersSoFar += f"""\nAmong them, sub-problems {predecessors} are directly related to this sub-problem, so please pay special attention to them."""
                        
                        
                        subask = f"""\nThe sub-problem to solve now is xxx: {subtask}
Based on the information above, please provide a concise and clear answer"""

                        if len(answerDict)>0:
                            query = answersSoFar+subask
                        else:
                            query = subask

                        Q = [{'role':'system', 'content':sys_q},
                            {'role':'user', 'content':query},]
                        
                        result = askLLM(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=300)                        
                        answerDict[number] = {'subtask':subtask, 'answer':result}
                        progress_bar.update(1)

                progress_bar.close()
                # 已经问完了所有的subtask,最后问一次得到最终的答案
                Q.append({'role':'assistant', 'content':result})
                Q.append({'role':'user', 'content':"""Now that all the sub-problems have been solved, you should provide your final answer in the format \\boxed{YOUR_ANSWER}.
Please give the final answer without any additional explanation or clarification."""})
                # finalResult = askChatGPT(Q, model=GPT_MODEL, temperature=1)
                finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
                # print('图上推理 done')
                logger.info("\n%s", "="*80)
                logger.info("🟢 FINAL RESULT")
                logger.info("%s", finalResult)
                logger.info("%s\n", "="*80)
                  
                inferRes[str(question_id)] = {
                        "question": question,
                        "gold_answer": gold_answer,
                        "infer_answer": finalResult,
                        "subtask_answers": answerDict,
                    }
                success = True
            except (KeyError, ValueError) as e:
                attempts += 1
                tb = traceback.format_exc()
                logger.error(f"[attempt {attempts}] taskid={question_id} 业务校验失败: {e}\n{tb}")
                last_err = e
        
        if attempts == MAX_TRY:
            error_Q += 1
            logger.info(f'run error {MAX_TRY}+')

    # 构造保存路径,先保存推理结果，保存到results/
    save_dir = f'./results/{aftername}/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    model_name = re.sub(r'[<>:"/\\|?*]', '_', config['decompose_MODEL'])
    infer_save_path = os.path.join(save_dir, f'{aftername}_{formatted_now}_{model_name}.json')
    with open(infer_save_path, 'w', encoding='utf-8') as f:
        json.dump(inferRes, f, indent=2, ensure_ascii=False)
    logger.info(f"Inference results saved to: {infer_save_path}")
    
    # 进行数值评估/统计F1 em acc math_equal 保存到outputs/
    output_dir = f'./output/{aftername}/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    input_list = [item['question'] for item in inferRes.values()]
    output_list = [item['infer_answer'] for item in inferRes.values()]
    metrics,success_Q, false_Q = run_evaluation(inferRes, input_list, output_list, dataset_name, output_dir=output_dir, total_time=time.time() - start_time,  apply_backoff=False, answer_model=answer_MODEL, aftername=aftername)
    print(metrics)
    
    # 计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    logger.info(f"100 solving 运行耗时: {hours}h, {minutes}min, {seconds}s")
    
    # 计算执行准确率
    logger.info(f'\n{tokens_path}')
    logger.info(f'Correct_Q: {success_Q}')
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

    