'''
根据adapter的任务分配策略进行协同推理
'''
# -*- coding: utf-8 -*-
# 任务分解
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
# 设置客户端
openaiClient = setOpenAi(keyid=0)
llamaClient = setLocal()
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "task_decomposition_result"


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
    logger, _ = setup_logger(aftername)
    
    # 构造 token 保存路径
    token_dir = f'Tokens/{aftername}/{dataset_name}'
    os.makedirs(token_dir, exist_ok=True)
    tokens_path = os.path.join(token_dir, f'token_usage_{formatted_now}.json')
    if not os.path.exists(tokens_path):
        with open(tokens_path, 'w') as f:
            json.dump({}, f)

    # 读取任务分解的模型配置信息
    with open('./MATH_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path

    # 构造数据集路径和输出路径
    file_path = f'../task_datasets/{dataset_name}/test.json'
    # 保存路径结构
    save_dir = f"./results/{aftername}/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）
    model_name = re.sub(r'[<>:"/\\|?*]', '_', config['decompose_MODEL'])
    save_path = os.path.join(save_dir, f"task_decomposition_{formatted_now}_{model_name}.json")

    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)

    success_Q = 0
    error_Q = 0
    N = len(problems)
    question_ids = list(range(N))

    MAX_TRY = 5
    step1Res = {}

    for question_id in tqdm(question_ids):
        step1Res[question_id] = {}
        question = problems[question_id]['Question']
        type = problems[question_id]['subject']
        gold_answer = problems[question_id]['answer']

        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('problem content:')
        logger.info(question)
        logger.info('\n')

        attempts = 0
        success = False
        while attempts < MAX_TRY and not success:
            try:
                decompose_steps = decompose_sql(clients, question, type, config)
                print(f"decompose_steps:{decompose_steps}/n")
                # print("执行成功------------------------------------------------")
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                # print(f"执行成功convert_steps_to_format******************************{steps}")
                formatted_steps = '; '.join([f'step{i+1}: {step}' for i, step in enumerate(steps)])
                # print(f"formatted_steps:{formatted_steps}/n")

                relations_test = construct_dependencies_without_traversal(clients, question, steps, config)
                # print(f"执行成功construct_dependencies_without_traversalxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx{relations_test}")
                G1 = create_dag_from_string(relations_test)
                # print(f"执行成功create_dag_from_string+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                reduced_dependencies = list(G1.edges())
                edges = []
                for item in reduced_dependencies:
                    src = item[0][:item[0].find('[')].strip()
                    tgt = item[1][:item[1].find('[')].strip()
                    edges.append((src, tgt))
                int_edges = [(int(e[0].split()[1]), int(e[1].split()[1])) for e in edges]

                node_depths = calculate_node_depths(edges)
                depths = reverseDict(node_depths)

                step1Res[question_id]['steps'] = steps
                step1Res[question_id]['steps_dict'] = steps_dict
                step1Res[question_id]['depths'] = depths
                step1Res[question_id]['int_edges'] = int_edges
                step1Res[question_id]['problemText'] = question
                step1Res[question_id]['allSubtask'] = formatted_steps
                step1Res[question_id]['nowSubtask'] = steps
                success_Q += 1
                success = True

            except Exception as e:
                attempts += 1
                logger.info(f"error: {attempts}; taskid: {question_id}; exception: {str(e)}")

        if attempts == MAX_TRY:
            error_Q += 1
            logger.info(f'run error {MAX_TRY}+')

    write_json_listoneline(save_path, step1Res)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(elapsed_time)

    logger.info(f"{dataset_name} solving 运行耗时: {hours}h, {minutes}min, {seconds}s")
    logger.info(f'{tokens_path}')
    logger.info(f'Correct_Q: {success_Q}')
    logger.info(f'Error_Q: {error_Q}')
    # 统计模型输出的token数量和成本
    # with open(tokens_path, 'r') as f:
    #     token_usage = json.load(f)
    #     total_tokens, total_cost = CountCost(token_usage)
    #     logger.info(f"Total Tokens: {total_tokens}")
    #     logger.info(f"Total Cost: ${total_cost:.2f}")


if __name__ == '__main__':
    main()
