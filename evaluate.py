import re
import json
import numpy as np
from colorama import Fore
from collections import Counter
import string
from tqdm import tqdm  # 导入tqdm
import os, time
from collections import defaultdict
from paper2.utils.math_util import is_equiv
from datetime import datetime
from paper2.utils.math_util import extract_last_model_name
from paper2.utils.infer_util import *
openaiClient = setOpenAi(keyid = 0)
llamaClient = setLocal()
clients = {'gpt': openaiClient, 'llama': llamaClient}
with open('./MATH_config.json', 'r') as f:
    config = json.load(f)
    
def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "\n**Final Information**"
        pattern_step = "\n**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
    return extracted_text

def evaluate_predictions(output, labeled_answer, mode='gen'):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
    pred_answer = extract_answer(output, mode=mode)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    else:
        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    # print(em, acc, f1, normalized_pred_answer, '|', normalized_ground_truth)
    return final_metric, pred_answer

def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, apply_backoff=False, answer_model=None, aftername=None):
    # Existing evaluation for other datasets
    avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
    num_valid_answer = 0
    if isinstance(filtered_data, dict):
        filtered_data = list(filtered_data.values())
    # If the dataset is GPQA, track metrics per domain
    domain_metrics = {}
    success_Q = 0
    false_Q = 0
    for item, input_prompt, result in tqdm(zip(filtered_data, input_list, output_list), total=len(filtered_data),  desc=f"{Fore.GREEN}Evaluating{Fore.RESET}", unit="item"):
        if dataset_name in ['gpqa', 'medmcqa', 'csqa', 'gsm8k']:
            labeled_answer = item["gold_answer"]
            # labeled_choice_answer = item["Correct Answer"]
            mode = 'choose'
        elif dataset_name in ['math500', 'aime', 'amc', 'champ']:
            labeled_answer = item["gold_answer"]
            mode = 'gen'
        elif dataset_name in ['all_math']:
            labeled_answer = item["solution"]
            mode = 'gen'
        elif dataset_name in ['drop', 'nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            labeled_answer = item["gold_answer"]
            mode = 'qa'
        elif dataset_name in ['pubhealth']:
            labeled_answer = item["answer"]
            mode = 'choose'
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        metric, pred_answer = evaluate_predictions(output=result, labeled_answer=labeled_answer, mode=mode)
        item['standardized_infer_answer'] = pred_answer
        item['Metrics'] = metric
        # 为了排除一些符号对数学结果的影响，采用大模型来判断执行结果准确性
        judgeAnswer = {'role':'user', 'content':f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct.
        Problem: {item['question']}

        Standard answer: {labeled_answer}

        Answer: {pred_answer}

        If the student's answer is correct, just output True; otherwise, just output False.
        No explanation is required.
        """}
        Q_judge = [judgeAnswer]
        # ifcorrect = askChatGPT(Q_judge, model=GPT_MODEL, temperature=1)  # 要么是True, 要么是False
        ifcorrect = askLLM(clients, Q_judge, tokens_path=None, model=config['judgeCorrect_MODEL'], temperature=1, max_tokens=300)
        print(f"ifcorrect:{ifcorrect}")
        if item['Metrics']['is_valid_answer']==False:
            false_Q += 1
            item['is_correct'] ='false' 
        elif  'True' in ifcorrect:
            success_Q += 1
            item['is_correct'] ='true' 
        else:
            false_Q += 1
            item['is_correct'] ='false' 
        

        # Determine the validity of the predicted answer
        my_method_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

        avg_em.append(metric['em'])
        avg_acc.append(metric['acc'])
        avg_f1.append(metric['f1'])
        avg_math.append(metric['math_equal'])

        if my_method_valid:
            num_valid_answer += 1

        # If the dataset is GPQA, attempt to track metrics per domain
        if dataset_name == 'gpqa':
            domain = item.get("High-level domain", "Unknown")
            if domain not in domain_metrics:
                domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0}
            domain_metrics[domain]['total_num'] += 1
            domain_metrics[domain]['em'].append(metric['em'])
            domain_metrics[domain]['acc'].append(metric['acc'])
            domain_metrics[domain]['f1'].append(metric['f1'])
            domain_metrics[domain]['math_equal'].append(metric['math_equal'])
            if my_method_valid:
                domain_metrics[domain]['num_valid_answer'] += 1

    t = time.localtime()

    # 计算平均的metrics
    overall_results = {
        'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
        'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
        'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
        'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
        'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        'is_correct': f'{success_Q} of {len(input_list)}',
    }

    # If the dataset is GPQA, output average metrics per domain
    domain_avg_metrics = {}
    if dataset_name == 'gpqa':
        for dm, m in domain_metrics.items():
            domain_avg_metrics[dm] = {
                'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}'
            }

    # 保存总体和分domain的指标
    final_metrics = {'overall': overall_results}
    if dataset_name == 'gpqa':
        final_metrics['per_domain'] = domain_avg_metrics
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    answer_model = extract_last_model_name(answer_model)
    result_json_name = f'{aftername}_{formatted_now}_{answer_model}.json'
    metrics_json_name = f'{aftername}_{formatted_now}_{answer_model}_metrics.json'
    if apply_backoff:
        result_json_name = output_dir
        metrics_json_name = output_dir.replace('.json', '.metrics.backoff.json')
    # print(f"filtered_data: {filtered_data}")
    # print(f"final_metrics: {final_metrics}")
    # Save prediction results and metrics
    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)
    return final_metrics, success_Q, false_Q

# 有时候推理的结果是对的，但是没法正确读取结果  //box的形式