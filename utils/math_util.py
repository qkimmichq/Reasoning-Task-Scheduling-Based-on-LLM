import json
import re
def _fix_fracs(string):
    """修复LaTeX分数表达式的格式问题,确保\frac后面跟着正确的括号格式"""
    # 按\frac分割字符串
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            # 处理\frac后直接跟着{的情况
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                # 处理\frac后跟着两个字符但不是{的情况
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    # 处理\frac后跟着字符和{的情况
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    """移除字符串右侧的单位文本(以'\text{ '开头的部分)
    
    参数:
        string: 可能包含单位文本的字符串
        
    返回:
        移除单位文本后的字符串，如果不存在单位文本则返回原字符串
    """
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    """标准化数学表达式字符串的实用函数"""
    # 处理换行符
    string = string.replace("\n", "")
    
    # 处理反斜杠转义字符
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    
    # 统一分数表示形式
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    
    # 移除左右定界符
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    
    # 处理角度符号
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    
    # 处理特殊符号
    string = string.replace("\\$", "")
    
    # 移除右侧单位和百分号
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    
    # 标准化小数点表示
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    
    # 简化简单变量赋值表达式
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    
    # 修复数学表达式格式
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    
    # 特殊处理0.5的表示
    if string == "0.5":
        string = "\\frac{1}{2}"
    
    # 处理分数斜杠表示
    string = _fix_a_slash_b(string)
    
    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2
    
def seconds_to_hms(seconds):
    """将秒数转换为小时、分钟、秒的元组格式"""
    hours = seconds // 3600  # 计算总小时数
    seconds %= 3600          # 计算剩余秒数
    minutes = seconds // 60   # 计算总分钟数
    seconds %= 60            # 计算剩余秒数
    return hours, minutes, seconds

def CountCost(token_usage):
    cost_per_1000_tokens = {
        "gpt-4-turbo": {"prompt": 10, "completion": 30},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        "gpt-4": {"prompt": 30, "completion": 60},
        "gpt-4o": {"prompt": 2.5, "completion": 10}
    }
    
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
        
def write_json_listoneline(file_path, data):
    try:
        # 自定义递归函数，用于处理 list 和其他类型的数据
        def custom_json_encoder(obj, indent=0):
            # 定义缩进
            indent_str = ' ' * indent
            
            if isinstance(obj, dict):
                # 处理 dict 类型
                json_str = '{\n'
                for i, (key, value) in enumerate(obj.items()):
                    if i > 0:
                        json_str += ',\n'
                    json_str += f'{indent_str}  "{key}": {custom_json_encoder(value, indent + 2)}'
                json_str += f'\n{indent_str}}}'
                return json_str

            elif isinstance(obj, list):
                # 处理 list 类型，不换行
                return json.dumps(obj, separators=(',', ':'))

            else:
                # 处理其他类型
                return json.dumps(obj, ensure_ascii=False)

        # 打开文件并写入数据
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(custom_json_encoder(data))
        
        print(f"数据成功写入 {file_path}")
    except Exception as e:
        print(f"发生错误：{e}")

def quantile(lst, alpha):
    # 确保 alpha 在 0 和 1 之间
    if not 0 <= alpha <= 1:
        raise ValueError("alpha should be between 0 and 1")

    # 排序列表
    sorted_lst = sorted(lst)

    # 计算分位数的索引
    index = int(alpha * len(sorted_lst))

    # 如果索引等于列表长度，返回最后一个元素
    if index == len(sorted_lst):
        index -= 1

    # 返回分位数的值
    return sorted_lst[index]

def reverseDict(original_dict):
    reversed_dict = {}

    # 遍历原始字典的键值对
    for key, value in original_dict.items():
        if value in reversed_dict:
            reversed_dict[value].append(key)
        else:
            reversed_dict[value] = [key]
    return reversed_dict

def extract_last_model_name(input_string):
    # 使用正则表达式提取最后一个 '/' 后面的内容
    match = re.search(r'([^/]+)$', input_string)
    
    if match:
        return match.group(1)  # 返回匹配的部分，即最后一个 '/' 后的内容
    else:
        return None  # 如果没有匹配，返回 None

def load_jsonl(path):
    """
    读取 JSONL -> 返回 dict: {question_id(int): sample(dict)}
    每条 sample 结构与原始行相同。
    """
    problems = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            problems[i] = obj
    return problems

_NUMBER = re.compile(r"""
    (?:\#{4}\s*)?          # 可选 '####'
    \$?\s*                 # 可选货币符
    ([-+]?\d[\d,]*         # 整数部分（可含千分位）
      (?:/\d+)?            # 可选分数，例如 3/5
      (?:\.\d+)?           # 可选小数
    )
""", re.VERBOSE)

def extract_final_numeric(answer_text: str) -> str:
    """
    规则优先级：
    1) 取最后一个 '#### <number>' 后的数字
    2) \boxed{...}
    3) 文中出现的最后一个“像数”的片段
    返回去掉 $ 和逗号的纯字符串（例如 '18'、'3/5'、'12.5'）
    """
    # 1) 明确优先拿 #### 后面的
    m = re.search(r"####\s*\$?\s*([-\+]?\d[\d,]*(?:/\d+)?(?:\.\d+)?)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    # 2) \boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", answer_text)
    if m:
        return m.group(1).strip().replace(",", "").lstrip("$")
    # 3) 兜底：拿全文最后一个数字片段
    cands = _NUMBER.findall(answer_text)
    return cands[-1].replace(",", "") if cands else ""
