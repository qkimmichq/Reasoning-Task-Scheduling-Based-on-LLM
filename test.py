from openai import OpenAI

def get_task_instruction_math(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following math question. '
            'Please give the final answer without any additional explanation or clarification.\n\n'
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following math question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt
question = "How many positive whole-number divisors does 196 have?"
user_prompt = get_task_instruction_math(question, model_name='qwq')
client = OpenAI(
    base_url="http://127.0.0.1:6006/v1",  # ← 你的 OLLAMA_HOST
    api_key="ollama",                     # 必填但不会被用到
)

# Chat Completions（推荐）
messages = [{"role": "user", "content": user_prompt}]
chat = client.chat.completions.create(
    model="qwen3:4b",  # 必须与 `ollama list` 一致
    messages=messages,
    max_tokens=2048,
    temperature = 0.9
)
# print(chat)
print(chat.choices[0].message.content)