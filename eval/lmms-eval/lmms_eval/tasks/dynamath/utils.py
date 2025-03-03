import json
import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
import requests

from lmms_eval.tasks.mathvision.eval_utils import find_math_answer, is_equal, is_number
import re
import concurrent.futures
from typing import List, Dict, Any

NUM_SECONDS_TO_SLEEP = 5
API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

JUDGE_RULES = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.
# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{pred}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision, for example: 3.78 can be right if the ground truth is 3.781
- For questions requiring units, both value and unit must be correct

# Strict Output format
[0/1]"""

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAI(api_key=API_KEY)
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    client = AzureOpenAI(azure_endpoint=API_URL, api_version="2023-07-01-preview", api_key=API_KEY)


def get_chat_response(content: str, max_tokens: int, retries: int = 5):
    global MODEL_VERSION
    global client

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the correctness of the answer.",
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": MODEL_VERSION,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            return content
        except requests.exceptions.RequestException as e:
            eval_logger.warning(f"Request failed on attempt {attempt+1}: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP)
            if attempt == retries - 1:
                eval_logger.error(f"Failed to get response after {retries} attempts")
                return 0
        except Exception as e:
            eval_logger.error(f"Error on attempt {attempt+1}: {e}")
            return 0


def dynamath_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]


def dynamath_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    answer_type = doc["answer_type"]

    query_prompt = question
    if lmms_eval_specific_kwargs:
        if answer_type == "multiple choice":  # mcq
            query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['mc_prompt']}"
        else:
            query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['short_answer_prompt']}"
    return query_prompt


# def dynamath_gpt_eval_process_results(doc, results):
#     correct_list = []
#     for pred in results:
#         model_answer = pred.strip()
#         gt_answer = str(doc["ground_truth"])
#         gpt_response = get_chat_response(JUDGE_RULES.format(question=doc["question"], answer=gt_answer, pred=model_answer), 1024)
        
#         # Extract the number from the response, handling potential formats like '[0]' or '0'
#         try:
#             # First try to strip brackets if they exist
#             cleaned_response = gpt_response.strip('[]')
#             if cleaned_response.isdigit():
#                 score = int(cleaned_response)
#             else:
#                 # If that fails, look for any digit in the response
#                 import re
#                 digits = re.findall(r'\d+', gpt_response)
#                 if digits:
#                     score = int(digits[0])
#                 else:
#                     # Default to 0 if no numeric value found
#                     score = 0
                    
#             if score == 1:
#                 correct_list.append(True)
#             else:
#                 correct_list.append(False)
#         except Exception as e:
#             eval_logger.error(f"Error parsing GPT response '{gpt_response}': {e}")
#             correct_list.append(False)  # Default to incorrect on parsing errors

#     return {
#         "dynamath_gpt_eval_score": {
#             "response": results,
#             "scores": correct_list,
#         },
#     }


import re
import concurrent.futures
from typing import List, Dict, Any
from tqdm import tqdm

def dynamath_gpt_eval_process_results(doc, results):
    """
    并行版本的DynaMath评估函数，使用线程池同时处理多个预测结果，并显示进度条。
    
    Args:
        doc: 包含问题和标准答案的文档
        results: 要评估的模型预测结果列表
        
    Returns:
        包含评估分数的字典
    """
    # 获取标准答案
    gt_answer = str(doc["ground_truth"])
    
    # 准备评估参数列表
    eval_params = []
    for pred in results:
        model_answer = pred.strip()
        prompt = JUDGE_RULES.format(
            question=doc["question"], 
            answer=gt_answer, 
            pred=model_answer
        )
        eval_params.append((prompt, 1024))
    
    # 使用线程池并行处理所有预测，并添加进度条
    correct_list = []
    results_list = [None] * len(results)
    
    # 创建进度条
    with tqdm(total=len(results), desc="评估进度", unit="预测") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            # 并行提交所有请求
            future_to_index = {
                executor.submit(get_chat_response, prompt, max_tokens): i 
                for i, (prompt, max_tokens) in enumerate(eval_params)
            }
            
            # 收集结果（按完成顺序更新进度条）
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    gpt_response = future.result()
                    results_list[index] = gpt_response
                    
                    # 解析响应
                    try:
                        # 首先尝试去除可能存在的括号
                        cleaned_response = gpt_response.strip('[]')
                        if cleaned_response.isdigit():
                            score = int(cleaned_response)
                        else:
                            # 如果失败，查找响应中的任何数字
                            digits = re.findall(r'\d+', gpt_response)
                            if digits:
                                score = int(digits[0])
                            else:
                                # 如果没有找到数字，默认为0
                                score = 0
                                
                        # 在获取结果后立即更新正确列表
                        while len(correct_list) <= index:
                            correct_list.append(None)
                        correct_list[index] = (score == 1)
                    except Exception as e:
                        eval_logger.error(f"Error parsing GPT response '{gpt_response}': {e}")
                        while len(correct_list) <= index:
                            correct_list.append(None)
                        correct_list[index] = False  # 解析错误时默认为不正确
                except Exception as e:
                    eval_logger.error(f"Error in API call: {e}")
                    results_list[index] = ""
                    while len(correct_list) <= index:
                        correct_list.append(None)
                    correct_list[index] = False
                
                # 更新进度条
                pbar.update(1)
    
    # 确保结果列表完整且顺序正确
    while None in correct_list:
        index = correct_list.index(None)
        correct_list[index] = False
    
    return {
        "dynamath_gpt_eval_score": {
            "response": results,
            "scores": correct_list,
        },
    }


def dynamath_process_results(doc, results):
    correct_list = []
    for pred in results:
        model_answer = pred.strip()

        gt_answer = str(doc["ground_truth"])
        if len(doc["options"]) > 0:
            gt_answer_value = doc["options"][ord(gt_answer) - ord("A")]
        else:
            gt_answer_value = ""

        for c in "ABCDE":
            if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                model_answer = c
        if is_number(model_answer.split("is ")[-1].rstrip(".")):
            model_answer = model_answer.split("is ")[-1].rstrip(".")
        if "oxed{" not in model_answer:
            for flag in ["the final answer is", "the answer is", "the correct answer is", "the answer should be"]:
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
                flag = flag.replace("the", "The")
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
        elif model_answer.count("oxed{") > 1:
            model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]

        model_answer = (
            find_math_answer(model_answer)
            .replace("(a)", "a")
            .replace("(b)", "b")
            .replace("(c)", "c")
            .replace("(d)", "d")
            .replace("(e)", "e")
            .replace("{a}", "a")
            .replace("{b}", "b")
            .replace("{c}", "c")
            .replace("{d}", "d")
            .replace("{e}", "e")
            .rstrip(".")
            .lstrip(":")
            .strip()
        )
        correct = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
        correct_list.append(correct)
    return {
        "dynamath_standard_eval": {
            # "question": doc["question"],
            # "answer": doc["answer"],
            "response": results,
            # "subject": doc["subject"],
            # "level": doc["level"],
            "scores": correct_list,
        },
    }


def dynamath_aggregate_results_eval(results):
    total = len(results)
    correct = sum(1 for idx, result in enumerate(results) if results[idx]["scores"][0])
    accuracy = round(correct / total * 100, 2)
    return accuracy
