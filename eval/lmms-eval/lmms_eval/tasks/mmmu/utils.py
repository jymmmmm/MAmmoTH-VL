import ast
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

MULTI_CHOICE_PROMPT = "First give the reasoning and then write your final answer in the format 'Answer: X' where X is the option's letter from the given choices. If the answer is not in the choices, please give the closest option."
OPEN_ENDED_PROMPT = "First give the reasoning and then write your final answer in the format 'Answer: X' where X is a single word or phrase that answers the question."
# MULTI_CHOICE_PROMPT = "Write your final answer in the format 'Answer: X' where X is the option's letter from the given choices. If the answer is not in the choices, please give the closest option."
# OPEN_ENDED_PROMPT = "Write your final answer in the format 'Answer: X' where X is a single word or phrase that answers the question."

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

def parse_VWImodel_response(response):
    if not response:
        return None
    
    try:
        # 移除markdown标记
        response = response.replace('*', '')
        
        # 寻找最后一个 "Answer:" 后的内容
        answer_index = response.rfind("Answer:")
        if answer_index == -1:
            return None
            
        # 提取答案部分
        answer_part = response[answer_index:].strip()
        
        # 提取字母答案（去除空白字符和标点符号）
        answer = answer_part.split(":")[-1].strip().strip("., ")
        
        return answer
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None

def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def construct_prompt(doc):
    question = doc["question"]
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = parse_options(ast.literal_eval(doc["options"]))
        # parsed_options already prepends a newline so no need to add space here
        question = f"{question}\n{parsed_options}\n\n{MULTI_CHOICE_PROMPT}"
    else:
        question = f"{question}\n\n{OPEN_ENDED_PROMPT}"
    return question


def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    if config["metadata"]["interleaved_format"]:
        question = replace_images_tokens(question)
    return question


def mmmu_doc_to_visual(doc):
    prompt = construct_prompt(doc)
    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and  swap space as _
    image_tokens = sorted(list(set([image_token.strip("<>").replace(" ", "_") for image_token in image_tokens])))
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    return visual


def mmmu_process_results(doc, results):
    pred = results[0]
    if doc["question_type"] == "multiple-choice":
        index2ans, all_choices = get_multi_choice_info(ast.literal_eval(doc["options"]))
        parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    else:
        parsed_pred = parse_open_response(pred)
    id = doc["id"]
    mmmu_acc = {"id": id, "subdomain": extract_subset_name(doc["id"]), "question_type": doc["question_type"], "answer": doc["answer"], "parsed_pred": parsed_pred}
    return {
        "mmmu_acc": mmmu_acc,
        "submission": {
            id: pred,
        },
    }


def extract_subset_name(input_string):
    # Define a regex pattern to match "validation_" at the beginning and "_<number>" at the end
    split = input_string.split("_")[0]
    pattern = re.compile(rf"^{split}_(.+?)_\d+$")
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')


def mmmu_test_aggregate_results_for_submission(results, args):
    path = generate_submission_file("mmmu_test_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")


def mmmu_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["subdomain"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_mmmu(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results["num_example"] for cat_results in in_domain_cat_results.values()])
        printable_results["Overall-" + domain] = {
            "num": int(in_domain_data_num),
            "acc": round(in_domain_ins_acc, 5),
        }
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
            }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    print(printable_results)
    return printable_results["Overall"]["acc"]


##################
# Helper functions written by official MMMU repo.
##################


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def evaluate_mmmu(samples):
    """
    Batch evaluation for multiple choice and open questions.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L219
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        if sample["question_type"] == "multiple-choice":
            correct = eval_multi_choice(gold_i, pred_i)
        else:  # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    # 使用新的解析函数
    parsed_answer = parse_VWImodel_response(response)
    if parsed_answer is None:
        # 如果解析失败,随机选择一个选项
        return random.choice(all_choices)
        
    # 确保解析出的答案是有效的选项
    if parsed_answer in all_choices:
        return parsed_answer
    
    # 如果解析出的答案无效,返回随机选项
    return random.choice(all_choices)


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L100
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbersz
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def check_is_number(string):
    """
    Check if the given string a number.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L65
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L76
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def parse_open_response(response):
    # 使用新的解析函数
    parsed_answer = parse_VWImodel_response(response)
    if parsed_answer is None:
        return []
        
    # 转换为列表格式并标准化
    pred_list = []
    pred_list.extend(normalize_str(parsed_answer))
    
    # 如果是数字,也添加提取的数字
    numbers = extract_numbers(parsed_answer)
    for num in numbers:
        pred_list.extend(normalize_str(num))
    
    # 去重
    pred_list = list(set(pred_list))
    
    return pred_list


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices
