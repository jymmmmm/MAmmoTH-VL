import json
import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mathvision.mathvision_evals import MathVisionEvaluator

# 配置加载部分保持不变
with open(Path(__file__).parent / "mathvision.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# API配置部分保持不变
API_TYPE = os.getenv("API_TYPE", "openai")
if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

mathvision_evaluator = MathVisionEvaluator(api_key=API_KEY, gpt_model=config["metadata"]["gpt_eval_model_name"])


def mathvision_doc_to_visual(doc):
    """
    处理文档中的图像
    注意：只关注decoded_image和image字段
    """
    # 如果有decoded_image字段且非空
    if "decoded_image" in doc and doc["decoded_image"] is not None:
        return [doc["decoded_image"].convert("RGB")]
    
    # 如果有image路径字段，尝试加载图像
    elif "image" in doc and doc["image"]:
        try:
            from PIL import Image
            img_path = doc["image"]
            if not os.path.isabs(img_path):
                # 假设图像路径是相对于某个基础目录的
                base_dir = os.getenv("MATHVISION_IMAGES_DIR", ".")
                img_path = os.path.join(base_dir, img_path)
            
            img = Image.open(img_path)
            return [img.convert("RGB")]
        except Exception as e:
            eval_logger.error(f"Error loading image {doc.get('image')}: {e}")
            return []
    
    return []


def mathvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    生成文本查询
    注意：只关注question, options, answer, level字段
    """
    # 准备问题数据
    choices = doc.get("options", [])
    if isinstance(choices, str):
        try:
            # 尝试解析字符串格式的选项
            choices = json.loads(choices.replace("'", '"'))
        except:
            choices = choices.strip('[]').replace("'", "").split(", ")
    
    # 确定问题类型和答案类型
    question_type = "multiple_choice" if choices else "open_ended"
    answer = doc.get("answer")
    answer_type = "text"
    
    # 构建问题对象
    problem = {
        "question": doc["question"],
        "choices": choices,
        "answer": answer,
        "question_type": question_type,
        "answer_type": answer_type,
    }
    
    # 创建查询提示
    query_prompt = mathvision_evaluator.create_one_query(
        problem,
        shot_num=lmms_eval_specific_kwargs["shot"],
        shot_type=lmms_eval_specific_kwargs["shot_type"],
        use_caption=lmms_eval_specific_kwargs["use_caption"],
        use_ocr=lmms_eval_specific_kwargs["use_ocr"],
    )
    
    # 保存查询到文档中
    doc["query"] = query_prompt
    return query_prompt


def mathvision_process_results(doc, results):
    """
    处理模型预测结果
    注意：只关注id, question, options, answer字段
    """
    prediction = results[0].strip()
    
    # 准备问题数据
    choices = doc.get("options", [])
    if isinstance(choices, str):
        try:
            choices = json.loads(choices.replace("'", '"'))
        except:
            choices = choices.strip('[]').replace("'", "").split(", ")
    
    # 确定问题类型和答案类型
    question_type = "multiple_choice" if choices else "open_ended"
    answer_type = "text"
    
    problem = {
        "query": doc.get("query", doc["question"]),
        "choices": choices,
        "answer": doc.get("answer"),
        "question_type": question_type,
        "answer_type": answer_type,
    }
    
    # 提取答案
    extraction = mathvision_evaluator.extract_answer(prediction, problem, config["metadata"]["quick_extract"])
    
    # 规范化提取的答案
    prediction = mathvision_evaluator.normalize_extracted_answer(
        extraction, 
        problem["choices"], 
        problem["question_type"], 
        problem["answer_type"], 
    )
    
    # 验证答案
    true_false = mathvision_evaluator.safe_equal(prediction, problem["answer"]) if problem["answer"] is not None else False
    
    # 构建结果
    result = {
        "question_id": doc["id"],
        "query": problem["query"],
        "choices": choices,
        "answer": doc.get("answer"),
        "extraction": extraction,
        "prediction": prediction,
        "true_false": true_false,
        "level": doc.get("level"),
        "metadata": {
            "split": "testmini"
        }
        
    }
    
    return {
        "gpt_eval_score": result,
        "submission": result,
    }


def mathvision_aggregate_results(results, args, *, calculate_gain=False, random_scores=None):
    """
    聚合评估结果，与原代码保持一致，但只关注必要的目标分析类别
    """
    split_flag = results[0]["metadata"]["split"]
    full_pids = [result["question_id"] for result in results]
    total = len(results)
    correct = sum(1 for idx, pid in enumerate(full_pids) if results[idx]["true_false"])
    accuracy = round(correct / total * 100, 2)
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    for result in results:
        result.update(result.pop("metadata"))

    results_dict = {result["question_id"]: result for result in results}
    df = pd.DataFrame(results_dict).T
    
    # 仅分析必要的目标类别
    target_keys = ["question_type", "answer_type", "level", "grade"]

    for key in target_keys:
        if key not in df.columns:
            continue
            
        values = df[key].explode().unique()
        scores[key] = {}
        for value in values:
            if pd.isna(value):
                continue
            correct, total, acc = mathvision_evaluator.get_acc_with_contion(df, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))

    if calculate_gain and random_scores:
        for key in scores:
            if key == "average":
                gain = round(float(scores[key]["accuracy"]) - float(random_scores[key]["accuracy"]), 2)
                scores[key]["acc_gain"] = gain
            else:
                for sub_key in scores[key]:
                    if sub_key in random_scores.get(key, {}):
                        gain = round(float(scores[key][sub_key]["accuracy"]) - float(random_scores[key][sub_key]["accuracy"]), 2)
                        scores[key][sub_key]["acc_gain"] = gain

    path = generate_submission_file(f"mathvision_{split_flag}_scores.json", args)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=4)
    eval_logger.info(f"Saved results to {path}")
    if scores["average"]["accuracy"] == 0:
        return None
    return scores["average"]["accuracy"]