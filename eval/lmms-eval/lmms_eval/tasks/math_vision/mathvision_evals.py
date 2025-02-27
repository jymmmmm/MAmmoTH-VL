import os
import re
import time
import json

import requests
from Levenshtein import distance
from loguru import logger as eval_logger

# 保持原有的示例数据
shot_examples = [
    {
        "question": "How much money does Ruth need to buy a baking dish, a casserole dish, and an ice cream scoop? (Unit: $)",
        "caption": "The image shows a table with a variety of items on it, including a baking dish, ice cream scoop, casserole dish, and rolling pin. The text in the image says:\n\n```\nbaking dish\n$4.00\nice cream scoop\n$6.00\ncasserole dish\n$3.00\nrolling pin\n$4.00\n```",
        "ocr": "[([5, 3], 'baking dish'), ([177, 5], '$4.00'), ([7, 41], 'ice cream scoop'), ([177, 37], '$6.00'), ([9, 69], 'casserole dish'), ([177, 69], '$3.00'), ([5, 98], 'rolling pin'), ([177, 101], '$4.00')]",
        "solution": """
Find the total cost of a baking dish, a casserole dish, and an ice cream scoop.\n\n$4.00 + $3.00 + $6.00 = $13.00\n\nRuth needs $13.00.
""",
        "code": """
baking_dish_price = 4.00
casserole_dish_price = 3.00
ice_cream_scoop_price = 6.00

ans = baking_dish_price + casserole_dish_price + ice_cream_scoop_price
print(ans)
""",
    },
    {
        "question": "What is the largest city in the nation where this plane is headquartered?",
        "choices": ["hong kong", "osaka", "shanghai", "tokyo"],
        "caption": 'The image shows a large passenger jet parked on a tarmac at an airport. The jet is white with red trim and has a red tail. It is sitting on top of a tarmac next to a building. The jet is being loaded with passengers and cargo. The text on the image says "Japan. Endless Discovery".',
        "solution": """
The caption mentions that the text on the image says "Japan. Endless Discovery". This indicates that the plane is headquartered in Japan. 

Among the Japanese cities, Tokyo is the largest city.

Thus, the answer is D (tokyo).
""",
        "code": """
def largest_city(caption, choices):
    countries_largest_cities = {
        'Japan': 'tokyo',
        'China': 'shanghai'
    }

    if "Japan" in caption:
        country = 'Japan'
    elif "China" in caption:
        country = 'China'

    for choice in choices:
        if choice == countries_largest_cities[country]:
            return choice
    return ""

choices = ['hong kong', 'osaka', 'shanghai', 'tokyo']
caption = "The image shows a large passenger jet parked on a tarmac at an airport. The jet is white with red trim and has a red tail. It is sitting on top of a tarmac next to a building. The jet is being loaded with passengers and cargo. The text on the image says 'Japan. Endless Discovery'."

print(largest_city(caption, choices))
""",
    },
    {
        "question": "If two sides of a triangle measure 12 and 7, which of the following cannot be the perimeter of the triangle?",
        "choices": ["29", "34", "37", "38"],
        "caption": "The image shows a triangle with two sides labeled 7 and 12. The triangle is drawn on a white background. There is no text other than the labels.",
        "ocr": "[([70, 74], '7'), ([324, 74], '12')]",
        "solution": """
To determine which of the given perimeters cannot be possible for the triangle, we apply the triangle inequality theorem. The sum of any two sides of a triangle must be greater than the third side.

For the maximum possible value of the third side:
12 + 7 = 19

The minimum possible value for the third side:
12 - 7 = 5

The third side for each option:
(A) 29 - 12 - 7 = 10 (valid)
(B) 34 - 12 - 7 = 15 (valid)
(C) 37 - 12 - 7 = 18 (valid)
(D) 38 - 12 - 7 = 19 (invalid because it should be less than 19)

Thus, the answer is D.
""",
        "code": """
def is_valid_triangle(a, b, perimeter):
    # Given a and b, find the third side
    third_side = perimeter - a - b
    
    # Check triangle inequality
    if (a + b > third_side) and (a + third_side > b) and (b + third_side > a):
        return True
    return False

# Given sides
a = 12
b = 7

# Given perimeters
perimeters = [29, 34, 37, 38]

# Check which perimeter is not valid
for p in perimeters:
    if not is_valid_triangle(a, b, p):
        print(p)
""",
    },
]

DEMO_PROMPT = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""


class MathVistaEvaluator:
    """MathVista评估器类，简化版只关注必要字段"""
    
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

    def __init__(self, api_key, gpt_model="gpt-3.5-turbo", quick_extract=False):
        self.api_key = api_key
        self.gpt_model = gpt_model
        self.quick_extract = quick_extract

    def _post_request(self, payload):
        """发送POST请求到API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_chat_response(self, prompt, temperature=0, max_tokens=256, n=1, patience=5, sleep_time=0):
        """获取聊天模型的响应"""
        messages = [
            {"role": "user", "content": prompt},
        ]
        payload = {"model": self.gpt_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "n": n}

        if self.API_TYPE == "azure":
            payload.pop("model")

        while patience > 0:
            patience -= 1
            try:
                response = self._post_request(payload)
                if n == 1:
                    prediction = response["choices"][0]["message"]["content"].strip()
                    if prediction and prediction != "":
                        return prediction
                else:
                    prediction = [choice["message"]["content"].strip() for choice in response["choices"]]
                    if prediction and prediction[0] != "":
                        return prediction

            except Exception as e:
                if "Rate limit" not in str(e):
                    eval_logger.error(e)

                if "Please reduce the length of the messages" in str(e):
                    eval_logger.error("!!Reduce prompt size")
                    # reduce input prompt and keep the tail
                    new_size = int(len(prompt) * 0.9)
                    new_start = len(prompt) - new_size
                    prompt = prompt[new_start:]
                    payload["messages"] = [
                        {"role": "user", "content": prompt},
                    ]

                if sleep_time > 0:
                    time.sleep(sleep_time)
        return ""

    def create_test_prompt(self, demo_prompt, query, response):
        """创建用于提取答案的测试提示"""
        demo_prompt = demo_prompt.strip()
        test_prompt = f"{query}\n\n{response}"
        full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
        return full_prompt

    def extract_answer(self, response, problem, quick_extract=False):
        """从模型响应中提取答案"""
        question_type = problem.get("question_type", "open_ended")
        answer_type = problem.get("answer_type", "text")
        choices = problem.get("choices", [])
        query = problem.get("query", "")

        if not response:
            return ""

        if question_type == "multi_choice" and response in choices:
            return response

        if answer_type == "integer" or answer_type == "number":
            try:
                extraction = int(response)
                return str(extraction)
            except ValueError:
                pass

        if answer_type == "float":
            try:
                extraction = str(float(response))
                return extraction
            except ValueError:
                pass

        # quick extraction
        if quick_extract:
            eval_logger.info("Quickly extracting answer...")
            # The answer is "text". -> "text"
            try:
                result = re.search(r'The answer is "(.*)"\.', response)
                if result:
                    extraction = result.group(1)
                    return extraction
            except re.error:
                pass

        # general extraction
        try:
            full_prompt = self.create_test_prompt(DEMO_PROMPT, query, response)
            extraction = self.get_chat_response(full_prompt, temperature=0, max_tokens=256, n=1)
            return extraction
        except Exception as e:
            eval_logger.error(e)
            eval_logger.error(f"Error in extracting answer for problem")

        return ""

    def get_most_similar(self, prediction, choices):
        """获取最相似的选项"""
        if not choices:
            return prediction
            
        distances = [distance(str(prediction), str(choice)) for choice in choices]
        ind = distances.index(min(distances))
        return choices[ind]

    def normalize_extracted_answer(self, extraction, choices, question_type, answer_type, precision):
        """规范化提取的答案"""
        # 处理选项
        if isinstance(choices, str):
            try:
                choices = json.loads(choices.replace("'", '"'))
            except:
                try:
                    choices = choices.strip('[]').replace("'", "").split(", ")
                except:
                    choices = []
                
        # 确保类型有默认值
        question_type = question_type or "open_ended"
        answer_type = answer_type or "text"
        precision = precision or 0.01
                
        if question_type == "multi_choice" or question_type == "multiple_choice":
            # 确保提取的答案是字符串
            if not isinstance(extraction, str):
                try:
                    extraction = str(extraction)
                except:
                    extraction = ""
            
            extraction = extraction.strip()

            # 从"(A) text"中提取"A"
            letter = re.findall(r"\(([a-zA-Z])\)", extraction)
            if len(letter) > 0:
                extraction = letter[0].upper()

            if choices:
                options = [chr(ord("A") + i) for i in range(len(choices))]

                if extraction in options:
                    # 将选项字母转换为文本，例如"A"->"text"
                    ind = options.index(extraction)
                    extraction = choices[ind]
                else:
                    # 选择最相似的选项
                    extraction = self.get_most_similar(extraction, choices)
                    
                if str(extraction) not in [str(choice) for choice in choices]:
                    extraction = self.get_most_similar(extraction, choices)

        elif answer_type == "integer" or answer_type == "number":
            try:
                extraction = str(int(float(extraction)))
            except:
                extraction = None

        elif answer_type == "float":
            try:
                extraction = str(round(float(extraction), int(precision)))
            except:
                extraction = None

        elif answer_type == "list":
            try:
                extraction = str(extraction)
            except:
                extraction = None

        return extraction

    def safe_equal(self, prediction, answer):
        """安全地比较预测答案和真实答案"""
        if prediction is None or answer is None:
            return False
            
        try:
            pred_str = str(prediction).strip().lower()
            ans_str = str(answer).strip().lower()
            
            # 字符串精确匹配
            if pred_str == ans_str:
                return True
                
            # 尝试数值比较
            try:
                pred_num = float(pred_str)
                ans_num = float(ans_str)
                return abs(pred_num - ans_num) < 1e-6
            except:
                pass
                
            return False
        except Exception as e:
            eval_logger.info(e)
            return False

    def get_acc_with_contion(self, res_pd, key, value):
        """计算特定条件下的准确率"""
        # 处理不同类型的键
        if key not in res_pd.columns:
            return 0, 0, "0.00"
            
        if key == "skills":
            total_pd = res_pd[res_pd[key].apply(lambda x: value in x if isinstance(x, list) else False)]
        else:
            # 直接用等于比较，而不是字符串强制转换
            if isinstance(value, (int, float)):
                total_pd = res_pd[res_pd[key] == value]
            else:
                # 对于字符串和其他类型，尝试精确匹配
                total_pd = res_pd[res_pd[key].astype(str) == str(value)]

        correct_pd = total_pd[total_pd["true_false"] == True]
        acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100) if len(total_pd) > 0 else "0.00"
        return len(correct_pd), len(total_pd), acc

    def create_one_query(self, problem, shot_type="solution", examples=shot_examples, shot_num=0, use_caption=False, use_ocr=False):
        """创建一个查询提示，只关注必要的字段"""
        # 提取问题信息，为缺失字段提供默认值
        question = problem.get("question", "")
        unit = problem.get("unit", "")
        
        # 处理选项字段
        choices = problem.get("choices", [])
        if isinstance(choices, str):
            try:
                choices = json.loads(choices.replace("'", '"'))
            except:
                try:
                    choices = choices.strip('[]').replace("'", "").split(", ")
                except:
                    choices = []
        
        # 设置默认值
        caption = problem.get("caption", "")
        ocr = problem.get("ocr", "")
        precision = problem.get("precision", 0.01)
        question_type = problem.get("question_type", "multiple_choice" if choices else "open_ended")
        answer_type = problem.get("answer_type", "number" if isinstance(problem.get("answer"), (int, float)) else "text")
        
        ### [1] 示例提示
        if shot_num == 0:
            demo_prompt = ""
        else:
            demos = []
            shot_num = min(shot_num, len(examples))
            for example in examples[:shot_num]:
                prompt = ""

                # 问题
                prompt += f"Question: {example['question']}"

                # 选项
                if "choices" in example:
                    texts = ["Choices:"]
                    for i, choice in enumerate(example["choices"]):
                        texts.append(f"({chr(ord('A')+i)}) {choice}")
                    prompt += "\n" + "\n".join(texts)

                # 图像描述
                if use_caption:
                    caption = example["caption"] if "caption" in example else ""
                    if caption != "":
                        prompt += "\n" + f"Image description: {caption}"

                # OCR文本
                if use_ocr:
                    ocr = example["ocr"] if "ocr" in example else ""
                    if ocr != "":
                        prompt += "\n" + f"Image detected text: {ocr}"

                # 解决方案
                if shot_type == "solution":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"Solution: {solution}"
                elif shot_type == "step-by-step" or shot_type == "think-step-by-step" or shot_type == "direct":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"{solution}"
                elif shot_type == "code":
                    code = example["code"].strip()
                    prompt += "\n" + f"Python code: {code}"

                demos.append(prompt)

            demo_prompt = "\n\n".join(demos)

        ### [2] 测试查询
        # 提示文本
        if shot_type == "solution":
            if question_type == "multi_choice" or question_type == "multiple_choice":
                hint_text = f"Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            else:
                if answer_type == "integer" or answer_type == "number":
                    hint_text = f"Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."
                elif answer_type == "float" and precision == 1:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."
                elif answer_type == "float" and precision == 2:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."
                elif answer_type == "list":
                    hint_text = f"Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
                else:  # text
                    hint_text = f"Hint: Please answer the question with a text response."