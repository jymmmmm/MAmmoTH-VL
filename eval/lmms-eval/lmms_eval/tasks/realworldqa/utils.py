import re

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter
from lmms_eval.filters import Filter  

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."


def realworldqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        question = question.replace(REPLACE_PROMPT, "")
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


# number_words_to_digits = {
#     "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
#     "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
#     "ten": "10"
# }


def realworldqa_process_results(doc, results):
    pred = results[0].lower().strip().rstrip(".")
    gt_ans = doc["answer"].lower().strip()

    print(f"Prediction: {pred}, Ground Truth: {gt_ans}")
    # assert gt_ans in ["a", "b", "c", "d"]
    score = 1.0 if pred == gt_ans else 0.0
    return {
        "exact_match": score,
    }


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        mapping_dict = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        super().__init__(mapping_dict, default_value=None)

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp.lower(), resp) for resp in inst]

        return [filter_set(resp) for resp in resps]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(doc["question"])

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes))

            # Process each response
            filtered = []
            for resp in r:
                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                # Try to match cleaned response with the choice text
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0])

        return filtered_resps


class FlexibleAnswerExtractFilter(Filter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 匹配独立答案(单个单词、字母或数字)
        self.simple_pattern = re.compile(r"^(yes|no|[A-Za-z0-9]+)$", re.IGNORECASE)
        # 匹配 "Answer: X" 格式
        self.answer_pattern = re.compile(
            r"Answer:?\s*(yes|no|[A-Za-z0-9]+)", 
            re.IGNORECASE
        )
    
    def _format_answer(self, answer):
        # 将答案转为首字母大写
        return answer.capitalize()
    
    def _extract_answer(self, resp):
        # 先尝试匹配简单答案
        simple_match = self.simple_pattern.search(resp.strip())
        if simple_match:
            return self._format_answer(simple_match.group(1))
        
        # 如果不是简单答案,查找 "Answer: X" 格式
        match = self.answer_pattern.search(resp)
        if match:
            return self._format_answer(match.group(1))
        
        # 如果都没找到,返回原始答案
        return resp
    
    def apply(self, resps, docs):
        filtered_resps = []
        
        for resp_list in resps:
            # 确保我们总是处理第一个响应
            if resp_list and isinstance(resp_list, list):
                answer = self._extract_answer(resp_list[0])
                filtered_resps.append(answer)  # 直接添加字符串，而不是列表
            else:
                filtered_resps.append("")  # 如果没有响应，添加空字符串
        
        return filtered_resps  # 返回字符串列表，而不是列表的列表