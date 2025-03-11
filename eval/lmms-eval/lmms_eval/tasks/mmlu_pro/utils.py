from functools import partial

choices = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "First think step by step and then write your final answer in the format 'Answer: X' where X is the option's letter from the given choices."
    return prompt


doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")


def custom_extract_answer(response):
    """
    Extract answer from response text, handling various formats.
    Returns the letter or None if no valid answer is found.
    """
    if not response:
        return None
        
    cleaned_response = response.replace('*', '')
    
    answer_index = cleaned_response.rfind("Answer:")
    if answer_index != -1:
        answer_part = cleaned_response[answer_index:].strip()
        answer = answer_part.split(":")[-1].strip().strip("., ")
        
        if answer and answer[0].isalpha():
            return answer[0].upper()
        
        if answer:
            return answer.upper()
    
    import re
    answer_match = re.search(r"Answer:\s*([A-J])", cleaned_response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    letter_match = re.search(r"\b([A-J])\s*$", cleaned_response.strip(), re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    
    standalone_match = re.search(r"^\s*([A-J])\s*$", cleaned_response, re.IGNORECASE | re.MULTILINE)
    if standalone_match:
        return standalone_match.group(1).upper()
    
    return None