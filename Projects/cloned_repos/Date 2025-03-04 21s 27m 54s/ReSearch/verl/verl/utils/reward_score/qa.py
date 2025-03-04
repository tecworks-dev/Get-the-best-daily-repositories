import re

def validate_template_format(text: str) -> tuple[bool, str]:
    """
    check if the text is a valid qa template
    return: (is_valid, error_message)
    """
    # check if <think></think> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> is not paired"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "missing <think> or </think> tags"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> appears more than once"        
    
    # check the order of search/result tags
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break
            
        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested incorrectly"
            
        current_pos = result_end_pos
    
    # check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"
    
    return True, "format is correct"

def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    return match.group(1)

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def compute_score_with_format(solution_str, ground_truth) -> float:
    solution_str_split = solution_str.split("Assistant:")
    response = solution_str_split[1]
    valid_template, reason = validate_template_format(response)
    if not valid_template:
        return 0, f'bad format: {reason}'

    if response.endswith("<|endoftext|>"):
        response = response[:-len("<|endoftext|>")]
    else:
        return 0, f'over length'

    answer_part = extract_answer(response)
    if answer_part is not None:
        try:
            answer = remove_boxed(last_boxed_only_string(answer_part))
        except Exception as e:
            return 0, f'find box error: {e}'
    else:
        return 0, f'cannot extract answer'

    if answer.lower() == ground_truth.lower():
        return 1, f'correct answer: {answer}'
    else:
        return 0.1, f'wrong answer but good format: {answer}'