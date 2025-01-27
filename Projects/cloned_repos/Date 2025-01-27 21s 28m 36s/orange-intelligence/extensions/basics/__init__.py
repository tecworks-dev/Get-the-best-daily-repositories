import json


def upper_case(text: str, **kwargs) -> str:
    return text.upper()


def lower_case(text: str, **kwargs) -> str:
    return text.lower()


def pretty_json(text: str, **kwargs) -> str:
    return json.dumps(json.loads(text), indent=4)


def a_complex_task_you_do_not_want_to_implement_now(text: str, **kwargs) -> str:
    return "This is a complex task that you do not want to implement now."
