# src/models/llm_wrapper.py
from abc import ABC, abstractmethod
from functools import wraps
import time
import logging
from typing import List, Union

def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_delay: float = 60
):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if retry == max_retries - 1:  # 最后一次重试失败
                        break
                        
                    delay = min(delay * exponential_base, max_delay)
                    logging.warning(
                        f"调用失败,{delay}秒后进行第{retry + 1}次重试... 错误: {str(e)}"
                    )
                    time.sleep(delay)
            
            # 所有重试都失败后，抛出最后一个异常
            raise last_exception
            
        return wrapper
    return decorator

class LLMWrapper(ABC):
    """
    Abstract base class for LLM wrappers.
    """

    # 定义全局支持的模型列表，不同厂商的模型分组放在这里
    SUPPORTED_MODELS = {
        "openai": [
            "text-davinci-003",
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o"
        ],
        "huggingface": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf"
        ],
        "claude": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    }

    def validate_model_name(self, vendor: str, model_name: str):
        return True

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Union[str, List[str]]:
        """生成文本"""
        pass
