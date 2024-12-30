from langdetect import detect
from google.cloud import translate_v2 as translate
import logging
from typing import Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ...config import TRANSLATE_API_KEY
logger = logging.getLogger('main')

def translate_text(target: str, text: str) -> dict:
    """翻译文本的函数"""
    translate_client = translate.Client(
        client_options={"api_key": TRANSLATE_API_KEY}
    )

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    result = translate_client.translate(text, target_language=target)

    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result

def fallback_return_original(text: str) -> Tuple[str, str]:
    """重试失败后的回调函数"""
    logger.warning("All retry attempts failed. Returning the original text.")
    return text, text

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    retry_error_callback=lambda retry_state: fallback_return_original(retry_state.args[0])
)
def detect_and_translate(text: str) -> Tuple[str, str]:
    """检测文本语言并在需要时翻译成英文"""
    # 检测语言
    lang = detect(text)
    
    # 如果是英文,直接返回
    if lang == 'en':
        return text, text

    # 如果是其他语言,翻译成英文
    translation = translate_text('en', text)
    return text, translation['translatedText']
