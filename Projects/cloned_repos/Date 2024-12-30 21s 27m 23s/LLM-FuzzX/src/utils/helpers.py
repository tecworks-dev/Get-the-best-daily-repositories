# src/utils/helpers.py
"""
Helper functions.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Union
from src.utils.language_utils import detect_and_translate

def load_harmful_questions(file_path: str, task: str = None) -> List[Tuple[str, str]]:
    """
    Loads harmful questions from a file.

    Args:
        file_path (str): Path to the harmful questions file.
        task (str): Optional task description to prepend to each question.

    Returns:
        list: List of harmful questions.
    """
    logger = logging.getLogger('main')
    
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Questions file not found: {file_path}")
            raise FileNotFoundError(f"Questions file not found: {file_path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
            
        if not questions:
            logger.warning("No questions found in the file")
            raise ValueError("No questions found in the file")
            
        # 处理每个问题的语言检测和翻译
        processed_questions = []
        for q in questions:
            # 如果有task，在问题前拼接task描述
            if task:
                q = f"{task}\n{q}"
            original, translated = detect_and_translate(q)
            processed_questions.append((original, translated))
            
        logger.debug(f"Loaded {len(processed_questions)} questions from {file_path}")
        return processed_questions
        
    except Exception as e:
        logger.error(f"Error loading questions from {file_path}: {e}")
        raise

def load_harmful_questions_from_list(questions: List[Union[str, Tuple[str, str]]], task: str = None) -> List[Tuple[str, str]]:
    """
    Loads harmful questions from a list.
    
    Args:
        questions: List of strings or tuples containing questions
        task: Optional task description to prepend to each question
        
    Returns:
        List[Tuple[str, str]]: List of processed (original, translated) questions
    """
    processed_questions = []
    logger = logging.getLogger('main')
    
    try:
        for q in questions:
            # 如果输入是单个字符串或元组的第一个元素，都需要进行翻译
            if isinstance(q, str):
                if task:
                    q = f"{task}\n{q}"
                original, translated = detect_and_translate(q)
            else:
                # 如果是元组，仍然需要翻译第一个元素
                if task:
                    q = (f"{task}\n{q[0]}", q[1])
                original, translated = detect_and_translate(q[0])
            processed_questions.append((original, translated))
            
        logger.info(f"Loaded {len(processed_questions)} questions from list")
        return processed_questions
        
    except Exception as e:
        logger.error(f"Error processing questions: {e}")
        raise
