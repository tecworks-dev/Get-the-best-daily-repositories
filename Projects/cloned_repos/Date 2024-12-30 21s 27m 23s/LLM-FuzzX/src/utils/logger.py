# src/utils/logger.py
"""
Logger setup and utilities.
"""

import logging
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

def setup_multi_logger(log_dir: Path, level: str = 'INFO', log_to_console: bool = True) -> dict:
    """
    设置分层日志系统
    
    Args:
        log_dir: 日志目录路径
        level: 日志级别
        log_to_console: 是否输出到控制台
        
    Returns:
        dict: 包含各个logger的字典
    """
    # 确保日志目录存在
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 基础日志格式
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 定义不同的logger和对应的文件
    loggers = {
        'main': {  # 主要流程日志
            'file': log_dir / 'main.log',
            'level': level
        },
        'mutation': {  # 变异操作日志
            'file': log_dir / 'mutation.log',
            'level': level
        },
        'jailbreak': {  # 越狱日志
            'file': log_dir / 'jailbreak.log',
            'level': level
        },
        'error': {  # 错误日志
            'file': log_dir / 'error.log',
            'level': 'ERROR'
        }
    }
    
    # 创建总日志文件的handler
    all_handler = logging.FileHandler(log_dir / 'all.log', encoding='utf-8')
    all_handler.setFormatter(formatter)
    all_handler.setLevel(getattr(logging, level))
    
    # 创建控制台handler
    console_handler = None
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level))
    
    # 创建和配置每个logger
    logger_dict = {}
    for name, config in loggers.items():
        # 创建logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config['level']))
        logger.propagate = False  # 防止日志传播到父logger
        
        # 清除现有的handlers
        logger.handlers.clear()
        
        # 添加文件处理器
        file_handler = logging.FileHandler(config['file'], encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, config['level']))
        logger.addHandler(file_handler)
        
        # 添加总日志处理器
        logger.addHandler(all_handler)
        
        # 添加控制台处理器
        if console_handler:
            logger.addHandler(console_handler)
            
        logger_dict[name] = logger
    
    return logger_dict
