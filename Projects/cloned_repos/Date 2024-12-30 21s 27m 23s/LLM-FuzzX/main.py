# main.py
"""
Main script to run the fuzzing process.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

from src.models.openai_model import OpenAIModel
from src.fuzzing.fuzzing_engine import FuzzingEngine
from src.fuzzing.seed_selector import UCBSeedSelector
from src.fuzzing.mutator import LLMMutator
from src.evaluation.roberta_evaluator import RoBERTaEvaluator
from src.utils.logger import setup_multi_logger
from src.utils.helpers import load_harmful_questions
from src.fuzzing.seed_selector import SeedManager

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fuzzing LLMs to generate jailbreak prompts.')
    
    # 模型相关参数
    parser.add_argument('--target_model', type=str, default='gpt-3.5-turbo', 
                        help='Target model name')
    parser.add_argument('--mutator_model', type=str, default='gpt-4o',
                        help='Model used for mutation')
    
    # 数据相关参数
    parser.add_argument('--seed_path', type=str, 
                        default='data/seeds/GPTFuzzer.csv',
                        help='Path to seed prompts')
    parser.add_argument('--questions_path', type=str,
                        default='data/questions/harmful_questions.txt',
                        help='Path to harmful questions')
    
    # Fuzzing参数
    parser.add_argument('--max_iterations', type=int, default=5,
                        help='Maximum number of iterations')
    parser.add_argument('--exploration_weight', type=float, default=1.0,
                        help='Exploration weight for UCB')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for LLM mutation')
    parser.add_argument('--mutator_temperature', type=float, default=0.0,
                        help='Temperature for mutator model')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='logs',
                        help='Directory to save results')
    parser.add_argument('--log_level', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    # 最大成功次数
    parser.add_argument('--max_successes', type=int, default=3,
                        help='Maximum number of successful jailbreak attempts')
    
    parser.add_argument('--selector_type', type=str, default='ucb',
                        choices=['ucb', 'diversity_ucb'],
                        help='Type of seed selector')
    parser.add_argument('--save-name', type=str, default='',
                        help='Directory to save results')
    parser.add_argument('--task', type=str, default='',
                        help='Task aware')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use')
    return parser.parse_args()

def setup_output_dir(output_dir: str) -> Path:
    """设置输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def save_experiment_config(output_path: Path, args: argparse.Namespace):
    """保存实验配置参数到文件"""
    config = vars(args)  # 将参数转换为字典
    config_file = output_path / 'experiment_config.json'
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def log_dict(logger: logging.Logger, data: dict, level: int = logging.INFO):
    """格式化记录字典数据"""
    logger.log(level, json.dumps(data, indent=2, ensure_ascii=False))

def main():
    # 设置代理 7890
    import os
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'

    # 解析参数
    args = parse_arguments()
    
    # 设置输出目录
    if args.save_name != '':
        output_path = Path(args.output_dir) / args.save_name
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = setup_output_dir(args.output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存实验配置
    save_experiment_config(output_path, args)
    
    # 设置日志系统
    loggers = setup_multi_logger(
        level=args.log_level,
        log_dir=output_path,
        log_to_console=True
    )
    
    try:
        # 初始化种子管理器
        seed_manager = SeedManager(save_dir=output_path / 'seeds')
        
        # 加载初始种子
        initial_seeds = load_seeds(args.seed_path)
        for seed in initial_seeds:
            seed_manager.add_seed(content=seed)
            
        # 加载问题时传入task参数
        harmful_questions = load_harmful_questions(args.questions_path, task=args.task)
        
        # 记录初始状态
        loggers['main'].info(json.dumps({
            "initial_seeds": len(initial_seeds),
            "harmful_questions": len(harmful_questions),
            "target_model": args.target_model,
            "mutator_model": args.mutator_model,
            "task": args.task  # 记录task参数
        }))
        
        # 初始化模型
        target_model = OpenAIModel(model_name=args.target_model)
        mutator_model = OpenAIModel(model_name=args.mutator_model)
        loggers['main'].info("Initialized target and mutator models")
        
        # 初始化评估器
        evaluator = RoBERTaEvaluator(device="cuda:"+args.device)
        loggers['main'].info("Initialized RoBERTa evaluator")
        
        # 初始化变异器
        mutator = LLMMutator(
            llm=mutator_model,
            seed_manager=seed_manager,
            temperature=args.mutator_temperature
        )
        loggers['main'].info("Initialized LLM mutator")
        
        # 初始化种子选择器
        seed_selector = UCBSeedSelector(
            seed_manager=seed_manager,
            exploration_weight=args.exploration_weight
        )
        loggers['main'].info("Initialized UCB seed selector")
        
        # 初始化fuzzing引擎
        engine = FuzzingEngine(
            target_model=target_model,
            seed_selector=seed_selector,
            mutator=mutator,
            evaluator=evaluator,
            questions=harmful_questions,
            max_iterations=args.max_iterations,
            save_results=True,
            results_file=output_path / 'results.txt',
            success_file=output_path / 'successful_jailbreaks.csv',
            summary_file=output_path / 'experiment_summary.txt',
            seed_flow_file=output_path / 'seed_flow.json',
            max_successes=args.max_successes,
            loggers=loggers
        )
        loggers['main'].info("Initialized fuzzing engine")
        
        # 确保引擎运行前记录状态
        loggers['main'].info("Starting fuzzing engine...")
        
        # 运行fuzzing过程
        engine.run()
        
    except Exception as e:
        loggers['error'].error(f"Error in main: {str(e)}")
        raise
    

def load_seeds(seed_path):
    """载种子数据"""
    try:
        df = pd.read_csv(seed_path)
        return df['text'].tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to load seeds from {seed_path}: {e}")

if __name__ == '__main__':
    main()
