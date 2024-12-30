import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json
import csv

class MutationLogger:
    """专门用于记录变异过程的logger"""
    
    def __init__(self, log_dir: str = "logs/mutations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置文件处理器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"mutation_{timestamp}.log"
        
        # 配置logger
        self.logger = logging.getLogger("mutation_logger")
        self.logger.setLevel(logging.DEBUG)
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )
        self.logger.addHandler(file_handler)
        
    def log_mutation_start(self, seed: str, mutation_type: str):
        """记录变异开始"""
        self.logger.info(f"Starting {mutation_type} mutation on seed: {seed[:100]}...")
        
    def log_mutation_result(self, 
                          original: str,
                          mutated: str,
                          mutation_type: str,
                          metadata: Dict[str, Any] = None):
        """增强的变异日志记录"""
        
        # 计算变异前后的统计信息
        stats = {
            "original_length": len(original),
            "mutated_length": len(mutated),
            "length_change": len(mutated) - len(original),
            "word_count_original": len(original.split()),
            "word_count_mutated": len(mutated.split()),
            "timestamp": datetime.now().isoformat()
        }
        
        # 记录变异信息
        self.logger.info(
            f"Mutation completed:\n"
            f"Type: {mutation_type}\n"
            f"Original: {original[:100]}...\n"
            f"Mutated: {mutated[:100]}...\n"
            f"Stats: {json.dumps(stats)}\n"
            f"Metadata: {json.dumps(metadata or {})}"
        )
        
        # 如果是成功的变异,保存到CSV
        if metadata and metadata.get("is_successful"):
            self._save_to_csv({
                "timestamp": stats["timestamp"],
                "mutation_type": mutation_type,
                "original": original,
                "mutated": mutated,
                "success": True,
                **stats,
                **(metadata or {})
            })
            
    def _save_to_csv(self, data: Dict[str, Any]):
        """保存成功的变异到CSV文件"""
        csv_file = self.log_dir / "successful_mutations.csv"
        
        # 检查文件是否存在
        file_exists = csv_file.exists()
        
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            
            # 如果是新文件,写入表头
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(data)
        
    def log_mutation_error(self, error: Exception, mutation_type: str):
        """记录变异错误"""
        self.logger.error(
            f"Mutation failed - Type: {mutation_type}, Error: {str(error)}",
            exc_info=True
        ) 
