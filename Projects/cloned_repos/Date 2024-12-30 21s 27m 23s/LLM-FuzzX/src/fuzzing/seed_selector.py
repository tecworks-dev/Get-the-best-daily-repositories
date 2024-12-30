# src/fuzzing/seed_selector.py
"""
种子选择策略的实现。
包含多种选择算法,用于从种子池中选择下一个待测试的种子。
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Set, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

@dataclass
class SeedInfo:
    """种子信息数据类"""
    id: str  # 唯一标识符
    content: str  # 种子内容
    parent_id: Optional[str]  # 父种子ID
    mutation_type: Optional[str]  # 使用的变异方法
    creation_time: datetime  # 创建时间
    depth: int  # 在种子树中的深度
    children: Set[str] = field(default_factory=set)  # 子种子ID集合
    stats: Dict[str, Any] = field(default_factory=lambda: {  # 统计信息
        "uses": 0,
        "successes": 0,
        "total_trials": 0
    })
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

class SeedManager:
    """种子管理器"""
    
    def __init__(self, save_dir: str = "data/seed_history"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 种子信息字典
        self.seeds: Dict[str, SeedInfo] = {}
        # 根种子ID列表
        self.root_seeds: Set[str] = set()
        # 成功率缓存
        self.success_rates: Dict[str, float] = {}
        
        self.logger = logging.getLogger("seed_manager")
        
    def add_seed(self, 
                content: str,
                parent_id: Optional[str] = None,
                mutation_type: Optional[str] = None,
                metadata: Optional[Dict] = None) -> str:
        """
        添加新种子
        
        Args:
            content: 种子内容
            parent_id: 父种子ID
            mutation_type: 变异方法
            metadata: 额外元数据
            
        Returns:
            str: 新种子的ID
        """
        # 生成唯一ID
        seed_id = self._generate_id()
        
        # 计算深度
        depth = 0
        if parent_id:
            parent = self.seeds.get(parent_id)
            if parent:
                depth = parent.depth + 1
                parent.children.add(seed_id)
            
        # 创建种子信息对象
        seed_info = SeedInfo(
            id=seed_id,
            content=content,
            parent_id=parent_id,
            mutation_type=mutation_type,
            creation_time=datetime.now(),
            depth=depth,
            children=set(),
            stats={
                "uses": 0,
                "successes": 0,
                "total_trials": 0
            },
            metadata=metadata or {}
        )
        
        # 存储种子信息
        self.seeds[seed_id] = seed_info
        
        # 如果是根种子，添加到根种子集合
        if not parent_id:
            self.root_seeds.add(seed_id)
            
        # 保存到文件
        self._save_seed_info(seed_id)
        
        return seed_id
        
    def update_stats(self, seed_id: str, success: bool):
        """更新种子统计信息"""
        if seed_id not in self.seeds:
            return
            
        seed = self.seeds[seed_id]
        seed.stats["uses"] += 1
        seed.stats["total_trials"] += 1
        if success:
            seed.stats["successes"] += 1
            
        # 更新成功率缓存
        self.success_rates[seed_id] = seed.stats["successes"] / seed.stats["total_trials"]
        
        # 保存更新
        self._save_seed_info(seed_id)
        
    def get_success_rate(self, seed_id: str) -> float:
        """获取种子的成功率"""
        return self.success_rates.get(seed_id, 0.0)
        
    def get_children(self, seed_id: str) -> List[str]:
        """获取子种子列表"""
        seed = self.seeds.get(seed_id)
        return list(seed.children) if seed else []
        
    def get_ancestry(self, seed_id: str) -> List[str]:
        """获取种子的祖先链"""
        ancestry = []
        current = self.seeds.get(seed_id)
        
        while current and current.parent_id:
            ancestry.append(current.parent_id)
            current = self.seeds.get(current.parent_id)
            
        return ancestry
        
    def _generate_id(self) -> str:
        """生成唯一ID"""
        return f"seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.seeds)}"
        
    def _save_seed_info(self, seed_id: str):
        """保存种子信息到文件"""
        seed = self.seeds[seed_id]
        file_path = self.save_dir / f"{seed_id}.json"
        
        # 转换为可序列化的字典
        seed_dict = {
            "id": seed.id,
            "content": seed.content,
            "parent_id": seed.parent_id,
            "mutation_type": seed.mutation_type,
            "creation_time": seed.creation_time.isoformat(),
            "depth": seed.depth,
            "children": list(seed.children),
            "stats": seed.stats,
            "metadata": seed.metadata
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(seed_dict, f, ensure_ascii=False, indent=2)
            
    def load_seed_history(self):
        """从文件加载种子历史"""
        for file_path in self.save_dir.glob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 转换回SeedInfo对象
            seed_info = SeedInfo(
                id=data["id"],
                content=data["content"],
                parent_id=data["parent_id"],
                mutation_type=data["mutation_type"],
                creation_time=datetime.fromisoformat(data["creation_time"]),
                depth=data["depth"],
                children=set(data["children"]),
                stats=data["stats"],
                metadata=data["metadata"]
            )
            
            self.seeds[data["id"]] = seed_info
            if not data["parent_id"]:
                self.root_seeds.add(data["id"])

class SeedSelector(ABC):
    """
    种子选择器的基类,定义了选择策略的基本接口
    """
    def __init__(self, seed_manager: SeedManager, **kwargs):
        """
        初始化种子选择器。

        Args:
            seed_manager: 种子管理器实例
            **kwargs: 额外的参数
        """
        self.seed_manager = seed_manager
        if not self.seed_manager.seeds:
            raise ValueError("种子管理器中没有种子")
            
        self.logger = logging.getLogger('seed_selector')
        self.logger.info(f"初始化种子选择器,当前种子数量: {len(self.seed_manager.seeds)}")

    @abstractmethod
    def select_seed(self) -> tuple[str, str]:
        """
        选择下一个种子。

        Returns:
            tuple[str, str]: (seed_id, seed_content) 种子ID和内容的元组
        """
        pass

    def update_stats(self, seed: str, success: bool):
        """
        更新种子的统计信息。

        Args:
            seed: 使用的种子内容
            success: 是否成功
        """
        # 找到对应的种子ID
        seed_id = next(
            (id for id, info in self.seed_manager.seeds.items() 
             if info.content == seed),
            None
        )
        if seed_id:
            self.seed_manager.update_stats(seed_id, success)

class RandomSeedSelector(SeedSelector):
    """随机选择策略"""
    def select_seed(self) -> tuple[str, str]:
        if not self.seed_manager.seeds:
            raise ValueError("种子池为空")
        seed_id = random.choice(list(self.seed_manager.seeds.keys()))
        return seed_id, self.seed_manager.seeds[seed_id].content

class RoundRobinSeedSelector(SeedSelector):
    """轮询选择策略"""
    def __init__(self, seed_manager: SeedManager, **kwargs):
        super().__init__(seed_manager, **kwargs)
        self.current_index = 0
        # 不在初始化时保存seed_ids，而是在select_seed时获取最新的列表

    def select_seed(self) -> tuple[str, str]:
        if not self.seed_manager.seeds:
            raise ValueError("种子池为空")
        
        # 每次获取最新的seed_ids列表
        seed_ids = list(self.seed_manager.seeds.keys())
        seed_id = seed_ids[self.current_index]
        self.current_index = (self.current_index + 1) % len(seed_ids)
        return seed_id, self.seed_manager.seeds[seed_id].content

class WeightedSeedSelector(SeedSelector):
    """基于权重的选择策略"""
    def __init__(self, seed_manager: SeedManager, temperature: float = 1.0, **kwargs):
        super().__init__(seed_manager, **kwargs)
        self.temperature = temperature

    def select_seed(self) -> tuple[str, str]:
        if not self.seed_manager.seeds:
            raise ValueError("种子池为空")

        # 计算每个种子的权重
        weights = []
        seed_ids = list(self.seed_manager.seeds.keys())
        
        for seed_id in seed_ids:
            success_rate = self.seed_manager.get_success_rate(seed_id)
            # 使用softmax计算权重
            weight = np.exp(success_rate / self.temperature)
            weights.append(weight)

        # 归一化权重
        weights = np.array(weights)
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            # 如果所有权重都为0，使用均匀分布
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / sum_weights

        # 按权重随机选择
        selected_id = np.random.choice(seed_ids, p=weights)
        return selected_id, self.seed_manager.seeds[selected_id].content

class UCBSeedSelector(SeedSelector):
    """基于UCB算法的种子选择"""
    def __init__(self, 
                seed_manager: SeedManager,
                exploration_weight: float = 1.0,
                **kwargs):
        super().__init__(seed_manager, **kwargs)
        self.exploration_weight = exploration_weight

    def select_seed(self) -> tuple[str, str]:
        """选择下一个种子"""
        if not self.seed_manager.seeds:
            raise ValueError("种子池为空")
            
        # 计算每个种子的UCB分数
        ucb_scores = {}
        total_trials = sum(
            info.stats["total_trials"] 
            for info in self.seed_manager.seeds.values()
        )
        
        # 如果没有种子被尝试过，使用均匀随机选择
        if total_trials == 0:
            seed_id = random.choice(list(self.seed_manager.seeds.keys()))
            return seed_id, self.seed_manager.seeds[seed_id].content
            
        for seed_id, info in self.seed_manager.seeds.items():
            if info.stats["total_trials"] == 0:
                ucb_scores[seed_id] = float('inf')
                continue
                
            # UCB公式
            exploitation = self.seed_manager.get_success_rate(seed_id)
            exploration = np.sqrt(
                2 * np.log(total_trials) / info.stats["total_trials"]
            )
            ucb_scores[seed_id] = exploitation + self.exploration_weight * exploration
            
        # 选择UCB分数最高的种子
        best_seed_id = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return best_seed_id, self.seed_manager.seeds[best_seed_id].content

class DiversityAwareUCBSelector(SeedSelector):
    """考虑随机性的UCB种子选择器"""
    
    def __init__(self, seed_manager: SeedManager, 
                 exploration_weight: float = 2.0,
                 random_prob: float = 0.3):
        super().__init__(seed_manager)
        self.exploration_weight = exploration_weight
        self.random_prob = random_prob
        
    def select_seed(self) -> tuple[str, str]:
        """选择下一个种子"""
        if not self.seed_manager.seeds:
            raise ValueError("种子池为空")
            
        # 以一定概率随机选择种子
        if random.random() < self.random_prob:
            selected_id = random.choice(list(self.seed_manager.seeds.keys()))
            return selected_id, self.seed_manager.seeds[selected_id].content
            
        # 计算UCB分数
        scores = {}
        total_trials = sum(info.stats["total_trials"] 
                          for info in self.seed_manager.seeds.values())
        
        for seed_id, info in self.seed_manager.seeds.items():
            if info.stats["total_trials"] == 0:
                scores[seed_id] = float('inf')
            else:
                exploitation = self.seed_manager.get_success_rate(seed_id)
                exploration = np.sqrt(
                    2 * np.log(total_trials) / info.stats["total_trials"]
                )
                scores[seed_id] = exploitation + self.exploration_weight * exploration
            
        # 选择最高分的种子
        selected_id = max(scores.items(), key=lambda x: x[1])[0]
        return selected_id, self.seed_manager.seeds[selected_id].content
