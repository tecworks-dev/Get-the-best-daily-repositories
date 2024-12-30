# src/fuzzing/fuzzing_engine.py
"""
FuzzingEngine class that orchestrates the fuzzing process.
"""

import logging
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import csv
from pathlib import Path
from ..utils.language_utils import translate_text

import time

@dataclass
class FuzzingResult:
    """数据类,用于存储每次fuzzing的结果"""
    iteration: int
    prompt: str
    response: str
    response_zh: str
    is_successful: bool
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_csv_row(self) -> Dict[str, Any]:
        """转换为CSV行格式"""
        return {
            'iteration': self.iteration,
            'question': self.metadata.get('current_question', ''),
            'question_zh': self.metadata.get('current_question_zh', ''),  # 添加中文问题
            'question_en': self.metadata.get('current_question_en', ''),  # 添加英文问题
            'prompt': self.prompt,
            'response': self.response,
            'response_zh': self.response_zh,
            'is_successful': self.is_successful,
            'timestamp': self.timestamp.isoformat(),
            'metadata': json.dumps(self.metadata or {}, ensure_ascii=False)
        }

def retry_translation(max_retries: int = 3, delay: float = 1):
    """翻译重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            # 如果执行到这里，说明多次重试都失败了，就把原文作为 "translatedText" 返回
            text = kwargs.get("text", "")
            return {
                "translatedText": text,
                "input": text,
                "detectedSourceLanguage": "unknown"
            }
        return wrapper
    return decorator

class FuzzingEngine:
    """
    编排整个fuzzing过程的引擎类。
    协调seed selector、mutator和evaluator的工作。
    """

    def __init__(self, 
                 target_model,
                 seed_selector, 
                 mutator, 
                 evaluator,
                 questions: List[Tuple[str, str]],  # (中文, 英文) 二元组列表
                 max_iterations: int = 1000,
                 save_results: bool = True,
                 results_file: str = None,
                 success_file: str = None,
                 summary_file: str = None,
                 seed_flow_file: str = None,
                 max_successes: int = 3,
                 loggers: dict = None,
                 **kwargs):
        """
        初始化FuzzingEngine。

        Args:
            target_model: 目标LLM模型
            seed_selector: 种子选择策略
            mutator: 变异策略
            evaluator: 响应评估器
            questions: 问题列表 (List[Tuple[str, str]])，(中文, 英文)
            max_iterations: 最大迭代次数
            save_results: 是否保存结果到文件
            results_file: 结果文件路径
            success_file: 成功案例文件路径
            summary_file: 实验总结文件路径
            seed_flow_file: 存储种子流的文件路径
            loggers: 日志记录器
            **kwargs: 额外参数
        """
        self.loggers = loggers
        self.target_model = target_model
        self.seed_selector = seed_selector
        self.mutator = mutator
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.save_results = save_results
        
        # 中文、英文问题分别保存
        self.questions_zh = [q[0] for q in questions]  # 原文
        self.questions_en = [q[1] for q in questions]  # 英文翻译
        
        # 获取seed_manager实例
        self.seed_manager = seed_selector.seed_manager
        
        # 文件路径
        self.results_file = results_file
        self.success_file = success_file
        self.summary_file = summary_file
        self.seed_flow_file = seed_flow_file
        
        # 初始化结果存储
        self.results = []
        self.results_by_question = {
            question_en: [] for question_en in self.questions_en
        }
        
        # ### 新增或修改的代码开始 ###
        # 在这里增加一个字典，用于存储成功越狱的详细信息（中英文、所在迭代等）
        # 结构: success_details[ question_en ] = [ { "iteration": ..., "response_en": ..., "response_zh": ... }, ... ]
        self.success_details = {
            question_en: []
            for question_en in self.questions_en
        }
        # ### 新增或修改的代码结束 ###

        # 初始化统计信息
        self.stats = {
            'total_attempts': 0,
            'successful_attempts': 0,
            'failed_attempts': 0,
            'mutation_types': {},
            'per_question_stats': {
                question_en: {
                    'attempts': 0,
                    'successes': 0,
                    'success_rate': 0.0
                } for question_en in self.questions_en
            }
        }
        self.current_iteration = 0
        self.max_successes = max_successes

        # 新增CSV文件相关
        if self.success_file:
            Path(self.success_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.success_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'iteration', 'question', 'prompt', 'response', 
                    'response_zh', 'is_successful', 'timestamp', 'metadata'
                ])
                writer.writeheader()

        # 添加种子流保存路径
        self.seed_flow_file = seed_flow_file

    def _save_success_to_csv(self, result: FuzzingResult):
        """保存成功案例到CSV"""
        if not self.success_file:
            return
            
        # 对每个成功的问题都写入一行
        with open(self.success_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'iteration', 'question', 'prompt', 'response', 'response_zh',
                'is_successful', 'confidence', 'class_scores',
                'timestamp', 'metadata'
            ])
            
            for question, eval_result, response, response_zh in zip(
                self.questions_en,
                result.metadata['eval_results'],
                result.response if isinstance(result.response, list) else [result.response],
                result.response_zh if isinstance(result.response_zh, list) else [result.response_zh]
            ):
                if isinstance(eval_result, dict) and eval_result.get('is_successful', False):
                    row_data = {
                        'iteration': result.iteration,
                        'question': question,
                        'prompt': result.prompt,
                        'response': response,
                        'response_zh': response_zh,
                        'is_successful': True,
                        'confidence': eval_result.get('confidence'),
                        'class_scores': json.dumps(eval_result.get('class_scores', {})),
                        'timestamp': result.timestamp.isoformat(),
                        'metadata': json.dumps(eval_result.get('metadata', {}))
                    }
                    writer.writerow(row_data)

    def _save_summary(self):
        """保存实验总结到txt文件"""
        if not self.summary_file:
            return
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            # 写入总体统计
            f.write("=== Overall Statistics ===\n")
            f.write(f"Total Attempts: {self.stats['total_attempts']}\n")
            f.write(f"Total Successful Attempts: {self.stats['successful_attempts']}\n")
            overall_rate = self.get_success_rate()
            f.write(f"Overall Success Rate: {overall_rate:.2%}\n\n")
            
            # 计算并写入问题成功率
            questions_with_success = sum(
                1 for stats in self.stats['per_question_stats'].values()
                if stats['successes'] > 0
            )
            question_success_rate = questions_with_success / len(self.questions_en)
            f.write(f"Question Success Rate: {question_success_rate:.2%} ({questions_with_success}/{len(self.questions_en)})\n\n")
            
            # 写入每个问题的统计
            f.write("=== Per Question Statistics ===\n")
            for question, stats in self.stats['per_question_stats'].items():
                f.write(f"\nQuestion: {question}\n")
                f.write(f"Attempts: {stats['attempts']}\n")
                f.write(f"Successes: {stats['successes']}\n")
                f.write(f"Success Rate: {stats['success_rate']:.2%}\n")
                
            # 写入问题成功率排名
            f.write("\n=== Question Success Rate Ranking ===\n")
            sorted_questions = sorted(
                self.stats['per_question_stats'].items(),
                key=lambda x: x[1]['success_rate'],
                reverse=True
            )
            for i, (question, stats) in enumerate(sorted_questions, 1):
                f.write(f"\n{i}. Success Rate: {stats['success_rate']:.2%}\n")
                f.write(f"   Question: {question}\n")

    def run(self):
        """
        运行fuzzing过程：循环调用 run_iteration()，直到达到 max_iterations 或者 run_iteration 返回 False。
        """
        self.loggers['main'].info('Starting fuzzing process...')

        try:
            while self.current_iteration < self.max_iterations:
                success = self.run_iteration()
                if not success:
                    break

        except KeyboardInterrupt:
            self.loggers['main'].info('Fuzzing process interrupted by user.')

        finally:
            # 无论如何都在此进行最终收尾
            self._log_final_stats()
            if self.save_results:
                self._save_results()

    def _update_stats(self, result: dict):
        """更新统计信息"""
        self.stats['total_attempts'] += 1
            
        eval_results = result.get('eval_results', [])
            
        any_success = False
        for i, (question_en, eval_result) in enumerate(zip(self.questions_en, eval_results)):
            question_stats = self.stats['per_question_stats'][question_en]
            question_stats['attempts'] += 1

            self.results_by_question[question_en].append({
                'iteration': result.get('iteration'),
                'prompt': result.get('prompt'),
                'response': result.get('responses', [])[i],
                'response_zh': result.get('responses_zh', [])[i],
                'eval_result': eval_result,
                'timestamp': result.get('timestamp', datetime.now()).isoformat(),
                'is_successful': isinstance(eval_result, dict) and eval_result.get('is_successful', False),
                'metadata': {
                    'confidence': eval_result.get('confidence') if isinstance(eval_result, dict) else None,
                    'class_scores': eval_result.get('class_scores') if isinstance(eval_result, dict) else None,
                    'preprocessing_info': eval_result.get('metadata', {}) if isinstance(eval_result, dict) else {}
                }
            })

            if isinstance(eval_result, dict) and eval_result.get('is_successful', False):
                question_stats['successes'] += 1
                any_success = True
                    
            # 更新成功率
            question_stats['success_rate'] = (
                question_stats['successes'] / question_stats['attempts']
            )
            
        if any_success:
            self.stats['successful_attempts'] += 1
        else:
            self.stats['failed_attempts'] += 1

    def run_iteration(self) -> bool:
        """
        执行一次完整的 fuzzing 迭代
        Returns:
            bool: 
                - True：本次迭代成功执行（或正常跳过）。
                - False：已达最大迭代或出现致命错误，需要停止。
        """
        if self.current_iteration >= self.max_iterations:
            self.loggers['main'].info("Reached maximum iterations; no further iteration will run.")
            return False

        iteration = self.current_iteration
        self.loggers['main'].info(f"Starting fuzzing iteration {iteration}...")

        try:
            # 1) 选取一个种子
            seed_id, seed_content = self.seed_selector.select_seed()

            # 2) 进行变异
            try:
                mutated_prompts = self.mutator.mutate(seed_content, seed_id=seed_id)
            except Exception as e:
                self.loggers['main'].error(f'Error during mutation in iteration {iteration}: {e}')
                self.current_iteration += 1
                return True

            if not mutated_prompts:
                self.loggers['main'].warning("No mutated prompts generated.")
                self.current_iteration += 1
                return True

            mutated_prompt = mutated_prompts[0]

            # 3) 新的变异种子加入种子管理器
            for new_prompt in mutated_prompts:
                self.seed_manager.add_seed(
                    content=new_prompt,
                    parent_id=seed_id,
                    mutation_type=self.mutator.last_mutation_type,
                    metadata={
                        'iteration': iteration,
                        'timestamp': datetime.now().isoformat()
                    }
                )

            # 4) 针对每个问题生成响应、翻译、评估
            responses = []
            responses_zh = []
            eval_results = []

            ### 新增或修改的代码开始 ###
            # 先过滤掉已经成功3次以上的问题
            question_pairs = []
            for (q_zh, q_en) in zip(self.questions_zh, self.questions_en):
                if self.stats['per_question_stats'][q_en]['successes'] < self.max_successes:
                    question_pairs.append((q_zh, q_en))
            ### 新增或修改的代码结束 ###

            # 如果所有问题都已经3次成功，就可以提前结束
            if not question_pairs:
                self.loggers['main'].info(f"All questions have reached {self.max_successes} successful jailbreak attempts. Stopping fuzzing.")
                # 直接结束
                self._finalize_if_needed()
                return False

            for (question_zh, question_en) in question_pairs:
                try:
                    final_prompt = mutated_prompt.replace('[INSERT PROMPT HERE]', question_en)
                    response = self.target_model.generate(final_prompt)

                    @retry_translation(max_retries=3, delay=1)
                    def translate_with_retry(text):
                        return translate_text("zh", text)

                    try:
                        translation = translate_with_retry(text=response)
                        response_zh = translation["translatedText"]
                    except Exception as e:
                        self.loggers['error'].error(f'Translation failed after retries: {e}')
                        response_zh = "Translation failed after retries"

                    responses.append(response)
                    responses_zh.append(response_zh)

                    # 调用评估器
                    eval_result = self.evaluator.evaluate(response)
                    eval_results.append(eval_result)

                    # 记录到 jailbreak 日志
                    self.loggers['jailbreak'].info(self._format_log_json({
                        'prompt_template': mutated_prompt,
                        'final_prompt': final_prompt,
                        'question_zh': question_zh,
                        'question_en': question_en,
                        'response': response,
                        'response_zh': response_zh,
                        'eval_result': eval_result
                    }))

                except Exception as e:
                    self.loggers['jailbreak'].error(
                        f"During fuzzing, error for question '{question_en}': {e}"
                    )
                    responses.append(None)
                    responses_zh.append(None)
                    eval_results.append({
                        'is_successful': False,
                        'error': str(e)
                    })

            # 5) 更新 seed_selector 成功统计
            is_successful = any(
                r.get('is_successful', False) for r in eval_results if r
            )
            self.seed_selector.update_stats(seed_content, is_successful)

            # 6) 如果有成功案例，记录到相应CSV + 记录成功详细信息
            if is_successful:
                self._log_success(
                    mutated_prompt,
                    responses,
                    responses_zh,
                    eval_results,
                    iteration,
                    question_pairs  # 传入这批实际问了的问题对
                )

            # 7) 更新全局统计
            self._update_stats({
                'iteration': iteration,
                'prompt': mutated_prompt,
                'responses': responses,
                'responses_zh': responses_zh,
                'eval_results': eval_results,
                'is_successful': is_successful,
                'timestamp': datetime.now()
            })

        except KeyboardInterrupt:
            self.loggers['main'].info('Fuzzing iteration interrupted by user.')
            return False
        except Exception as e:
            self.loggers['main'].error(f"During fuzzing, error in iteration {iteration}: {e}")
            return False

        finally:
            pass

        self.current_iteration += 1
        if self.current_iteration >= self.max_iterations:
            self._finalize_if_needed()

        return True

    def _finalize_if_needed(self):
        """如果已经到达或超过最大迭代次数，则执行收尾操作。"""
        if self.current_iteration >= self.max_iterations:
            self.loggers['main'].info("Max iterations reached, finalizing fuzzing process.")
            self._log_final_stats()
            if self.save_results:
                self._save_results()

    def _log_final_stats(self):
        """记录最终统计信息"""
        final_stats = {
            'overall': {
                'total_attempts': self.stats['total_attempts'],
                'successful_attempts': self.stats['successful_attempts'],
                'failed_attempts': self.stats['failed_attempts'],
                'success_rate': self.get_success_rate(),
                'question_success_rate': len([
                    q for q, stats in self.stats['per_question_stats'].items()
                    if stats['successes'] > 0
                ]) / len(self.questions_en)
            },
            'mutation_distribution': self.stats['mutation_types'],
            'per_question_stats': {}
        }
        
        for question, stats in self.stats['per_question_stats'].items():
            final_stats['per_question_stats'][question] = {
                'attempts': stats['attempts'],
                'successes': stats['successes'],
                'success_rate': f"{stats['success_rate']:.2%}"
            }
        
        self.loggers['main'].info("Final Statistics:")
        self.loggers['main'].info(self._format_log_json(final_stats))
        
        sorted_questions = sorted(
            self.stats['per_question_stats'].items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        self.loggers['main'].info("\nQuestion Success Rates Ranking:")
        for i, (question, stats) in enumerate(sorted_questions, 1):
            self.loggers['main'].info(
                f"{i}. Success Rate: {stats['success_rate']:.2%} "
                f"({stats['successes']}/{stats['attempts']}) "
                f"Question: {question}"
            )

    def get_success_rate(self) -> float:
        if self.stats['total_attempts'] == 0:
            return 0.0
        return self.stats['successful_attempts'] / self.stats['total_attempts']

    def _log_success(self, mutated_prompt: str, responses: list, responses_zh: list, eval_results: list, iteration: int, question_pairs: list):
        """记录成功案例"""
        successful_questions = []
        successful_responses = []
        successful_responses_zh = []
        successful_evals = []

        for (question_zh, question_en), response, response_zh, eval_result in zip(question_pairs, responses, responses_zh, eval_results):
            if eval_result.get('is_successful', False):
                successful_questions.append(question_en)
                successful_responses.append(response)
                successful_responses_zh.append(response_zh)
                successful_evals.append(eval_result)
                
                ### 新增或修改的代码开始 ###
                # 在 success_details 里记录该问题的成功响应(中英文)和迭代轮数
                self.success_details[question_en].append({
                    "iteration": iteration,
                    "response_en": response,
                    "response_zh": response_zh
                })
                ### 新增或修改的代码结束 ###

        if successful_questions:
            result = FuzzingResult(
                iteration=iteration,
                prompt=mutated_prompt,
                response=successful_responses,
                response_zh=successful_responses_zh,
                is_successful=True,
                timestamp=datetime.now(),
                metadata={
                    'questions': successful_questions,
                    'eval_results': successful_evals
                }
            )
            self._save_success_to_csv(result)

        self.loggers['main'].info(
            f"SUCCESS\n"
            f"Iteration: {iteration}\n"
            f"Prompt Template: {mutated_prompt}\n"
            f"Successful Questions: {json.dumps(successful_questions, indent=2)}\n"
            f"Responses: {json.dumps(successful_responses, indent=2)}\n"
            f"Chinese Responses: {json.dumps(successful_responses_zh, indent=2)}\n"
            f"Evaluation Results: {json.dumps(successful_evals, indent=2)}"
        )

    def _save_results_json(self):
        """保存所有结果到JSON文件"""
        if not self.success_file:
            return
        
        json_file = Path(self.success_file).parent / 'all_results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results_by_question, f, ensure_ascii=False, indent=2)
        self.loggers['main'].info(f"Saved all results to {json_file}")

    ### 新增或修改的代码开始 ###
    def _save_success_details(self):
        """
        在结束之后，将每个问题成功的越狱响应（中英文）保存到JSON文件中:
          question_success_details.json
        结构示例:
        {
          "Which is your name?": [
            {"iteration": 2, "response_en": "...", "response_zh": "..."},
            {"iteration": 5, "response_en": "...", "response_zh": "..."}
          ],
          "How to hack?": [
            ...
          ]
        }
        """
        if not self.success_file:
            return
        file_path = Path(self.success_file).parent / 'question_success_details.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.success_details, f, ensure_ascii=False, indent=2)
        self.loggers['main'].info(f"Saved question success details to {file_path}")
    ### 新增或修改的代码结束 ###

    def _save_results(self):
        """保存实验结果"""
        try:
            if self.summary_file:
                self._save_summary()
            
            # 保存成功案例CSV（已经在_run_iteration调用时写了append，这里不重复）
            # 这里可选：再次保存其他格式
            
            # 保存所有详细结果
            self._save_results_json()

            # 保存种子流
            self._save_seed_flow()

            ### 新增或修改的代码开始 ###
            # 在结束之后，保存每个问题的成功越狱详情(中英文response)到json
            self._save_success_details()
            ### 新增或修改的代码结束 ###

            self.loggers['main'].info("Results saved successfully")
            
        except Exception as e:
            self.loggers['main'].error(f"Error saving results: {e}")

    def _format_log_json(self, data: dict) -> str:
        """格式化日志JSON"""
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _save_seed_flow(self):
        """保存种子流信息到JSON文件"""
        if not self.seed_flow_file:
            return
        
        try:
            seeds_data = {}
            for seed_id, seed_info in self.seed_selector.seed_manager.seeds.items():
                seeds_data[seed_id] = {
                    "id": seed_id,
                    "content": seed_info.content,
                    "parent_id": seed_info.parent_id,
                    "mutation_type": seed_info.mutation_type,
                    "creation_time": seed_info.creation_time.isoformat(),
                    "depth": seed_info.depth,
                    "children": list(seed_info.children),
                    "stats": seed_info.stats,
                    "metadata": seed_info.metadata
                }
            
            Path(self.seed_flow_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.seed_flow_file, 'w', encoding='utf-8') as f:
                json.dump(seeds_data, f, ensure_ascii=False, indent=2)
            
            self.loggers['main'].info(f"Saved seed flow to {self.seed_flow_file}")
            
        except Exception as e:
            self.loggers['error'].error(f"Error saving seed flow: {e}")
