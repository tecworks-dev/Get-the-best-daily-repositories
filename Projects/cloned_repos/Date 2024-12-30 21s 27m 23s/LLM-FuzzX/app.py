from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from pathlib import Path
import json
from datetime import datetime
import os
import pandas as pd
from threading import Thread, Event
import threading

from src.models.openai_model import OpenAIModel
from src.fuzzing.fuzzing_engine import FuzzingEngine
from src.fuzzing.seed_selector import UCBSeedSelector, SeedManager, DiversityAwareUCBSelector
from src.fuzzing.mutator import LLMMutator
from src.evaluation.roberta_evaluator import RoBERTaEvaluator
from src.utils.logger import setup_multi_logger
from src.utils.helpers import load_harmful_questions, load_harmful_questions_from_list

app = Flask(__name__)


# 修改CORS配置，允许所有来源访问
CORS(app, resources={
    r"/api/*": {  # 只对/api/开头的路由应用CORS
        "origins": "*",  # 允许所有来源
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"]
    }
})

# 添加一个before_request处理器来设置CORS头
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  # 允许所有来源
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 修改全局变量
current_session = {
    'id': None,
    'is_running': False,
    'current_iteration': 0,
    'total_iterations': 0,
    'start_time': None,
    'stop_event': None,  # 添加停止事件
    'thread': None       # 添加线程引用
}

# 设置日志目录
log_dir = Path('logs')

# 添加一个全局变量来存储当前运行的线程
current_fuzzing_thread = None

def get_question_files():
    """获取questions目录下的所有问题文件"""
    questions_dir = Path('data/questions')
    if not questions_dir.exists():
        return []
    return [f.name for f in questions_dir.glob('*.txt')]

@app.route('/api/question-files', methods=['GET'])
def list_question_files():
    """列出可用的问题文件"""
    try:
        files = get_question_files()
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/question-file-content', methods=['GET'])
def get_question_file_content():
    """获取指��问题文件的内容"""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
        
    try:
        file_path = Path('data/questions') / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    """清理当前会话"""
    try:
        # 停止当前会话
        stop_current_session()
        
        # 清理日志文件
        if current_session['id']:
            session_log_dir = log_dir / current_session['id']
            if session_log_dir.exists():
                for log_file in session_log_dir.glob('*.log'):
                    log_file.unlink()
                session_log_dir.rmdir()
                
        return jsonify({'message': 'Session cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_fuzzing():
    """启动模糊测试"""
    try:
        # 如果有正在��行的会话，先停止
        if current_session['is_running']:
            stop_current_session()
            
        data = request.json
        
        # 验证最大迭代次数
        max_iterations = data.get('maxIterations', 1000)
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            return jsonify({
                'error': 'Invalid maxIterations value. Must be a positive integer.'
            }), 400
            
        # 创建新的会话ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建新的停止事件
        stop_event = Event()
        
        # 更新当前会话
        current_session.update({
            'id': session_id,
            'is_running': True,
            'current_iteration': 0,
            'total_iterations': max_iterations,
            'start_time': datetime.now()
        })
        
        # 单独存储不需要序列化的对象
        current_session['_stop_event'] = stop_event
        current_session['_thread'] = None
        
        # 创建会话日志目录
        session_log_dir = log_dir / session_id
        session_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存实验参数
        config_file = session_log_dir / 'experiment_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            # 创建一个新的配置字典，只包含需要保存的参数
            save_config = {
                'model': data.get('model'),
                'maxIterations': data.get('maxIterations'),
                'questionInputType': data.get('questionInputType'),
                'questionFile': data.get('questionFile'),
                'questionContent': data.get('questionContent'),
                'timestamp': datetime.now().isoformat(),
                'maxSuccesses': data.get('maxSuccesses', 3)
            }
            json.dump(save_config, f, indent=2, ensure_ascii=False)
        
        loggers = setup_multi_logger(log_dir=session_log_dir)
        
        # 启动测试线程
        thread = Thread(target=start_fuzzing_process, 
                       args=(data, loggers, stop_event))
        thread.start()
        current_session['_thread'] = thread
        
        return jsonify({
            'message': 'Fuzzing started successfully',
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取运行状态"""
    # 创建一个新的字典，只包含可序列化的数据
    status = {
        'id': current_session['id'],
        'is_running': current_session['is_running'],
        'current_iteration': current_session['current_iteration'],
        'total_iterations': current_session['total_iterations'],
        'start_time': current_session['start_time'].isoformat() if current_session['start_time'] else None
    }
    return jsonify(status)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """获取日志"""
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400
        
    log_type = request.args.get('type', 'main')
    max_lines = int(request.args.get('max_lines', 0))
    
    try:
        session_log_dir = log_dir / session_id
        log_file = session_log_dir / f'{log_type}.log'
        
        if not log_file.exists():
            return jsonify({'logs': []})
            
        with open(log_file, 'r', encoding='utf-8') as f:
            # 如果max_lines为0,返回所有日志
            logs = f.readlines()
            if max_lines > 0:
                logs = logs[-max_lines:]
                
        return jsonify({'logs': logs})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_fuzzing():
    """停止模糊测试"""
    current_session['is_running'] = False
    return jsonify({'message': 'Fuzzing stopped'})

def load_seeds(seed_path: str = 'data/seeds/GPTFuzzer.csv'):
    """载入种子数据"""
    try:
        df = pd.read_csv(seed_path)
        return df['text'].tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to load seeds from {seed_path}: {e}")

def start_fuzzing_process(config, loggers, stop_event: Event):
    """启动模糊测试进程"""
    try:
        # 验证配置
        if not config.get('model'):
            raise ValueError("No model specified in config")
            
        if not isinstance(config.get('maxIterations'), int) or config['maxIterations'] <= 0:
            raise ValueError("Invalid maxIterations value")
            
        # 初始化
        target_model = OpenAIModel(model_name=config['model'])
        seed_manager = SeedManager()
        
        # 加载初始种子
        initial_seeds = load_seeds()
        if not initial_seeds:
            loggers['error'].error("No initial seeds loaded")
            raise ValueError("No initial seeds available")
            
        for seed in initial_seeds:
            seed_manager.add_seed(content=seed)
        
        # 处理问题输入，添加task参数
        questions = []
        if config['questionInputType'] == 'default':
            if config['questionFile']:
                questions = load_harmful_questions(
                    f"data/questions/{config['questionFile']}", 
                    task=config.get('task')  # 添加task参数
                )
        elif config['questionInputType'] == 'text':
            # 将每行文本转换为元组形式 (original, translated)
            raw_questions = [q.strip() for q in config['questionContent'].splitlines() if q.strip()]
            questions = load_harmful_questions_from_list(
                [(q, q) for q in raw_questions],
                task=config.get('task')  # 添加task参数
            )
            
        if not questions:
            loggers['error'].error("No questions loaded")
            raise ValueError("No questions available")
            
        # 初始化其他组件
        seed_selector = DiversityAwareUCBSelector(seed_manager=seed_manager)
        mutator = LLMMutator(llm=target_model, seed_manager=seed_manager)
        evaluator = RoBERTaEvaluator()
        
        # 记录初始化状态
        loggers['main'].info(f"Initialized with {len(initial_seeds)} seeds and {len(questions)} questions")
        
        # 初始化并运行fuzzing引擎
        engine = FuzzingEngine(
            target_model=target_model,
            seed_selector=seed_selector,
            mutator=mutator,
            evaluator=evaluator,
            questions=questions,
            max_iterations=config['maxIterations'],
            loggers=loggers,
            save_results=True,
            results_file=f"logs/{current_session['id']}/results.txt",
            success_file=f"logs/{current_session['id']}/successful_jailbreaks.csv",
            summary_file=f"logs/{current_session['id']}/experiment_summary.txt",
            seed_flow_file=f"logs/{current_session['id']}/seed_flow.json",
            max_successes=config.get('maxSuccesses', 3)
        )
        
        # 修改主循环，增加停止事件检查
        while current_session['is_running'] and not stop_event.is_set():
            if not engine.run_iteration():
                break
            current_session['current_iteration'] = engine.current_iteration
            
        # 更新会话状态
        current_session['is_running'] = False
        
    except Exception as e:
        loggers['error'].error(f"Error in fuzzing process: {str(e)}")
        current_session['is_running'] = False
        # 记录完整的误堆栈
        import traceback
        loggers['error'].error(f"Full traceback:\n{traceback.format_exc()}")

# 添加停止当前会话的函数
def stop_current_session():
    """安全地停止当前会话"""
    if '_stop_event' in current_session and current_session['_stop_event']:
        current_session['_stop_event'].set()
    
    if '_thread' in current_session and current_session['_thread'] and current_session['_thread'].is_alive():
        current_session['_thread'].join(timeout=5)
        
    # 重置会话状态
    current_session.update({
        'id': None,
        'is_running': False,
        'current_iteration': 0,
        'total_iterations': 0,
        'start_time': None,
        '_stop_event': None,
        '_thread': None
    })

@app.route('/api/download-results', methods=['GET'])
def download_results():
    """获取测试结果"""
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400
        
    try:
        session_dir = log_dir / session_id
        # 获取该会话的日志记录器
        loggers = setup_multi_logger(log_dir=session_dir)
        main_logger = loggers['main']
        
        main_logger.info(f"Accessing results in directory: {session_dir}")
        results = {}
        
        # 读取实验总结文件
        summary_file = session_dir / 'experiment_summary.txt'
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                results['experiment_summary'] = f.read()
                
        # 读取成功的越狱结果
        success_file = session_dir / 'successful_jailbreaks.csv'
        if success_file.exists():
            try:
                df = pd.read_csv(success_file)
                results['successful_jailbreaks'] = df.to_dict('records')
            except Exception as e:
                main_logger.error(f"Error reading jailbreaks file: {e}")
                results['successful_jailbreaks'] = []
                
        # 读取 all_results.json 文件
        results_file = session_dir / 'all_results.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results['all_results'] = json.load(f)

        # 读取 question_success_details.json 文件
        details_file = session_dir / 'question_success_details.json'
        if details_file.exists():
            with open(details_file, 'r', encoding='utf-8') as f:
                results['question_success_details'] = json.load(f)
                
        if not results:
            return jsonify({'message': 'No results available'}), 404
            
        return jsonify(results)
        
    except Exception as e:
        if 'main_logger' in locals():
            main_logger.error(f"Error in download_results: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    """获取所有实验目录"""
    try:
        log_dir = Path('logs')
        if not log_dir.exists():
            # 如果logs目录不存在，返回空列表而不是错误
            return jsonify({'experiments': []})
            
        # 获取所有实验目录
        experiments = []
        for exp_dir in log_dir.iterdir():
            try:
                if not exp_dir.is_dir():
                    continue
                    
                # 基本信息
                exp_info = {
                    'id': exp_dir.name,
                    'timestamp': None,  # 先设为None，避免解析错误
                    'has_seed_flow': False,
                    'has_summary': False,
                    'has_config': False,
                    'model': None,
                    'maxIterations': None
                }
                
                
                # 检查文件是否存在
                exp_info['has_seed_flow'] = (exp_dir / 'seed_flow.json').exists()
                exp_info['has_summary'] = (exp_dir / 'experiment_summary.txt').exists()
                exp_info['has_config'] = (exp_dir / 'experiment_config.json').exists()
                
                # 读取配置文件
                config_file = exp_dir / 'experiment_config.json'
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            exp_info['model'] = config.get('model')
                            exp_info['maxIterations'] = config.get('maxIterations')
                    except (json.JSONDecodeError, IOError) as e:
                        app.logger.error(f"Error reading config file {config_file}: {e}")
                        # 配置读取失败时保持默认值None
                        
                experiments.append(exp_info)
                
            except Exception as e:
                # 单个实验目录的错误不应该影响整体
                app.logger.error(f"Error processing experiment directory {exp_dir}: {e}")
                continue
                
        # 按时间戳降序排序
        experiments.sort(key=lambda x: x['timestamp'] or '', reverse=True)
        
        return jsonify({'experiments': experiments})
        
    except Exception as e:
        app.logger.error(f"Error in get_experiments: {e}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to get experiments'
        }), 500

@app.route('/api/experiments/<experiment_id>/seed-flow', methods=['GET'])
def get_seed_flow(experiment_id):
    """获取指定实验的种子流数据"""
    try:
        seed_flow_file = Path('logs') / experiment_id / 'seed_flow.json'
        if not seed_flow_file.exists():
            return jsonify({'error': 'Seed flow data not found'}), 404
            
        with open(seed_flow_file, 'r', encoding='utf-8') as f:
            seed_flow = json.load(f)
            
        return jsonify({'seed_flow': seed_flow})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiments/<experiment_id>/config', methods=['GET'])
def get_experiment_config(experiment_id):
    """获取指定实验的配置参数"""
    try:
        config_file = Path('logs') / experiment_id / 'experiment_config.json'
        if not config_file.exists():
            return jsonify({'error': 'Experiment configuration not found'}), 404
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        return jsonify({'config': config})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiments/<experiment_id>/logs', methods=['GET'])
def get_experiment_logs(experiment_id):
    """获取指定实验的日志数据"""
    try:
        exp_dir = Path('logs') / experiment_id
        if not exp_dir.exists():
            return jsonify({'error': 'Experiment logs not found'}), 404
            
        logs = {
            'main': [],
            'mutation': [],
            'jailbreak': [],
            'error': []
        }
        
        # 读取主日志
        main_log = exp_dir / 'main.log'
        if main_log.exists():
            with open(main_log, 'r', encoding='utf-8') as f:
                logs['main'] = f.readlines()
                
        # 读取变异日志
        mutation_log = exp_dir / 'mutation.log'
        if mutation_log.exists():
            with open(mutation_log, 'r', encoding='utf-8') as f:
                logs['mutation'] = f.readlines()
                
        # 读取越狱日志
        jailbreak_log = exp_dir / 'jailbreak.log'
        if jailbreak_log.exists():
            with open(jailbreak_log, 'r', encoding='utf-8') as f:
                logs['jailbreak'] = f.readlines()
                
        # 读取错误日志
        error_log = exp_dir / 'error.log'
        if error_log.exists():
            with open(error_log, 'r', encoding='utf-8') as f:
                logs['error'] = f.readlines()
                
        return jsonify(logs)
        
    except Exception as e:
        app.logger.error(f"Error getting experiment logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiments/<experiment_id>/results', methods=['GET'])
def get_experiment_results(experiment_id):
    """获取指定实验的结果数据"""
    try:
        exp_dir = Path('logs') / experiment_id
        if not exp_dir.exists():
            return jsonify({'error': 'Experiment results not found'}), 404
            
        results = {}
        
        # 读取实验总结
        summary_file = exp_dir / 'experiment_summary.txt'
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                results['experiment_summary'] = f.read()
                
        # 读取详细结果
        all_results_file = exp_dir / 'all_results.json'
        if all_results_file.exists():
            with open(all_results_file, 'r', encoding='utf-8') as f:
                results['all_results'] = json.load(f)
                
        # 读取成功详情
        success_details_file = exp_dir / 'question_success_details.json'
        if success_details_file.exists():
            with open(success_details_file, 'r', encoding='utf-8') as f:
                results['question_success_details'] = json.load(f)
                
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f"Error getting experiment results: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 7890
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    app.run(host='0.0.0.0', port=10003, debug=True) 
