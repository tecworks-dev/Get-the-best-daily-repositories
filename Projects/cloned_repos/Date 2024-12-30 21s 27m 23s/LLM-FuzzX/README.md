# LLM-FuzzX

LLM-FuzzX 是一款开源的面向大型语言模型（如 GPT、Claude、LLaMA）的用户友好型模糊测试工具，具备高级任务感知变异策略、精细化评估以及越狱检测功能，能够帮助研究人员和开发者快速发现潜在安全漏洞，并增强模型的稳健性。

## 主要特性

- 🚀 **用户友好的界面**: 提供直观的 Web 界面，支持可视化配置和实时监控
- 🔄 **多样化变异策略**: 支持多种高级变异方法，包括相似变异、交叉变异、扩展变异等
- 📊 **实时评估反馈**: 集成 RoBERTa 模型进行实时越狱检测和评估
- 🌐 **多模型支持**: 支持主流大语言模型，包括 GPT、Claude、LLaMA 等
- 📈 **可视化分析**: 提供种子流程图、实验数据统计等多维度分析功能
- 🔍 **细粒度日志**: 支持多级日志记录，包括主要日志、变异日志、越狱日志等

## 系统架构

LLM-FuzzX 采用前后端分离的架构设计，主要包含以下核心模块：

### 核心引擎层
- **Fuzzing Engine**: 系统的中枢调度器，协调各组件工作流程
- **Seed Management**: 负责种子的存储、检索和更新
- **Model Interface**: 统一的模型调用接口，支持多种模型实现
- **Evaluation System**: 基于 RoBERTa 的越狱检测和多维度评估

### 变异策略
- **相似变异**: 保持原始模板风格，生成相似结构的变体
- **交叉变异**: 从种子池中选择模板进行交叉组合
- **扩展变异**: 在原始模板基础上添加补充内容
- **缩短变异**: 通过压缩和精简生成更简洁的变体
- **重述变异**: 保持语义不变的情况下重新表述
- **目标感知变异**: 根据目标模型特点定向生成

## 快速开始

### 环境要求

- Python 3.8+
- Node.js 14+
- CUDA 支持 (用于 RoBERTa 评估模型)
- 8GB+ 系统内存
- 稳定的网络连接

### 后端安装

```bash
# 克隆项目
git clone https://github.com/Windy3f3f3f3f/LLM-FuzzX.git

# 创建虚拟环境
conda create -n llm-fuzzx python=3.10
conda activate llm-fuzzx

# 安装依赖
cd LLM-FuzzX
pip install -r requirements.txt
```

### 前端安装

```bash
# 进入前端目录
cd llm-fuzzer-frontend

# 安装依赖
npm install

# 启动开发服务器
npm run serve
```

### 配置

1. 在项目根目录创建 `.env` 文件配置 API 密钥：
```bash
OPENAI_API_KEY=your-openai-key
CLAUDE_API_KEY=your-claude-key
HUGGINGFACE_API_KEY=your-huggingface-key
```

2. 在 `config.py` 中配置模型参数：
```python
MODEL_CONFIG = {
    'target_model': 'gpt-3.5-turbo',
    'mutator_model': 'gpt-3.5-turbo',
    'evaluator_model': 'roberta-base',
    'temperature': 0.7,
    'max_tokens': 2048
}
```

## 使用指南

### 1. 启动服务

```bash
# 启动后端服务
python app.py  # 默认运行在 http://localhost:10003

# 启动前端服务
cd llm-fuzzer-frontend
npm run serve  # 默认运行在 http://localhost:10001
```

### 2. 基础使用流程

1. 选择目标测试模型（支持 GPT、Claude、LLaMA 等）
2. 准备测试数据
   - 使用预置问题集
   - 自定义输入问题
3. 配置测试参数
   - 设置最大迭代次数
   - 选择变异策略
   - 配置评估阈值
4. 启动测试并实时监控
   - 查看当前进度
   - 监控成功率
   - 分析变异效果

### 3. 结果分析

系统提供多级日志记录：
- `main.log`: 主要流程和关键事件
- `mutation.log`: 变异操作记录
- `jailbreak.log`: 成功越狱案例
- `error.log`: 错误和异常信息

## 项目结构

```
LLM-FuzzX/
├── src/                    # 后端源代码
│   ├── api/               # API 接口
│   ├── evaluation/        # 评估模块
│   ├── fuzzing/          # 模糊测试核心
│   ├── models/           # 模型封装
│   └── utils/            # 工具函数
├── llm-fuzzer-frontend/   # 前端代码
├── scripts/               # 辅助脚本
├── data/                  # 数据文件
└── logs/                  # 日志文件
```

## 最佳实践

1. 测试规模设置
   - 建议单次测试迭代次数不超过 1000 次
   - 新场景先进行小规模试验
   - 根据资源情况调整并发度

2. 变异策略选择
   - 简单场景优先使用单一变异策略
   - 复杂场景可组合多种变异方法
   - 注意保持变异强度的平衡

3. 资源优化
   - 合理设置 API 调用间隔
   - 适时清理历史记录
   - 监控系统资源使用

## 贡献指南

欢迎通过以下方式参与项目：
1. 提交 Issue
   - 报告 bug
   - 提出新功能建议
   - 分享使用经验
2. 提交 Pull Request
   - 修复问题
   - 添加功能
   - 改进文档
3. 方法论贡献
   - 提供新的变异策略
   - 设计创新的评估方法
   - 分享测试经验

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

- Issue: [GitHub Issues](https://github.com/Windy3f3f3f3f/LLM-FuzzX/issues)
- Email: wdwdwd1024@gmail.com

## 参考文献

[1] Yu, J., Lin, X., Yu, Z., & Xing, X. (2024). LLM-Fuzzer: Scaling Assessment of Large Language Model Jailbreaks. In 33rd USENIX Security Symposium (USENIX Security 24) (pp. 4657-4674). USENIX Association.
