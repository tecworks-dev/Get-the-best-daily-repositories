什么是 Pythonic 函数调用
-----------------------

by @karminski-牙医


Pythonic function call 是一种新的工具调用方式，它允许 LLM 通过生成 Python 代码来调用函数，而不是传统的 JSON schema 方式。

## 主要优势：

1. 更接近自然语言和人类思维方式
2. 能够在单次对话中处理复杂的多步骤任务
3. 支持更灵活的逻辑控制（如条件判断、循环等）
4. 充分利用了 LLM 在预训练中获得的程序性知识

## 如何使用

以下是一个典型的使用示例：

**首先定义可用的函数：**
```python
def check_availability(day: str, start_time: str, end_time: str) -> bool:
    """检查某个时间段是否可用"""
    pass

def make_appointment(day: str, start_time: str, end_time: str) -> dict:
    """预约会议"""
    pass

def add_to_reminders(reminder_text: str) -> bool:
    """添加提醒"""
    pass
```

**当用户提出请求时，LLM 会生成完整的 Python 代码来解决问题：**
```python
# 计算明天的日期
from datetime import datetime, timedelta
today = datetime.now()
tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")

# 定义时间段
start_time = "10:00"
end_time = "12:00"

# 检查可用性
is_available = check_availability(tomorrow, start_time, end_time)

# 如果时间段可用，则预约会议
appointment_result = (
    make_appointment(
        day=tomorrow,
        start_time=start_time,
        end_time=end_time,
        title="Meeting with Thesis Supervisor"
    )
    if is_available
    else {"appointment_made": False}
)

# 如果预约成功，添加提醒
if appointment_result["appointment_made"]:
    add_to_reminders("Meeting with Thesis Supervisor scheduled for 10:00 AM tomorrow")
```

## 执行过程

代码执行通过 exec-python 包来完成，它会：
1. 解析模型生成的代码
2. 在安全的环境中执行代码
3. 跟踪函数调用和变量状态
4. 返回结构化的执行结果

根据基准测试结果（DPAB-α），使用 Pythonic function call 的模型在复杂任务处理上通常比使用 JSON-based function call 的模型表现更好。例如，Claude 3.5 Sonnet 在 Pythonic 方式下得分为 87，而在 JSON 方式下仅为 45。

| Model Name                          | Pythonic | JSON |
|-------------------------------------|----------|------|
| Closed Models                       |          |      |
| Claude 3.5 Sonnet                  | 87       | 45   |
| o1-preview-2024-09-12              | 55       | 39   |
| o1-mini-2024-09-12                 | 59       | 35   |
| gpt-4o-2024-11-20                   | 60       | 30   |
| Open Models                         |          |      |
| > 100B Parameters                   |          |      |
| DeepSeek V3 (685B)                 | 63       | 33   |
| MiniMax-01                          | 62       | 40   |
| Llama-3.1-405B-Instruct            | 60       | 38   |
| > 30B Parameters                    |          |      |
| Qwen-2.5-Coder-32b-Instruct        | 68       | 32   |
| Qwen-2.5-72b-instruct              | 65       | 39   |
| Llama-3.3-70b-Instruct             | 59       | 40   |
| QwQ-32b-Preview                    | 47       | 21   |
| < 20B Parameters                    |          |      |
| Dria-Agent-a-7B                   | 70       | 38   |
| Qwen2.5-Coder-7B-Instruct          | 44       | 39   |
| Dria-Agent-a-3B                    | 72       | 31   |
| Qwen2.5-Coder-3B-Instruct          | 26       | 37   |
| Qwen-2.5-7B-Instruct               | 47       | 34   |
| Phi-4 (14B)                        | 55       | 35   |


## 参考资料

- [Python Is All You Need? Introducing Dria-Agent-α](https://huggingface.co/blog/andthattoo/dria-agent-a)
- [Dria Pythonic Agent Benchmark (DPAB)](https://huggingface.co/blog/andthattoo/dpab-a)
