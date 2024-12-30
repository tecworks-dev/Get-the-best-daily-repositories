# src/models/claude_model.py
from src.models.openai_model import OpenAIModel
from openai import OpenAI
from config import CLAUDE_API_KEY, BASE_URL
class ClaudeModel(OpenAIModel):
    """
    Claude API模型的包装类，使用OpenAI风格的API
    由于使用了代理，所以可以直接继承OpenAIModel的实现
    """
    def __init__(self, model_name: str, device: str = None, precision: str = None):
        """
        初始化Claude模型
        
        Args:
            model_name: Claude模型名称
            device: 不使用
            precision: 不使用
        """
        # 在调用父类初始化之前先验证模型名称
        self.validate_model_name("claude", model_name)
        
        # 调用父类初始化，但跳过OpenAI的模型验证
        self.model_name = model_name
        self.gen_kwargs = {"model": model_name}
        
        self.client = OpenAI(
            api_key=CLAUDE_API_KEY,
            base_url=BASE_URL
        )
