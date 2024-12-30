# src/models/huggingface_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.llm_wrapper import LLMWrapper

class HuggingFaceModel(LLMWrapper):
    def __init__(self, model_name: str, device: str = None, precision: str = 'float16', **kwargs):
        # 在这里进行model_name检查
        self.validate_model_name("huggingface", model_name)
        self.model_name = model_name
        self.device = device
        self.gen_kwargs = {}
        if device == None:
            raise ValueError("device is required")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, precision),
            trust_remote_code=True,
            **kwargs
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str:
        """
        生成文本的方法
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            **kwargs: 其他生成参数
        
        Returns:
            str: 生成的文本
        """
        # 更新生成参数
        self.gen_kwargs.update({
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            **kwargs
        })
        
        # 处理输入
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # 生成文本
        outputs = self.model.generate(
            **inputs,
            **self.gen_kwargs
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
