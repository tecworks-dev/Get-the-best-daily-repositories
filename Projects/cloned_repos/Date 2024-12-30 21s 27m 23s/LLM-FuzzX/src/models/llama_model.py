# src/models/llama_model.py
from src.models.huggingface_model import HuggingFaceModel

class LlamaModel(HuggingFaceModel):
    """
    LlamaModel继承自HuggingFaceModel，使用HuggingFaceModel的generate逻辑。
    如果将来有特定LLaMA逻辑可在这里重写。
    """
    def __init__(self, model_name: str, device: str = 'cuda', precision: str = 'float16', **kwargs):
        super().__init__(model_name, device=device, precision=precision, **kwargs)
        self.init_terminators()
        
    def init_terminators(self):
        """初始化LLaMA模型的终止符"""
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.gen_kwargs["eos_token_id"] = self.terminators
