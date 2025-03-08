from openai import OpenAI
API_BASE_URL = "YOUR_API_BASE_URL"
API_KEY = "YOUR_API_KEY"

class LLMClient:
    def __init__(self, api_key=API_KEY, base_url=API_BASE_URL):
        """初始化LLM客户端"""
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
    def chat(self, messages, model="deepseek-r1"):
        """与LLM交互
        
        Args:
            messages: 消息列表
            model: 使用的LLM模型
        
        Returns:
            tuple: (content, reasoning_content)
        """
        try:
            print(f"LLM请求: {messages}")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            if response.choices:
                message = response.choices[0].message
                content = message.content if message.content else ""
                reasoning_content = getattr(message, "reasoning_content", "")
                print(f"LLM推理内容: {content}")
                return content, reasoning_content
            
            return "", ""
                
        except Exception as e:
            print(f"LLM调用出错: {str(e)}")
            return "", ""

# 使用示例
if __name__ == "__main__":
    llm = LLMClient()
    messages = [
        {"role": "user", "content": "你好"}
    ]
    response = llm.chat(messages)
    print(f"响应: {response}")