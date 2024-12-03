# src/mcp_llm_bridge/llm_client.py
from typing import Dict, List, Any, Optional
import openai
from mcp_llm_bridge.config import LLMConfig
import logging
import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s%(reset)s:     %(cyan)s%(name)s%(reset)s - %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class LLMResponse:
    """Standardized response format focusing on tool handling"""
    def __init__(self, completion: Any):
        self.completion = completion
        self.choice = completion.choices[0]
        self.message = self.choice.message
        self.stop_reason = self.choice.finish_reason
        self.is_tool_call = self.stop_reason == "tool_calls"
        
        # Format content for bridge compatibility
        self.content = self.message.content if self.message.content is not None else ""
        self.tool_calls = self.message.tool_calls if hasattr(self.message, "tool_calls") else None
        
        # Debug logging
        logger.debug(f"Raw completion: {completion}")
        logger.debug(f"Message content: {self.content}")
        logger.debug(f"Tool calls: {self.tool_calls}")
        
    def get_message(self) -> Dict[str, Any]:
        """Get standardized message format"""
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": self.tool_calls
        }

class LLMClient:
    """Client for interacting with OpenAI-compatible LLMs"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.tools = []
        self.messages = []
        self.system_prompt = None
    
    def _prepare_messages(self) -> List[Dict[str, Any]]:
        """Prepare messages for API call"""
        formatted_messages = []
        
        if self.system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": self.system_prompt
            })
            
        formatted_messages.extend(self.messages)
        return formatted_messages
    
    async def invoke_with_prompt(self, prompt: str) -> LLMResponse:
        """Send a single prompt to the LLM"""
        self.messages.append({
            "role": "user",
            "content": prompt
        })
        
        return await self.invoke([])
    
    async def invoke(self, tool_results: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        """Invoke the LLM with optional tool results"""
        if tool_results:
            for result in tool_results:
                self.messages.append({
                    "role": "tool",
                    "content": str(result.get("output", "")),  # Convert to string and provide default
                    "tool_call_id": result["tool_call_id"]
                })
        
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=self._prepare_messages(),
            tools=self.tools if self.tools else None,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        response = LLMResponse(completion)
        self.messages.append(response.get_message())
        
        return response