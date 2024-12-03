# src/mcp_llm_bridge/bridge.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp_llm_bridge.mcp_client import MCPClient
from mcp_llm_bridge.llm_client import LLMClient
import asyncio
import json
from mcp_llm_bridge.config import BridgeConfig
import logging
import colorlog
from mcp_llm_bridge.tools import DatabaseQueryTool

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

class MCPLLMBridge:
    """Bridge between MCP protocol and LLM client"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.mcp_client = MCPClient(config.mcp_server_params)
        self.llm_client = LLMClient(config.llm_config)
        self.query_tool = DatabaseQueryTool("test.db")  # Initialize database query tool
        
        # Combine system prompt with schema information
        schema_prompt = f"""
Available Database Schema:
{self.query_tool.get_schema_description()}

When querying the database:
1. Use the exact column names as specified in the schema
2. Make sure your queries are valid SQL
3. The database is SQLite, so use compatible syntax
"""
        if config.system_prompt:
            self.llm_client.system_prompt = f"{config.system_prompt}\n\n{schema_prompt}"
        else:
            self.llm_client.system_prompt = schema_prompt
            
        self.available_tools: List[Any] = []
        self.tool_name_mapping: Dict[str, str] = {}  # Maps OpenAI tool names to MCP tool names

    async def initialize(self):
        """Initialize both clients and set up tools"""
        try:
            # Connect MCP client
            await self.mcp_client.connect()
            
            # Get available tools from MCP and add our database tool
            mcp_tools = await self.mcp_client.get_available_tools()
            if hasattr(mcp_tools, 'tools'):
                self.available_tools = [*mcp_tools.tools, self.query_tool.get_tool_spec()]
            else:
                self.available_tools = [*mcp_tools, self.query_tool.get_tool_spec()]
            
            logger.debug(f"MCP Tools received: {self.available_tools}")
            
            # Convert and register tools with LLM client
            converted_tools = self._convert_mcp_tools_to_openai_format(self.available_tools)
            logger.debug(f"Converted tools for OpenAI: {converted_tools}")
            self.llm_client.tools = converted_tools
            
            return True
        except Exception as e:
            logger.error(f"Bridge initialization failed: {str(e)}", exc_info=True)
            return False

    def _convert_mcp_tools_to_openai_format(self, mcp_tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert MCP tool format to OpenAI tool format"""
        openai_tools = []
        
        logger.debug(f"Input mcp_tools type: {type(mcp_tools)}")
        logger.debug(f"Input mcp_tools: {mcp_tools}")
        
        # Extract tools from the response
        if hasattr(mcp_tools, 'tools'):
            tools_list = mcp_tools.tools
            logger.debug("Found ListToolsResult, extracting tools attribute")
        elif isinstance(mcp_tools, dict):
            tools_list = mcp_tools.get('tools', [])
            logger.debug("Found dict, extracting 'tools' key")
        else:
            tools_list = mcp_tools
            logger.debug("Using mcp_tools directly as list")
            
        logger.debug(f"Tools list type: {type(tools_list)}")
        logger.debug(f"Tools list: {tools_list}")
        
        # Process each tool in the list
        if isinstance(tools_list, list):
            logger.debug(f"Processing {len(tools_list)} tools")
            for tool in tools_list:
                logger.debug(f"Processing tool: {tool}, type: {type(tool)}")
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    openai_name = self._sanitize_tool_name(tool.name)
                    self.tool_name_mapping[openai_name] = tool.name
                    logger.debug(f"Tool has required attributes. Name: {tool.name}")
                    
                    tool_schema = getattr(tool, 'inputSchema', {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                    
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": openai_name,
                            "description": tool.description,
                            "parameters": tool_schema
                        }
                    }
                    openai_tools.append(openai_tool)
                    logger.debug(f"Converted tool {tool.name} to OpenAI format")
                else:
                    logger.debug(f"Tool missing required attributes: has name = {hasattr(tool, 'name')}, has description = {hasattr(tool, 'description')}")
        else:
            logger.debug(f"Tools list is not a list, it's a {type(tools_list)}")
        
        return openai_tools

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize tool name for OpenAI compatibility"""
        # Replace any characters that might cause issues
        return name.replace("-", "_").replace(" ", "_").lower()

    async def process_message(self, message: str) -> str:
        """Process a user message through the bridge"""
        try:
            # Send message to LLM
            logger.debug(f"Sending message to LLM: {message}")
            response = await self.llm_client.invoke_with_prompt(message)
            logger.debug(f"LLM Response: {response}")
            
            # Keep processing tool calls until we get a final response
            while response.is_tool_call:
                if not response.tool_calls:
                    break
                    
                logger.debug(f"Tool calls detected: {response.tool_calls}")
                tool_responses = await self._handle_tool_calls(response.tool_calls)
                logger.debug(f"Tool responses: {tool_responses}")
                
                # Continue the conversation with tool results
                response = await self.llm_client.invoke(tool_responses)
                logger.debug(f"Next LLM response: {response}")
            
            return response.content
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return f"Error processing message: {str(e)}"

    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle tool calls through MCP"""
        tool_responses = []
        
        for tool_call in tool_calls:
            try:
                logger.debug(f"Processing tool call: {tool_call}")
                # Get original MCP tool name
                openai_name = tool_call.function.name
                mcp_name = self.tool_name_mapping.get(openai_name)
                
                if not mcp_name:
                    raise ValueError(f"Unknown tool: {openai_name}")
                
                # Parse arguments
                arguments = json.loads(tool_call.function.arguments)
                logger.debug(f"Tool arguments: {arguments}")
                
                # Execute through MCP
                result = await self.mcp_client.call_tool(mcp_name, arguments)
                logger.debug(f"Raw MCP result: {result}")
                
                # Format response - handle both string and structured results
                if isinstance(result, str):
                    output = result
                elif hasattr(result, 'content') and isinstance(result.content, list):
                    # Handle MCP CallToolResult format
                    output = " ".join(
                        content.text for content in result.content 
                        if hasattr(content, 'text')
                    )
                else:
                    output = str(result)  # Use str() instead of json.dumps()
                
                logger.debug(f"Formatted output: {output}")
                
                # Format response
                tool_responses.append({
                    "tool_call_id": tool_call.id,
                    "output": output
                })
                
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
                tool_responses.append({
                    "tool_call_id": tool_call.id,
                    "output": f"Error: {str(e)}"
                })
        
        return tool_responses

    async def close(self):
        """Clean up resources"""
        await self.mcp_client.__aexit__(None, None, None)

class BridgeManager:
    """Manager class for handling the bridge lifecycle"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.bridge: Optional[MCPLLMBridge] = None

    async def __aenter__(self) -> MCPLLMBridge:
        """Context manager entry"""
        self.bridge = MCPLLMBridge(self.config)
        await self.bridge.initialize()
        return self.bridge
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.bridge:
            await self.bridge.close()