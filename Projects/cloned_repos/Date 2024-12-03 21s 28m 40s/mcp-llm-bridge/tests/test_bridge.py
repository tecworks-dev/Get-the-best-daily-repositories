# tests/test_bridge.py
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from mcp import StdioServerParameters
from mcp_llm_bridge.config import BridgeConfig, LLMConfig
from mcp_llm_bridge.bridge import MCPLLMBridge, BridgeManager

@pytest.fixture
def mock_mcp_tool():
    mock = MagicMock()
    mock.name = "test_tool"
    mock.description = "A test tool"
    mock.inputSchema = {
        "type": "object",
        "properties": {
            "arg1": {"type": "string"}
        }
    }
    return mock

@pytest.fixture
def mock_config():
    return BridgeConfig(
        mcp_server_params=StdioServerParameters(
            command="uvx",
            args=["mcp-server-sqlite", "--db-path", "test.db"],
            env=None
        ),
        llm_config=LLMConfig(
            api_key="test-key",
            model="gpt-4",
            base_url=None
        ),
        system_prompt="Test system prompt"
    )

@pytest.fixture
def mock_llm_response():
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function = MagicMock()
    tool_call.function.name = "test_tool"
    tool_call.function.arguments = '{"arg1": "test"}'
    
    response = MagicMock()
    response.tool_calls = [tool_call]
    return response

@pytest.mark.asyncio
async def test_bridge_initialization(mock_config, mock_mcp_tool):
    with patch('mcp_llm_bridge.bridge.MCPClient') as MockMCPClient:
        # Setup mocks
        mock_mcp_instance = AsyncMock()
        mock_mcp_instance.get_available_tools.return_value = [mock_mcp_tool]
        MockMCPClient.return_value = mock_mcp_instance

        # Create bridge
        bridge = MCPLLMBridge(mock_config)
        
        # Test initialization
        success = await bridge.initialize()
        assert success == True
        
        # Verify MCP client was initialized
        MockMCPClient.assert_called_once()
        mock_mcp_instance.connect.assert_called_once()
        mock_mcp_instance.get_available_tools.assert_called_once()

@pytest.mark.asyncio
async def test_tool_conversion(mock_config, mock_mcp_tool):
    bridge = MCPLLMBridge(mock_config)
    converted_tools = bridge._convert_mcp_tools_to_openai_format([mock_mcp_tool])
    
    assert len(converted_tools) == 1
    assert converted_tools[0]["type"] == "function"
    assert converted_tools[0]["function"]["name"] == "test_tool"
    assert converted_tools[0]["function"]["description"] == "A test tool"
    assert "parameters" in converted_tools[0]["function"]

@pytest.mark.asyncio
async def test_message_processing(mock_config, mock_llm_response):
    with patch('mcp_llm_bridge.bridge.MCPClient') as MockMCPClient, \
         patch('mcp_llm_bridge.bridge.LLMClient') as MockLLMClient:
        
        # Setup mocks
        mock_mcp_instance = AsyncMock()
        mock_llm_instance = AsyncMock()
        
        # Create a mock response without tool calls
        mock_response = MagicMock()
        mock_response.content = "Final response"
        mock_response.tool_calls = None
        
        mock_llm_instance.invoke_with_prompt.return_value = mock_response
        
        MockMCPClient.return_value = mock_mcp_instance
        MockLLMClient.return_value = mock_llm_instance

        # Create and initialize bridge
        bridge = MCPLLMBridge(mock_config)
        await bridge.initialize()

        # Test message processing
        response = await bridge.process_message("Test message")
        
        # Verify interactions
        mock_llm_instance.invoke_with_prompt.assert_called_once_with("Test message")
        assert response == "Final response"

@pytest.mark.asyncio
async def test_tool_call_handling(mock_config, mock_llm_response):
    with patch('mcp_llm_bridge.bridge.MCPClient') as MockMCPClient, \
         patch('mcp_llm_bridge.bridge.LLMClient') as MockLLMClient:
        
        # Setup mocks
        mock_mcp_instance = AsyncMock()
        mock_mcp_instance.call_tool.return_value = {"result": "tool_result"}
        
        mock_llm_instance = AsyncMock()
        mock_llm_instance.invoke_with_prompt.return_value = mock_llm_response
        mock_llm_instance.invoke.return_value = MagicMock(content="Final response")
        
        MockMCPClient.return_value = mock_mcp_instance
        MockLLMClient.return_value = mock_llm_instance

        # Create and initialize bridge
        bridge = MCPLLMBridge(mock_config)
        bridge.tool_name_mapping = {"test_tool": "test_tool"}
        await bridge.initialize()

        # Test tool call handling
        tool_responses = await bridge._handle_tool_calls(mock_llm_response.tool_calls)
        
        # Verify tool execution
        assert len(tool_responses) == 1
        assert tool_responses[0]["tool_call_id"] == "call_1"
        mock_mcp_instance.call_tool.assert_called_once_with(
            "test_tool", 
            {"arg1": "test"}
        )

@pytest.mark.asyncio
async def test_bridge_manager(mock_config):
    with patch('mcp_llm_bridge.bridge.MCPLLMBridge') as MockBridge:
        # Setup mock
        mock_bridge_instance = AsyncMock()
        mock_bridge_instance.initialize.return_value = True
        MockBridge.return_value = mock_bridge_instance

        # Test context manager
        async with BridgeManager(mock_config) as bridge:
            assert bridge is mock_bridge_instance
            mock_bridge_instance.initialize.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling(mock_config):
    with patch('mcp_llm_bridge.bridge.MCPClient') as MockMCPClient:
        # Setup mock to raise an error
        mock_mcp_instance = AsyncMock()
        mock_mcp_instance.connect.side_effect = Exception("Connection error")
        MockMCPClient.return_value = mock_mcp_instance

        # Create bridge
        bridge = MCPLLMBridge(mock_config)
        
        # Test initialization failure
        success = await bridge.initialize()
        assert success == False

@pytest.mark.asyncio
async def test_tool_name_sanitization(mock_config):
    bridge = MCPLLMBridge(mock_config)
    
    test_cases = [
        ("test-tool", "test_tool"),
        ("test tool", "test_tool"),
        ("Test-Tool", "test_tool"),
        ("test_tool", "test_tool"),
        ("test-tool-123", "test_tool_123"),
    ]
    
    for input_name, expected_output in test_cases:
        assert bridge._sanitize_tool_name(input_name) == expected_output

@pytest.mark.asyncio
async def test_bridge_cleanup(mock_config):
    with patch('mcp_llm_bridge.bridge.MCPClient') as MockMCPClient:
        # Setup mock
        mock_mcp_instance = AsyncMock()
        MockMCPClient.return_value = mock_mcp_instance

        # Create bridge
        bridge = MCPLLMBridge(mock_config)
        
        # Test cleanup
        await bridge.close()
        mock_mcp_instance.__aexit__.assert_called_once()