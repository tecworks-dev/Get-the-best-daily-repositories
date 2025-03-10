"""
Tool Agent module that extends the react agent with tool execution capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
import importlib
import logging

from anus.core.agent.react_agent import ReactAgent

class ToolAgent(ReactAgent):
    """
    An agent that can use tools to interact with its environment.
    
    Extends the ReactAgent with the ability to discover, load, and execute tools.
    """
    
    def __init__(
        self, 
        name: Optional[str] = None, 
        max_iterations: int = 10, 
        tools: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize a ToolAgent instance.
        
        Args:
            name: Optional name for the agent.
            max_iterations: Maximum number of thought-action cycles to perform.
            tools: Optional list of tool names to load.
            **kwargs: Additional configuration options for the agent.
        """
        super().__init__(name=name, max_iterations=max_iterations, **kwargs)
        self.tools: Dict[str, Any] = {}
        
        # Load specified tools or default tools
        if tools:
            for tool_name in tools:
                self.load_tool(tool_name)
    
    def load_tool(self, tool_name: str) -> bool:
        """
        Load a tool by name.
        
        Args:
            tool_name: The name of the tool to load.
            
        Returns:
            True if the tool was successfully loaded, False otherwise.
        """
        try:
            # Dynamically import the tool module
            module_path = f"anus.tools.{tool_name}"
            module = importlib.import_module(module_path)
            
            # Get the tool class (assumed to be the same name as the module but capitalized)
            class_name = "".join(word.capitalize() for word in tool_name.split("_")) + "Tool"
            tool_class = getattr(module, class_name)
            
            # Instantiate the tool
            tool_instance = tool_class()
            
            # Register the tool
            self.tools[tool_name] = tool_instance
            
            self.log_action("load_tool", {"tool_name": tool_name, "status": "success"})
            return True
            
        except (ImportError, AttributeError, Exception) as e:
            self.log_action("load_tool", {"tool_name": tool_name, "status": "error", "error": str(e)})
            logging.error(f"Failed to load tool {tool_name}: {e}")
            return False
    
    def _execute_action(self, action_name: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action using the appropriate tool.
        
        Args:
            action_name: The name of the action/tool to execute.
            action_input: The input parameters for the action.
            
        Returns:
            The observation from executing the action.
        """
        # Check if the action corresponds to a loaded tool
        if action_name in self.tools:
            try:
                tool = self.tools[action_name]
                result = tool.execute(**action_input)
                return {"status": "success", "result": result}
            except Exception as e:
                error_message = f"Error executing tool {action_name}: {str(e)}"
                logging.error(error_message)
                return {"status": "error", "error": error_message}
        else:
            # Try to load the tool if it's not already loaded
            if self.load_tool(action_name):
                # Retry execution with the newly loaded tool
                return self._execute_action(action_name, action_input)
            else:
                return {"status": "error", "error": f"Unknown action or tool: {action_name}"}
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools and their descriptions.
        
        Returns:
            A list of dictionaries containing tool information.
        """
        tool_info = []
        for name, tool in self.tools.items():
            info = {
                "name": name,
                "description": getattr(tool, "description", "No description available"),
                "parameters": getattr(tool, "parameters", {})
            }
            tool_info.append(info)
        return tool_info 