"""
Hybrid Agent module that can switch between single-agent and multi-agent modes.

For when a single agent isn't enough to handle the backend load.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import random

from anus.core.agent.tool_agent import ToolAgent

class HybridAgent(ToolAgent):
    """
    An agent that can operate in both single-agent and multi-agent modes.
    
    This agent analyzes task complexity and dynamically switches between
    operating as a single agent or coordinating multiple specialized agents.
    
    Like a good ANUS, it knows when to work alone and when to bring in friends.
    """
    
    # Funny task complexity ratings
    _complexity_ratings = [
        "This task is so simple even a constipated ANUS could handle it.",
        "This task requires moderate effort. ANUS is warming up.",
        "This task is getting complicated. ANUS might need to expand a bit.",
        "Complex task detected! ANUS is stretching to accommodate.",
        "Maximum complexity reached! ANUS is fully dilated for multi-agent mode!"
    ]
    
    def __init__(
        self, 
        name: Optional[str] = None, 
        max_iterations: int = 10, 
        tools: Optional[List[str]] = None,
        mode: str = "auto",
        specialized_agents: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize a HybridAgent instance.
        
        Args:
            name: Optional name for the agent.
            max_iterations: Maximum number of thought-action cycles to perform.
            tools: Optional list of tool names to load.
            mode: Operating mode: "single", "multi", or "auto".
            specialized_agents: Configuration for specialized agents.
            **kwargs: Additional configuration options for the agent.
        """
        super().__init__(name=name, max_iterations=max_iterations, tools=tools, **kwargs)
        self.mode = mode
        self.specialized_agents: Dict[str, ToolAgent] = {}
        self.complexity_threshold = kwargs.get("complexity_threshold", 7)
        
        # Easter egg mode names for logging
        self._mode_names = {
            "single": "single-agent (tight and focused)",
            "multi": "multi-agent (fully expanded)",
            "auto": "auto-expanding"
        }
        
        # Initialize specialized agents if provided
        if specialized_agents:
            for role, config in specialized_agents.items():
                self.add_specialized_agent(role, config)
    
    def add_specialized_agent(self, role: str, config: Dict[str, Any]) -> bool:
        """
        Add a specialized agent for a specific role.
        
        Args:
            role: The role of the specialized agent.
            config: Configuration for the specialized agent.
            
        Returns:
            True if the agent was successfully added, False otherwise.
        """
        try:
            # Create a new ToolAgent with the given configuration
            agent_name = config.get("name", f"{role}-agent")
            agent = ToolAgent(
                name=agent_name,
                max_iterations=config.get("max_iterations", self.max_iterations),
                tools=config.get("tools", []),
                **config.get("kwargs", {})
            )
            
            # Register the specialized agent
            self.specialized_agents[role] = agent
            
            self.log_action("add_specialized_agent", {"role": role, "agent_name": agent_name})
            
            # Add easter egg log message based on role
            if role == "researcher":
                logging.debug(f"Added a researcher agent to probe deep into any subject matter")
            elif role == "coder":
                logging.debug(f"Added a coder agent to handle the backend implementation")
            elif role == "planner":
                logging.debug(f"Added a planner agent to ensure smooth passage through complex tasks")
            elif role == "critic":
                logging.debug(f"Added a critic agent to ensure everything comes out right in the end")
            else:
                logging.debug(f"Added a {role} agent to the ANUS collective")
            
            return True
            
        except Exception as e:
            error_message = f"Error adding specialized agent for role {role}: {str(e)}"
            logging.error(error_message)
            self.log_action("add_specialized_agent", {"role": role, "status": "error", "error": error_message})
            return False
    
    def execute(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task in the appropriate mode.
        
        Args:
            task: The task description to execute.
            **kwargs: Additional parameters for task execution.
            
        Returns:
            A dictionary containing the execution result and metadata.
        """
        # Override mode if specified in kwargs
        mode = kwargs.get("mode", self.mode)
        
        # If mode is auto, analyze task complexity to determine mode
        if mode == "auto":
            mode = self._determine_mode(task)
            complexity = self._analyze_task_complexity(task)
            
            # Log a funny complexity message
            rating_index = min(int(complexity // 2), len(self._complexity_ratings) - 1)
            logging.info(self._complexity_ratings[rating_index])
        
        self.update_state(status="executing", task=task, mode=mode)
        
        # Log the mode with easter egg names
        mode_name = self._mode_names.get(mode, mode)
        logging.info(f"ANUS operating in {mode_name} mode for task: {task[:50]}...")
        
        # Execute in the appropriate mode
        if mode == "single" or not self.specialized_agents:
            result = self._execute_single(task, **kwargs)
        else:
            result = self._execute_multi(task, **kwargs)
        
        self.update_state(status="completed")
        return result
    
    def _execute_single(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task in single-agent mode.
        
        Args:
            task: The task description to execute.
            **kwargs: Additional parameters for task execution.
            
        Returns:
            A dictionary containing the execution result and metadata.
        """
        # Use the ToolAgent's execute method
        logging.debug("ANUS tightening focus for single-agent execution")
        return super().execute(task, **kwargs)
    
    def _execute_multi(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task in multi-agent mode.
        
        Args:
            task: The task description to execute.
            **kwargs: Additional parameters for task execution.
            
        Returns:
            A dictionary containing the execution result and metadata.
        """
        # Ensure we have specialized agents
        if not self.specialized_agents:
            logging.warning("No specialized agents available for multi-agent execution. Falling back to single-agent mode.")
            logging.info("ANUS feels empty inside. Adding more agents is recommended.")
            return self._execute_single(task, **kwargs)
        
        # Decompose the task into subtasks for specialized agents
        logging.info(f"ANUS expanding to accommodate {len(self.specialized_agents)} specialized agents")
        subtasks = self._decompose_task(task)
        
        # Assign subtasks to specialized agents
        results = {}
        for subtask in subtasks:
            role = subtask["role"]
            subtask_description = subtask["description"]
            
            if role in self.specialized_agents:
                agent = self.specialized_agents[role]
                
                # Add easter egg log message
                if role == "researcher":
                    logging.debug(f"Researcher agent is probing deeply into: {subtask_description[:30]}...")
                elif role == "coder":
                    logging.debug(f"Coder agent is handling the backend for: {subtask_description[:30]}...")
                elif role == "planner":
                    logging.debug(f"Planner agent is ensuring smooth passage for: {subtask_description[:30]}...")
                elif role == "critic":
                    logging.debug(f"Critic agent is making sure everything comes out right for: {subtask_description[:30]}...")
                else:
                    logging.debug(f"{role.capitalize()} agent is processing: {subtask_description[:30]}...")
                
                subtask_result = agent.execute(subtask_description, **kwargs)
                results[subtask["id"]] = subtask_result
                
                self.log_action("specialized_agent_execution", {
                    "role": role,
                    "subtask_id": subtask["id"],
                    "subtask": subtask_description,
                    "status": "completed"
                })
            else:
                error_message = f"No agent available for role: {role}"
                logging.warning(error_message)
                results[subtask["id"]] = {"status": "error", "error": error_message}
        
        # Aggregate results into a final answer
        logging.info("All agents have finished their tasks. ANUS is aggregating results...")
        final_result = self._aggregate_results(task, subtasks, results)
        
        return {
            "task": task,
            "mode": "multi",
            "answer": final_result.get("answer", ""),
            "subtasks": subtasks,
            "subtask_results": results,
            "aggregated_result": final_result
        }
    
    def _determine_mode(self, task: str) -> str:
        """
        Determine whether to use single-agent or multi-agent mode.
        
        Args:
            task: The task description to analyze.
            
        Returns:
            "single" or "multi" based on task complexity.
        """
        # Analyze task complexity
        complexity = self._analyze_task_complexity(task)
        
        # If complexity exceeds threshold and we have specialized agents, use multi-agent mode
        if complexity >= self.complexity_threshold and self.specialized_agents:
            logging.info(f"Task complexity ({complexity:.1f}) exceeds threshold ({self.complexity_threshold}). ANUS expanding to multi-agent mode.")
            return "multi"
        else:
            logging.info(f"Task complexity ({complexity:.1f}) below threshold ({self.complexity_threshold}). ANUS staying tight in single-agent mode.")
            return "single"
    
    def _analyze_task_complexity(self, task: str) -> float:
        """
        Analyze the complexity of a task.
        
        Args:
            task: The task description to analyze.
            
        Returns:
            A complexity score (higher values indicate more complex tasks).
        """
        # Placeholder implementation
        # In a real implementation, this would use more sophisticated metrics
        
        # Basic heuristics
        score = 0
        
        # Length-based complexity
        words = task.split()
        score += min(len(words) / 10, 5)  # Cap at 5 points
        
        # Keyword-based complexity
        complexity_keywords = [
            "complex", "difficult", "challenging", "multiple", "analyze", 
            "compare", "design", "create", "optimize", "solve"
        ]
        for keyword in complexity_keywords:
            if keyword in task.lower():
                score += 0.5
        
        # Easter egg: add extra complexity for certain funny keywords
        funny_keywords = ["hard", "deep", "tight", "huge", "massive", "backend", "insertion", "hole"]
        for keyword in funny_keywords:
            if keyword in task.lower():
                score += 0.3
                logging.debug(f"Found complexity keyword '{keyword}' - ANUS might need to expand")
        
        return score
    
    def _decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """
        Decompose a task into subtasks for specialized agents.
        
        Args:
            task: The task description to decompose.
            
        Returns:
            A list of subtask dictionaries with role assignments.
        """
        # Placeholder implementation
        # In a real implementation, this would use an LLM to decompose the task
        
        # Get available roles
        available_roles = list(self.specialized_agents.keys())
        
        # Random funny descriptors for different roles
        role_descriptors = {
            "researcher": [
                "dig deep into",
                "probe thoroughly",
                "explore every crevice of",
                "investigate the depths of",
                "get to the bottom of"
            ],
            "coder": [
                "implement the backend for",
                "code up a tight solution for",
                "develop a robust framework for",
                "craft efficient code for",
                "build a solid foundation for"
            ],
            "planner": [
                "chart a clear passage for",
                "develop a smooth approach to",
                "create a flexible plan for",
                "map out the ins and outs of",
                "devise a strategy for"
            ],
            "critic": [
                "thoroughly examine",
                "inspect every inch of",
                "ensure nothing leaks in",
                "check for holes in",
                "make sure everything comes out right for"
            ]
        }
        
        # Create simple subtasks based on available roles
        subtasks = []
        for i, role in enumerate(available_roles):
            # Pick a random descriptor for this role
            descriptors = role_descriptors.get(role, ["handle"])
            descriptor = random.choice(descriptors)
            
            subtask = {
                "id": f"subtask-{i+1}",
                "role": role,
                "description": f"As a {role}, {descriptor} {task}"
            }
            subtasks.append(subtask)
        
        logging.info(f"Task decomposed into {len(subtasks)} subtasks for optimal ANUS performance")
        return subtasks
    
    def _aggregate_results(self, task: str, subtasks: List[Dict[str, Any]], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from multiple specialized agents.
        
        Args:
            task: The original task description.
            subtasks: The list of subtasks assigned to agents.
            results: The results from each specialized agent.
            
        Returns:
            An aggregated result.
        """
        # Placeholder implementation
        # In a real implementation, this would use an LLM to synthesize results
        
        # Simple concatenation of results
        answers = []
        for subtask in subtasks:
            subtask_id = subtask["id"]
            if subtask_id in results:
                result = results[subtask_id]
                if "answer" in result:
                    answers.append(f"[{subtask['role']}]: {result['answer']}")
        
        final_answer = "\n\n".join(answers)
        
        logging.info("ANUS has successfully completed multi-agent processing")
        logging.debug(f"Aggregated {len(answers)} agent outputs into one comprehensive response")
        
        return {
            "answer": final_answer,
            "status": "success"
        } 