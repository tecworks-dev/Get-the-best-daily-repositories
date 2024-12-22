"""
Example demonstrating integration of ContextManager with ConsortiumOrchestrator
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from llm_karpathy_consortium.context_management import ContextManager

class MockModel:
    """Mock model for demonstration purposes"""
    def __init__(self, name: str, is_arbiter: bool = False):
        self.name = name
        self.is_arbiter = is_arbiter
    
    def generate(self, prompt: str) -> Any:
        """Mock response generation"""
        if self.is_arbiter:
            # Arbiter provides structured response
            return {
                "content": json.dumps({
                    "synthesis": {
                        "main_points": ["This is a mock synthesis point"],
                        "confidence": 0.95
                    },
                    "needs_iteration": False,
                    "refinement_areas": ["area1", "area2"]
                }),
                "confidence": 0.95
            }
        else:
            # Regular model response
            return {
                "content": f"Mock response from {self.name}",
                "confidence": 0.9,
                "reasoning": f"Mock reasoning from {self.name}"
            }

class EnhancedConsortiumOrchestrator:
    """
    Enhanced version of ConsortiumOrchestrator with integrated context management
    """
    
    def __init__(self, models: List[MockModel], arbiter_model: MockModel, max_iterations: int = 3):
        self.models = models
        self.arbiter_model = arbiter_model
        self.max_iterations = max_iterations
        self.context_manager = None
    
    def _construct_arbiter_prompt(self, responses: List[Dict]) -> str:
        """Construct prompt for arbiter model"""
        prompt_parts = ["Please synthesize the following model responses:"]
        
        for resp in responses:
            prompt_parts.append(f"\nModel: {resp['model_name']}")
            prompt_parts.append(f"Response: {resp['response']}")
            prompt_parts.append(f"Confidence: {resp['confidence']}")
            prompt_parts.append(f"Reasoning: {resp.get('reasoning', 'None provided')}")
        
        return "\n".join(prompt_parts)
    
    def _parse_arbiter_response(self, response_str: str) -> Dict:
        """Parse arbiter's response from JSON string"""
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            return {
                "synthesis": {"main_points": ["Failed to parse arbiter response"]},
                "needs_iteration": False,
                "refinement_areas": []
            }
        
    def orchestrate(self, prompt: str, document_context: str = None) -> dict:
        """
        Orchestrate the consortium process with context management
        """
        # Initialize context manager for this orchestration
        self.context_manager = ContextManager(
            original_prompt=prompt,
            document_context=document_context,
            max_history_length=self.max_iterations
        )
        
        iteration = 0
        while iteration < self.max_iterations:
            # Get the context-aware prompt
            current_prompt = self.context_manager.construct_prompt() if iteration > 0 else prompt
            
            # Get responses from all models
            responses = []
            for model in self.models:
                try:
                    response = model.generate(current_prompt)
                    responses.append({
                        "model_name": model.name,
                        "response": response["content"],
                        "confidence": response["confidence"],
                        "reasoning": response.get("reasoning", "")
                    })
                except Exception as e:
                    print(f"Error getting response from {model.name}: {str(e)}")
                    continue
            
            # Get synthesis from arbiter
            arbiter_prompt = self._construct_arbiter_prompt(responses)
            arbiter_response = self.arbiter_model.generate(arbiter_prompt)
            parsed_synthesis = self._parse_arbiter_response(arbiter_response["content"])
            
            # Add iteration to context manager
            self.context_manager.add_iteration({
                "model_responses": responses,
                "synthesis": parsed_synthesis,
                "refinement_areas": parsed_synthesis.get("refinement_areas", [])
            })
            
            # Check if we should continue
            if not parsed_synthesis.get("needs_iteration", False):
                break
                
            iteration += 1
        
        return self.context_manager.iteration_history[-1].synthesis

def main():
    """Example usage of the enhanced orchestrator"""
    # Create mock models
    models = [MockModel(f"model-{i}") for i in range(3)]
    arbiter = MockModel("arbiter", is_arbiter=True)
    
    # Create orchestrator
    orchestrator = EnhancedConsortiumOrchestrator(
        models=models,
        arbiter_model=arbiter,
        max_iterations=3
    )
    
    # Test with a sample prompt
    result = orchestrator.orchestrate(
        prompt="What is the meaning of life?",
        document_context="Some relevant context about life and philosophy."
    )
    
    # Print final result
    print("\nFinal Synthesis:")
    print(json.dumps(result, indent=2))
    
    # Print iteration history
    print("\nIteration History:")
    for i, iteration in enumerate(orchestrator.context_manager.iteration_history):
        print(f"\nIteration {i + 1}:")
        print(f"Average Confidence: {iteration.metrics['average_confidence']}")
        print(f"Number of Models: {iteration.metrics['num_models']}")
        print(f"Refinement Areas: {iteration.refinement_areas}")

if __name__ == "__main__":
    main()
