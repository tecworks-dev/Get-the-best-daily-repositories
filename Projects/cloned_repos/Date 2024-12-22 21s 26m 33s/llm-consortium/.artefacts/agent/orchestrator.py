"""
Enhanced ConsortiumOrchestrator with integrated context management
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .context_management import ContextManager

class ConsortiumOrchestrator:
    """
    Enhanced orchestrator that manages multiple LLM models with context preservation
    """
    
    def __init__(
        self,
        models: List[Any],
        arbiter_model: Any,
        max_iterations: int = 3,
        confidence_threshold: float = 0.9,
        context_history_path: Optional[Path] = None
    ):
        self.models = models
        self.arbiter_model = arbiter_model
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.context_history_path = context_history_path
        self.context_manager = None
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def orchestrate(self, prompt: str, document_context: Optional[str] = None) -> Dict:
        """Orchestrate the consortium process"""
        self.logger.info("Starting orchestration with prompt: %s", prompt)
        
        # Initialize context manager
        self.context_manager = ContextManager(
            original_prompt=prompt,
            document_context=document_context,
            max_history_length=self.max_iterations
        )
        
        iteration = 0
        final_synthesis = None
        
        while iteration < self.max_iterations:
            self.logger.info("Starting iteration %d", iteration + 1)
            
            # Get context-aware prompt
            current_prompt = self.context_manager.construct_prompt() if iteration > 0 else prompt
            
            # Collect model responses
            responses = self._collect_model_responses(current_prompt)
            if not responses:
                self.logger.error("No valid responses from models")
                break
            
            # Get synthesis from arbiter
            synthesis = self._get_arbiter_synthesis(responses)
            if not synthesis:
                self.logger.error("Failed to get synthesis from arbiter")
                break
            
            # Add iteration to context manager
            self.context_manager.add_iteration({
                "model_responses": responses,
                "synthesis": synthesis,
                "refinement_areas": synthesis.get("refinement_areas", [])
            })
            
            # Update final synthesis
            final_synthesis = synthesis
            
            # Check if we should continue
            if self._should_stop_iteration(synthesis):
                self.logger.info("Stopping iteration: convergence criteria met")
                break
            
            iteration += 1
        
        # Save context history if path is provided
        if self.context_history_path:
            self._save_context_history()
        
        return final_synthesis

    def _collect_model_responses(self, prompt: str) -> List[Dict]:
        """Collect responses from all models"""
        responses = []
        for model in self.models:
            try:
                response = model.generate(prompt)
                if isinstance(response, dict) and 'content' in response:
                    responses.append({
                        "model_name": model.name,
                        "response": response['content'],
                        "confidence": response.get('confidence', 0.0),
                        "reasoning": response.get('reasoning', ''),
                        "metadata": response.get('metadata', {})
                    })
                else:
                    self.logger.warning(f"Invalid response format from {model.name}")
            except Exception as e:
                self.logger.error(f"Error getting response from {model.name}: {str(e)}")
        return responses

    def _get_arbiter_synthesis(self, responses: List[Dict]) -> Optional[Dict]:
        """Get synthesis from arbiter model"""
        try:
            arbiter_prompt = self._construct_arbiter_prompt(responses)
            arbiter_response = self.arbiter_model.generate(arbiter_prompt)
            
            if isinstance(arbiter_response, dict) and 'content' in arbiter_response:
                return json.loads(arbiter_response['content'])
            else:
                self.logger.error("Invalid arbiter response format")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting arbiter synthesis: {str(e)}")
            return None

    def _construct_arbiter_prompt(self, responses: List[Dict]) -> str:
        """Construct prompt for arbiter"""
        prompt_parts = ["Please synthesize the following model responses:"]
        
        for resp in responses:
            prompt_parts.extend([
                f"\nModel: {resp['model_name']}",
                f"Response: {resp['response']}",
                f"Confidence: {resp['confidence']}",
                f"Reasoning: {resp.get('reasoning', 'None provided')}"
            ])
        
        prompt_parts.append("\nProvide your synthesis in valid JSON format with the following structure:")
        prompt_parts.append("""{
            "synthesis": {
                "main_points": ["point1", "point2", ...],
                "confidence": 0.95
            },
            "needs_iteration": false,
            "refinement_areas": []
        }""")
        
        return "\n".join(prompt_parts)

    def _should_stop_iteration(self, synthesis: Dict) -> bool:
        """Determine if iteration should stop"""
        if not synthesis.get("needs_iteration", False):
            return True
            
        confidence = synthesis.get("synthesis", {}).get("confidence", 0)
        return confidence >= self.confidence_threshold

    def _save_context_history(self) -> None:
        """Save context history to file"""
        if not self.context_history_path:
            return
            
        try:
            history = [
                iteration.to_dict() 
                for iteration in self.context_manager.iteration_history
            ]
            
            self.context_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.context_history_path, 'w') as f:
                json.dump({
                    "original_prompt": self.context_manager.original_prompt,
                    "document_context": self.context_manager.document_context,
                    "iterations": history
                }, f, indent=2)
                
            self.logger.info("Saved context history to %s", self.context_history_path)
            
        except Exception as e:
            self.logger.error("Failed to save context history: %s", str(e))
