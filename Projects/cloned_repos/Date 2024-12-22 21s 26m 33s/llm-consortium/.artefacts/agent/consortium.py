import asyncio
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, field
import json
import csv
from datetime import datetime
import os
from pathlib import Path

import llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConsortiumResponse:
    """Represents a response from a model in the consortium.
    
    Attributes:
        model_name: Name of the model that generated the response
        response: The actual response text
        confidence: Confidence score between 0 and 1
        metadata: Additional metadata about the response
    """
    model_name: str
    response: str
    confidence: float
    metadata: Dict[Any, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the response data after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if not self.model_name:
            raise ValueError("Model name cannot be empty")

class ConsortiumConfig:
    """Configuration class for the Consortium plugin."""
    def __init__(
        self,
        history_file: Union[str, Path] = "consortium_history.csv",
        default_arbiter: str = "claude-3-sonnet-20240307",
        default_confidence_threshold: float = 0.9,
        default_max_iterations: int = 2
    ):
        self.history_file = Path(history_file)
        self.default_arbiter = default_arbiter
        self.default_confidence_threshold = default_confidence_threshold
        self.default_max_iterations = default_max_iterations

class ConsortiumPlugin(llm.Plugin):
    """Plugin for managing a consortium of language models with arbitration.
    
    This plugin allows running prompts through multiple models and using an arbiter
    model to select or synthesize the best response.
    """
    
    name = 'consortium'

    def __init__(self, config: Optional[ConsortiumConfig] = None):
        """Initialize the consortium plugin.
        
        Args:
            config: Optional configuration object. If not provided, default config is used.
        """
        self.config = config or ConsortiumConfig()
        self._ensure_history_file()

    def _ensure_history_file(self) -> None:
        """Ensure the history file exists with headers."""
        if not self.config.history_file.exists():
            self.config.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'prompt', 'models', 'arbiter',
                    'iterations', 'final_response', 'confidence'
                ])

    def log_interaction(
        self,
        prompt: str,
        models: List[str],
        arbiter: str,
        iterations: int,
        response: str,
        confidence: float
    ) -> None:
        """Log consortium interaction to CSV file.
        
        Args:
            prompt: The original prompt
            models: List of models used
            arbiter: Name of the arbiter model
            iterations: Number of iterations performed
            response: Final response text
            confidence: Final confidence score
        """
        try:
            with open(self.config.history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    prompt,
                    ','.join(models),
                    arbiter,
                    iterations,
                    response,
                    confidence
                ])
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    async def get_model_response(self, model_name: str, prompt: str) -> ConsortiumResponse:
        """Get response from a specific model.
        
        Args:
            model_name: Name of the model to use
            prompt: The prompt to send to the model
            
        Returns:
            ConsortiumResponse object containing the model's response
        """
        try:
            model = llm.get_model(model_name)
            response = await model.complete_async(prompt)
            
            confidence = self._extract_confidence(str(response))
                
            return ConsortiumResponse(
                model_name=model_name,
                response=str(response),
                confidence=confidence,
                metadata={'raw_response': response}
            )
        except Exception as e:
            logger.error(f"Error getting response from {model_name}: {e}")
            return ConsortiumResponse(
                model_name=model_name,
                response=f"Error: {str(e)}",
                confidence=0.0,
                metadata={'error': str(e)}
            )

    def _extract_confidence(self, response_text: str) -> float:
        """Extract confidence score from response text.
        
        Args:
            response_text: The response text to parse
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if '<confidence>' in response_text and '</confidence>' in response_text:
                confidence_str = response_text.split('<confidence>')[1].split('</confidence>')[0]
                confidence = float(confidence_str)
                return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
        except Exception as e:
            logger.warning(f"Failed to extract confidence: {e}")
        return 0.0

    async def get_arbiter_decision(
        self,
        arbiter_model: str,
        responses: List[ConsortiumResponse],
        original_prompt: str
    ) -> ConsortiumResponse:
        """Get arbitration decision from the arbiter model.
        
        Args:
            arbiter_model: Name of the arbiter model to use
            responses: List of responses to arbitrate
            original_prompt: The original prompt
            
        Returns:
            ConsortiumResponse containing the arbiter's decision
        """
        arbiter = llm.get_model(arbiter_model)
        
        arbitration_prompt = f"""
As an arbiter, analyze these model responses and choose the best one or synthesize a better response.
Original prompt: {original_prompt}

Responses:
{json.dumps([{
    'model': r.model_name,
    'response': r.response,
    'confidence': r.confidence
} for r in responses], indent=2)}

Provide your decision in this format:
<confidence>0.95</confidence>
<response>
Your chosen or synthesized response here
</response>

Explanation of your decision:
"""
        
        response = await arbiter.complete_async(arbitration_prompt)
        confidence = self._extract_confidence(str(response))
        
        return ConsortiumResponse(
            model_name=arbiter_model,
            response=str(response),
            confidence=confidence or 0.95,  # Default high confidence for arbiter
            metadata={'raw_response': response}
        )

    def cli(self, args: List[str]) -> None:
        """CLI interface for the consortium plugin.
        
        Args:
            args: Command line arguments
        """
        import argparse
        
        parser = argparse.ArgumentParser(description="Run a prompt through a model consortium")
        parser.add_argument("prompt", help="The prompt to process")
        parser.add_argument("-m", "--model", action="append", required=True,
                          help="Model to use (can be specified multiple times)")
        parser.add_argument("--arbiter-model", default=self.config.default_arbiter,
                          help="Model to use for arbitration")
        parser.add_argument("--confidence-threshold", type=float,
                          default=self.config.default_confidence_threshold,
                          help="Confidence threshold for accepting responses")
        parser.add_argument("--max-iterations", type=int,
                          default=self.config.default_max_iterations,
                          help="Maximum number of iteration attempts")
        
        parsed_args = parser.parse_args(args)
        
        # Run the consortium
        result = asyncio.run(self.run_consortium(
            prompt=parsed_args.prompt,
            models=parsed_args.model,
            arbiter_model=parsed_args.arbiter_model,
            confidence_threshold=parsed_args.confidence_threshold,
            max_iterations=parsed_args.max_iterations
        ))
        
        print(result.response)

    async def run_consortium(
        self,
        prompt: str,
        models: List[str],
        arbiter_model: str,
        confidence_threshold: float,
        max_iterations: int
    ) -> ConsortiumResponse:
        """Run the consortium process.
        
        Args:
            prompt: The prompt to process
            models: List of models to use
            arbiter_model: Model to use for arbitration
            confidence_threshold: Minimum confidence threshold
            max_iterations: Maximum number of iterations
            
        Returns:
            ConsortiumResponse containing the final response
        """
        if not models:
            raise ValueError("At least one model must be specified")
            
        iteration = 0
        final_response = None
        
        while iteration < max_iterations:
            # Get responses from all models
            responses = await asyncio.gather(
                *[self.get_model_response(model, prompt) for model in models]
            )
            
            # Check if any response meets the confidence threshold
            high_confidence_responses = [
                r for r in responses if r.confidence >= confidence_threshold
            ]
            
            if high_confidence_responses:
                final_response = max(high_confidence_responses,
                                   key=lambda x: x.confidence)
                break
            
            # Get arbitration if no high confidence responses
            final_response = await self.get_arbiter_decision(
                arbiter_model, responses, prompt
            )
            
            if final_response.confidence >= confidence_threshold:
                break
                
            iteration += 1
            
            # Update prompt for next iteration if needed
            prompt = f"""Previous responses were not confident enough. 
Original prompt: {prompt}
Please provide a more detailed and confident response."""
        
        # Log the interaction
        self.log_interaction(
            prompt=prompt,
            models=models,
            arbiter=arbiter_model,
            iterations=iteration + 1,
            response=final_response.response,
            confidence=final_response.confidence
        )
        
        return final_response

# Register the plugin
registry = [ConsortiumPlugin]
