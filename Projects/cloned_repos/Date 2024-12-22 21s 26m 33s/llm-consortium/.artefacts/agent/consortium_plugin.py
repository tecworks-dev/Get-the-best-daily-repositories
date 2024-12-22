import asyncio
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
import json
import csv
from datetime import datetime
import os
from enum import Enum

import llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConsortiumError(Exception):
    """Base exception for consortium-related errors."""
    pass

class ModelError(ConsortiumError):
    """Exception for model-related errors."""
    pass

class ConfigurationError(ConsortiumError):
    """Exception for configuration-related errors."""
    pass

class ResponseStatus(Enum):
    """Enumeration for response status."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class ConsortiumConfig:
    """Configuration for the consortium plugin."""
    default_arbiter_model: str = "claude-3-sonnet-20240307"
    default_confidence_threshold: float = 0.9
    default_max_iterations: int = 2
    response_timeout: float = 30.0
    history_file: str = "consortium_history.csv"
    
    def validate(self):
        """Validate configuration settings."""
        if not 0 < self.default_confidence_threshold <= 1:
            raise ConfigurationError("Confidence threshold must be between 0 and 1")
        if self.default_max_iterations < 1:
            raise ConfigurationError("Max iterations must be at least 1")
        if self.response_timeout <= 0:
            raise ConfigurationError("Response timeout must be positive")

@dataclass
class ConsortiumResponse:
    """Represents a response from a model in the consortium."""
    model_name: str
    response: str
    confidence: float
    status: ResponseStatus
    metadata: Dict[Any, Any]
    timestamp: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "model_name": self.model_name,
            "response": self.response,
            "confidence": self.confidence,
            "status": self.status.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

class ConsortiumPlugin(llm.Plugin):
    """Plugin for managing a consortium of language models with arbitration."""
    
    name = 'consortium'

    def __init__(self, config: Optional[ConsortiumConfig] = None):
        self.config = config or ConsortiumConfig()
        self.config.validate()
        self._ensure_history_file()

    def _ensure_history_file(self):
        """Ensure the history file exists with headers."""
        if not os.path.exists(self.config.history_file):
            with open(self.config.history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'prompt', 'models', 'arbiter',
                    'iterations', 'final_response', 'confidence',
                    'status', 'metadata'
                ])

    def log_interaction(self, prompt: str, models: List[str], arbiter: str,
                       iterations: int, response: ConsortiumResponse):
        """Log consortium interaction to CSV file."""
        try:
            with open(self.config.history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    prompt,
                    ','.join(models),
                    arbiter,
                    iterations,
                    response.response,
                    response.confidence,
                    response.status.value,
                    json.dumps(response.metadata)
                ])
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")

    def _parse_response_confidence(self, response_text: str) -> float:
        """Parse confidence value from response text."""
        try:
            if '<confidence>' in response_text and '</confidence>' in response_text:
                confidence_str = response_text.split('<confidence>')[1].split('</confidence>')[0]
                confidence = float(confidence_str)
                return max(0.0, min(1.0, confidence))
        except Exception as e:
            logger.warning(f"Error parsing confidence: {e}")
        return 0.0

    async def get_model_response(self, model_name: str, prompt: str) -> ConsortiumResponse:
        """Get response from a specific model with timeout."""
        try:
            model = llm.get_model(model_name)
            response_task = model.complete_async(prompt)
            response = await asyncio.wait_for(response_task, timeout=self.config.response_timeout)
            
            response_text = str(response)
            confidence = self._parse_response_confidence(response_text)
            
            return ConsortiumResponse(
                model_name=model_name,
                response=response_text,
                confidence=confidence,
                status=ResponseStatus.SUCCESS,
                metadata={'raw_response': response_text}
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting response from {model_name}")
            return ConsortiumResponse(
                model_name=model_name,
                response="Error: Response timeout",
                confidence=0.0,
                status=ResponseStatus.TIMEOUT,
                metadata={'error': 'timeout'}
            )
        except Exception as e:
            logger.error(f"Error getting response from {model_name}: {e}")
            return ConsortiumResponse(
                model_name=model_name,
                response=f"Error: {str(e)}",
                confidence=0.0,
                status=ResponseStatus.ERROR,
                metadata={'error': str(e)}
            )

    async def get_arbiter_decision(self, arbiter_model: str, 
                                 responses: List[ConsortiumResponse],
                                 original_prompt: str) -> ConsortiumResponse:
        """Get arbitration decision from the arbiter model."""
        arbiter = llm.get_model(arbiter_model)
        
        arbitration_prompt = f"""
You are an expert arbiter tasked with analyzing multiple model responses and selecting or synthesizing the best response.

Original prompt: {original_prompt}

Model Responses:
{json.dumps([r.to_dict() for r in responses], indent=2)}

Instructions:
1. Analyze each response for accuracy, relevance, and completeness
2. Either select the best response or synthesize a better one
3. Provide your confidence level (0-1) in the decision
4. Format your response as shown below

Response Format:
<confidence>0.95</confidence>
<reasoning>
Explain your decision process here
</reasoning>
<response>
Your chosen or synthesized response here
</response>
"""
        
        try:
            response = await asyncio.wait_for(
                arbiter.complete_async(arbitration_prompt),
                timeout=self.config.response_timeout
            )
            
            response_text = str(response)
            confidence = self._parse_response_confidence(response_text)
            
            return ConsortiumResponse(
                model_name=arbiter_model,
                response=response_text,
                confidence=confidence,
                status=ResponseStatus.SUCCESS,
                metadata={'raw_response': response_text}
            )
        except Exception as e:
            logger.error(f"Arbiter error: {e}")
            return ConsortiumResponse(
                model_name=arbiter_model,
                response=f"Arbitration error: {str(e)}",
                confidence=0.0,
                status=ResponseStatus.ERROR,
                metadata={'error': str(e)}
            )

    def cli(self, args: List[str]) -> None:
        """CLI interface for the consortium plugin."""
        import argparse
        
        parser = argparse.ArgumentParser(description="Run a prompt through a model consortium")
        parser.add_argument("prompt", help="The prompt to process")
        parser.add_argument("-m", "--model", action="append", required=True,
                          help="Model to use (can be specified multiple times)")
        parser.add_argument("--arbiter-model", default=self.config.default_arbiter_model,
                          help="Model to use for arbitration")
        parser.add_argument("--confidence-threshold", type=float, 
                          default=self.config.default_confidence_threshold,
                          help="Confidence threshold for accepting responses")
        parser.add_argument("--max-iterations", type=int, 
                          default=self.config.default_max_iterations,
                          help="Maximum number of iteration attempts")
        
        parsed_args = parser.parse_args(args)
        
        try:
            result = asyncio.run(self.run_consortium(
                prompt=parsed_args.prompt,
                models=parsed_args.model,
                arbiter_model=parsed_args.arbiter_model,
                confidence_threshold=parsed_args.confidence_threshold,
                max_iterations=parsed_args.max_iterations
            ))
            
            print(json.dumps(result.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Error running consortium: {e}")
            print(f"Error: {str(e)}")

    async def run_consortium(self, prompt: str, models: List[str],
                           arbiter_model: str, confidence_threshold: float,
                           max_iterations: int) -> ConsortiumResponse:
        """Run the consortium process."""
        iteration = 0
        final_response = None
        
        while iteration < max_iterations:
            logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
            
            # Get responses from all models
            responses = await asyncio.gather(
                *[self.get_model_response(model, prompt) for model in models]
            )
            
            # Filter out error responses
            valid_responses = [r for r in responses if r.status == ResponseStatus.SUCCESS]
            
            if not valid_responses:
                logger.warning("No valid responses received from models")
                break
            
            # Check for high confidence responses
            high_confidence_responses = [
                r for r in valid_responses if r.confidence >= confidence_threshold
            ]
            
            if high_confidence_responses:
                final_response = max(high_confidence_responses,
                                   key=lambda x: x.confidence)
                logger.info("Found high confidence response")
                break
            
            # Get arbitration if no high confidence responses
            final_response = await self.get_arbiter_decision(
                arbiter_model, valid_responses, prompt
            )
            
            if final_response.confidence >= confidence_threshold:
                logger.info("Arbiter provided high confidence response")
                break
                
            iteration += 1
            
            # Update prompt for next iteration if needed
            prompt = f"""Previous responses were not confident enough. 
Original prompt: {prompt}
Please provide a more detailed and confident response."""
        
        if final_response is None:
            final_response = ConsortiumResponse(
                model_name="consortium",
                response="Failed to get satisfactory response after all iterations",
                confidence=0.0,
                status=ResponseStatus.ERROR,
                metadata={'error': 'max_iterations_reached'}
            )
        
        # Log the interaction
        self.log_interaction(
            prompt=prompt,
            models=models,
            arbiter=arbiter_model,
            iterations=iteration + 1,
            response=final_response
        )
        
        return final_response

# Register the plugin
registry = [ConsortiumPlugin]
